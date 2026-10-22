"""Optional action-chunk means with decaying, spectrally normalized slopes."""

import math
from numbers import Integral, Real

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm

from offlinerlkit.policy import TD3BCPolicy


CHUNK_POLICY_MODES = ("standard", "decay-sn")
CHUNK_POLICY_ALGOS = ("td3bc", "iql", "mopo", "mobile")


def _validate_rho(rho):
    if (
        isinstance(rho, bool)
        or not isinstance(rho, Real)
        or not math.isfinite(rho)
        or not 0 < rho <= 1
    ):
        raise ValueError("chunk_policy_rho must be finite and satisfy 0 < rho <= 1.")
    return float(rho)


def _validate_c(c):
    if isinstance(c, bool) or not isinstance(c, Real) or not math.isfinite(c) or c <= 0:
        raise ValueError("chunk_policy_c must be finite and positive.")
    return float(c)


def validate_chunk_policy_config(config, algo):
    """Validate the complete, versioned metadata for an enabled chunk actor."""
    if not isinstance(config, dict) or set(config) not in (
        {"version", "type", "rho"}, {"version", "type", "rho", "c"},
    ):
        raise ValueError("chunk_policy config must contain version, type, rho, and optionally c.")
    if (
        isinstance(config["version"], bool)
        or not isinstance(config["version"], Integral)
        or config["version"] not in (1, 2)
        or (config["version"] == 2 and algo == "td3bc")
    ):
        raise ValueError("Unsupported chunk_policy config version for this algorithm.")
    if config["type"] != "decay-sn":
        raise ValueError("Unsupported chunk_policy config type; expected 'decay-sn'.")
    if algo not in CHUNK_POLICY_ALGOS:
        raise ValueError(f"decay-sn chunk policy is unsupported for algorithm {algo!r}.")
    canonical = {"version": int(config["version"]), "type": "decay-sn", "rho": _validate_rho(config["rho"])}
    c = _validate_c(config.get("c", 1.0))
    # Keep default schemas identical so existing runs remain reusable.
    if c != 1.0:
        canonical["c"] = c
    return canonical


def get_chunk_policy_config(args, algo):
    # Reload the recorded architecture, not the current training default.
    config = getattr(args, "chunk_policy_config", None)
    if config is not None:
        return validate_chunk_policy_config(config, algo)
    mode = getattr(args, "chunk_policy", "standard")
    rho = getattr(args, "chunk_policy_rho", None)
    c = getattr(args, "chunk_policy_c", None)
    if mode not in CHUNK_POLICY_MODES:
        raise ValueError(f"Unsupported chunk_policy mode {mode!r}; expected {CHUNK_POLICY_MODES}.")
    if mode == "standard":
        if rho is not None:
            raise ValueError("chunk_policy_rho requires --chunk-policy decay-sn.")
        if c is not None:
            raise ValueError("chunk_policy_c requires --chunk-policy decay-sn.")
        return None
    version = 1 if algo == "td3bc" else 2
    return validate_chunk_policy_config({
        "version": version, "type": mode, "rho": rho, "c": 1.0 if c is None else c,
    }, algo)


def fit_observation_stats(observations):
    """Match TD3BC's fixed statistics on real training chunk-start observations."""
    observations = np.asarray(observations, dtype=np.float32)
    if (
        observations.ndim != 2 or 0 in observations.shape
        or not np.isfinite(observations).all()
    ):
        raise ValueError("Observation normalization requires a nonempty finite observation matrix.")
    return observations.mean(axis=0), observations.std(axis=0) + np.float32(1e-3)


class NormalizedObservationBackbone(nn.Module):
    """Standardize observations while preserving a critic's appended actions."""

    def __init__(self, backbone, mean, std):
        super().__init__()
        self.backbone = backbone
        self.output_dim = backbone.output_dim
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).detach().clone())
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).detach().clone())
        self._validate_stats()

    def _validate_stats(self):
        if (
            self.mean.ndim != 1 or self.mean.numel() == 0
            or self.std.shape != self.mean.shape
            or not torch.isfinite(self.mean).all()
            or not torch.isfinite(self.std).all()
            or not (self.std > 0).all()
        ):
            raise ValueError("Observation mean/std must be finite matching vectors with positive std.")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )
        try:
            self._validate_stats()
        except ValueError as error:
            error_msgs.append(f"{prefix}{error}")

    def forward(self, features):
        obs_dim = self.mean.numel()
        observations = (features[..., :obs_dim] - self.mean) / self.std
        if features.shape[-1] != obs_dim:
            observations = torch.cat((observations, features[..., obs_dim:]), dim=-1)
        return self.backbone(observations)


class DecayingChunkMean(nn.Module):
    """Flatten b_k + C rho**k W_k h in the environment's chunk order."""

    def __init__(self, latent_dim, action_dim, chunk_length, rho, c=1.0):
        super().__init__()
        for name, value in (("latent_dim", latent_dim), ("action_dim", action_dim),
                            ("chunk_length", chunk_length)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if action_dim % chunk_length:
            raise ValueError("action_dim must be divisible by chunk_length.")
        self.chunk_length = int(chunk_length)
        self.primitive_dim = int(action_dim // chunk_length)
        self.rho = _validate_rho(rho)
        c = _validate_c(c)
        self.heads = nn.ModuleList(
            nn.Linear(int(latent_dim), self.primitive_dim)
            for _ in range(self.chunk_length)
        )
        self.register_buffer("decay", self.rho ** torch.arange(self.chunk_length))
        # One gain per output head, not per layer. Persist it in the existing
        # buffer, preserving legacy checkpoint keys and the exact C=1 path.
        if c != 1.0:
            self.decay.mul_(c)

    def forward(self, features):
        return torch.cat([
            head.bias + self.decay[k] * F.linear(features, head.weight, None)
            for k, head in enumerate(self.heads)
        ], dim=-1)


class DecayingActor(nn.Module):
    """TD3 actor interface with a separate decaying mean for each chunk position."""

    def __init__(self, backbone, action_dim, max_action, device, chunk_length, rho, c=1.0):
        super().__init__()
        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        self.last = DecayingChunkMean(
            backbone.output_dim, action_dim, chunk_length, rho, c=c
        ).to(device)
        self._max = max_action

    def forward(self, obs):
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        return self._max * torch.tanh(self.last(self.backbone(obs)))


def _chunk_linears(actor):
    mean = actor.last if hasattr(actor, "last") else actor.dist_net.mu
    if not isinstance(mean, DecayingChunkMean):
        raise ValueError("Chunk spectral normalization requires a DecayingChunkMean.")
    return [module for module in actor.backbone.modules() if isinstance(module, nn.Linear)] + list(mean.heads)


def apply_chunk_spectral_norm(actor):
    """Attach unit-target spectral normalization after raw-weight initialization.

    PyTorch uses 15 initial power iterations and one per training access. The
    resulting spectral norm is an estimate, not an exact Lipschitz certificate.
    Distribution scale heads and biases are deliberately left unconstrained.
    """
    for layer in _chunk_linears(actor):
        spectral_norm(layer)


@torch.no_grad()
def refresh_chunk_spectral_norm(actor):
    """Refresh target estimates from target weights without changing actor mode."""
    for layer in _chunk_linears(actor):
        parametrization = layer.parametrizations.weight[0]
        was_training = parametrization.training
        try:
            parametrization.train(True)
            parametrization(layer.parametrizations.weight.original)
        finally:
            parametrization.train(was_training)


class ChunkTD3BCPolicy(TD3BCPolicy):
    def _sync_weight(self):
        super()._sync_weight()
        refresh_chunk_spectral_norm(self.actor_old)
