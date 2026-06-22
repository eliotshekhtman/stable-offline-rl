# Tasks:
# - Adapt CleanDiffuser's DQL actor and critic to OfflineRL-Kit's policy interface.
# - Train and checkpoint DQL without OfflineRL-Kit's per-epoch environment evaluation.

import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.nn_diffusion import DQLMlp
from cleandiffuser.utils import DQLCritic, FreezeModules
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.policy import BasePolicy
from offlinerlkit.utils.logger import Logger


CLEANDIFFUSER_COMMIT = "05f17fc9dbeae7c19a5e264632c9ae9aaac5994e"


class DQLPolicy(BasePolicy):
    def __init__(
        self,
        obs_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
        total_steps: int,
        device: str,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.action_dim = len(action_low)
        self.discount = 0.99
        self.eta = 1.0
        self.diffusion_steps = 5
        self.num_candidates = 50
        self.weight_temperature = 50.0
        self.inference_temperature = 0.5

        diffusion_model = DQLMlp(obs_dim, self.action_dim, emb_dim=64, timestep_emb_type="positional")
        self.actor = DiscreteDiffusionSDE(
            diffusion_model,
            IdentityCondition(dropout=0.0),
            predict_noise=True,
            optim_params={"lr": 3e-4},
            x_min=torch.as_tensor(action_low, dtype=torch.float32, device=self.device)[None],
            x_max=torch.as_tensor(action_high, dtype=torch.float32, device=self.device)[None],
            diffusion_steps=self.diffusion_steps,
            ema_rate=0.995,
            device=self.device,
        )
        # Register CleanDiffuser's modules so the normal PyTorch state_dict contains them.
        self.diffusion_model = self.actor.model
        self.diffusion_model_ema = self.actor.model_ema
        self.critic = DQLCritic(obs_dim, self.action_dim, hidden_dim=256).to(self.device)
        self.critic_target = deepcopy(self.critic).requires_grad_(False).eval()
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.actor_scheduler = CosineAnnealingLR(self.actor.optimizer, T_max=total_steps)
        self.critic_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=total_steps)

        self.register_buffer("obs_mean", torch.as_tensor(obs_mean, dtype=torch.float32, device=self.device))
        self.register_buffer("obs_std", torch.as_tensor(obs_std, dtype=torch.float32, device=self.device))
        self.register_buffer("gradient_step", torch.zeros((), dtype=torch.long, device=self.device))

    def train(self, mode: bool = True):
        self.training = mode
        self.diffusion_model.train(mode)
        self.diffusion_model_ema.eval()
        self.critic.train(mode)
        self.critic_target.eval()
        return self

    def eval(self):
        return self.train(False)

    def learn(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        obs = self._normalize(batch["observations"])
        next_obs = self._normalize(batch["next_observations"])
        actions = batch["actions"]
        # Keep rewards unnormalized because generated datasets do not retain complete episodes.
        rewards = batch["rewards"]
        terminals = batch["terminals"]
        batch_size = len(obs)

        current_q1, current_q2 = self.critic(obs, actions)
        with torch.no_grad():
            next_actions = self._sample(next_obs, batch_size, use_ema=True, temperature=1.0)
            target_q = torch.min(*self.critic_target(next_obs, next_actions))
            target_q = rewards + (1.0 - terminals) * self.discount * target_q
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        bc_loss = self.actor.loss(actions, obs)
        new_actions = self._sample(obs, batch_size, use_ema=False, temperature=1.0, requires_grad=True)
        with FreezeModules([self.critic]):
            q1, q2 = self.critic(obs, new_actions)
        if np.random.uniform() > 0.5:
            q_loss = -q1.mean() / q2.abs().mean().detach()
        else:
            q_loss = -q2.mean() / q1.abs().mean().detach()
        actor_loss = bc_loss + self.eta * q_loss
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.actor_scheduler.step()
        self.critic_scheduler.step()
        step = int(self.gradient_step.item())
        if step % 5 == 0:
            if step >= 1000:
                self.actor.ema_update()
            with torch.no_grad():
                for parameter, target_parameter in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_parameter.data.mul_(0.995).add_(parameter.data, alpha=0.005)
        self.gradient_step.add_(1)

        return {
            "loss/bc": bc_loss.item(),
            "loss/q": q_loss.item(),
            "loss/critic": critic_loss.item(),
            "target_q_mean": target_q.mean().item(),
        }

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        del deterministic  # CleanDiffuser DQL inference is stochastic by design.
        observations = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        observations = self._normalize(observations)
        batch_size = len(observations)
        candidates_obs = observations.repeat_interleave(self.num_candidates, dim=0)
        candidates = self._sample(
            candidates_obs,
            batch_size * self.num_candidates,
            use_ema=True,
            temperature=self.inference_temperature,
        )
        q_values = self.critic_target.q_min(candidates_obs, candidates).view(batch_size, self.num_candidates)
        weights = torch.softmax(q_values * self.weight_temperature, dim=1)
        indices = torch.multinomial(weights, 1).squeeze(1)
        candidates = candidates.view(batch_size, self.num_candidates, self.action_dim)
        return candidates[torch.arange(batch_size, device=self.device), indices].cpu().numpy()

    def _normalize(self, observations: torch.Tensor) -> torch.Tensor:
        return (observations - self.obs_mean) / self.obs_std

    def _sample(
        self,
        observations: torch.Tensor,
        sample_count: int,
        use_ema: bool,
        temperature: float,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        prior = torch.zeros((sample_count, self.action_dim), device=self.device)
        actions, _ = self.actor.sample(
            prior,
            solver="ddpm",
            n_samples=sample_count,
            sample_steps=self.diffusion_steps,
            use_ema=use_ema,
            temperature=temperature,
            condition_cfg=observations,
            w_cfg=1.0,
            requires_grad=requires_grad,
        )
        return actions


def build_dql_policy(buffer: ReplayBuffer, action_low: np.ndarray, action_high: np.ndarray, total_steps: int, device: str) -> DQLPolicy:
    obs_mean = buffer.observations.mean(axis=0, keepdims=True)
    obs_std = buffer.observations.std(axis=0, keepdims=True)
    obs_std[obs_std == 0.0] = 1.0
    return DQLPolicy(
        obs_dim=buffer.observations.shape[1],
        action_low=action_low,
        action_high=action_high,
        obs_mean=obs_mean,
        obs_std=obs_std,
        total_steps=total_steps,
        device=device,
    )


def train_dql(
    policy: DQLPolicy,
    buffer: ReplayBuffer,
    logger: Logger,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    checkpoint_epochs: list[int],
) -> None:
    start_time = time.time()
    total_steps = 0
    for epoch in range(1, epochs + 1):
        policy.train()
        progress = tqdm(range(steps_per_epoch), desc=f"Epoch #{epoch}/{epochs}")
        for _ in progress:
            losses = policy.learn(buffer.sample(batch_size))
            progress.set_postfix(**losses)
            for key, value in losses.items():
                logger.logkv_mean(key, value)
            total_steps += 1
        logger.set_timestep(total_steps)
        logger.dumpkvs()
        torch.save(policy.state_dict(), f"{logger.checkpoint_dir}/policy.pth")
        if epoch in checkpoint_epochs:
            checkpoint_dir = Path(logger.checkpoint_dir) / f"step_{total_steps}"
            checkpoint_dir.mkdir(exist_ok=True)
            torch.save(policy.state_dict(), checkpoint_dir / "policy.pth")

    logger.log(f"total time: {time.time() - start_time:.2f}s")
    torch.save(policy.state_dict(), f"{logger.model_dir}/policy.pth")
    logger.close()
