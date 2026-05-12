from pathlib import Path
try:
    import gymnasium as gym
except ImportError:
    import gym
import numpy as np
import numpy.random as rand
import torch
from torch.utils.data import Dataset, DataLoader
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3 import SAC
from sb3_contrib import TQC
from stable_baselines3 import PPO
import hard_stable

def CollectTrajs(env_name: str,
                         policy_path: str,
                         max_timesteps: int = 300,
                         num_trajectories: int = 1,
                         noised: int = 0,
                         noise_scale: float = 0.0,
                         prop_noised: float = 0.5,
                         deterministic: bool = True,
                         seed: int = None):
    """
    Collects trajectories by rolling out a pretrained expert policy in a Gymnasium environment.

    Args:
        env_name (str): the Gymnasium environment ID.
        policy_path (str): path to the saved expert policy (Stable Baselines3 format).
        max_timesteps (int): maximum steps per trajectory.
        num_trajectories (int): number of trajectories to collect.
        noise_scale (float): covariance-scaling factor of unit-scaled Gaussian noise injection.
        deterministic (bool): whether to use deterministic actions (if expert has exploration mode).
        seed (int): random seed for environment.

    Returns:
        List[dict]: a list of trajectories, each containing observations, actions, rewards, dones, infos.
    """
    # policy: BaseAlgorithm = BaseAlgorithm.load(policy_path)
    if env_name == 'Humanoid-v5':
        env = gym.make(env_name)
        policy = TQC.load(policy_path)
    # Any other v2/v3/v4/v5 continuous env uses SAC
    elif env_name == "Swimmer-v5":
        env = gym.make(env_name)
        policy = PPO.load(policy_path)
    elif any(env_name.endswith(f'-v{v}') for v in (2,3,4,5)):
        env = gym.make(env_name)
        policy = SAC.load(policy_path)
    # The hard_stable custom env
    elif env_name == 'hard_stable':
        expert_path = Path('./experts/hard_stable_perturb.pt')
        if expert_path.exists():
            state_dict = torch.load(expert_path)
        else:
            # first time: create and save
            env = hard_stable.create_gym_environment(
                d=4, pair_first=True, pytorch_seed=seed
            )
            state_dict = env.perturbation_model.state_dict()
            torch.save(state_dict, expert_path)
        env = hard_stable.create_gym_environment(
            d=4, pair_first=True, pytorch_seed=seed, state_dict=state_dict
        )
        policy = hard_stable.EmbedExpertTorch(
            K=env.K.copy(), model_state_dict=state_dict, tau=env.tau, d_model_input=4
        )
    else:
        raise ValueError(f"Unknown environment name: {env_name}")
    
    if seed is not None:
        env.reset(seed=seed)


    trajectories = []
    action_dim = env.action_space.shape[0]
    if noise_scale > 0:
        noise = rand.randn(num_trajectories, max_timesteps, action_dim) * (noise_scale/np.sqrt(action_dim))
        for ep in range(num_trajectories):
            obs, info = env.reset()
            traj = {
                'obs': [],
                'acts': [],
                'rewards': [],
                'dones': [],
                # 'infos': []
            }
            for t in range(max_timesteps):
                traj['obs'].append(obs.copy())
                action, _ = policy.predict(obs, deterministic=deterministic)
                # noise injection
                if noised == 1:
                    # if noised=1, then record noisy action
                    action = action + noise[ep,t]
                    traj['acts'].append(action)
                else:
                    # else, record clean action, then execute noisy action
                    traj['acts'].append(action)
                    action = action + noise[ep,t]
                # only collect prop_noised * num_traj noise-injected trajectories
                if ep < int(prop_noised * num_trajectories):
                    obs, reward, done, truncated, _ = env.step(action)
                else:
                    obs, reward, done, truncated, _ = env.step(action)
                traj['rewards'].append(reward)
                traj['dones'].append(done or truncated)
                # traj['infos'].append(info)
                if done or truncated:
                    break
            trajectories.append(traj)
    else:
        for ep in range(num_trajectories):
            obs, info = env.reset()
            traj = {
                'obs': [],
                'acts': [],
                'rewards': [],
                'dones': [],
                # 'infos': []
            }
            for t in range(max_timesteps):
                traj['obs'].append(obs.copy())
                action, _ = policy.predict(obs, deterministic=deterministic)
                traj['acts'].append(action)
                obs, reward, done, truncated, _ = env.step(action)
                traj['rewards'].append(reward)
                traj['dones'].append(done or truncated)
                # traj['infos'].append(info)
                if done or truncated:
                    break
            trajectories.append(traj)
    env.close()
    return trajectories

class TrajDataset(Dataset):
    """
    Convert trajectory list to torch dataset.
    Each sample is:
      - input: observation at time t
      - label: stack of action(s) from t to t+chunk_len-1
    """
    def __init__(self, trajectories, chunk_len: int = 1):
        self.chunk_len = chunk_len
        obs_list = []
        act_list = []
        for traj in trajectories:
            observations = traj['obs']
            actions = traj['acts']
            T = len(actions)
            for t in range(T - chunk_len + 1):
                obs_list.append(observations[t])
                if chunk_len == 1:
                    act_list.append(actions[t])
                else:
                    act_list.append(np.stack(actions[t:t+chunk_len], axis=0))
        # Convert to tensors but keep on CPU
        self.inputs = torch.tensor(np.stack(obs_list), dtype=torch.float32)
        self.labels = torch.tensor(np.stack(act_list), dtype=torch.float32)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

def TrajLoader(trajs, chunk_len: int = 1, batch_size: int = 100, device: str = 'cpu'):
    dataset = TrajDataset(trajs, chunk_len=chunk_len)
    return DataLoader(dataset, batch_size, shuffle=True, num_workers=0, pin_memory=(device != 'cpu'))
