import os
import json
import itertools
import argparse

import numpy as np
import torch
import rollout

from rollout import CollectTrajs, TrajLoader
from train import MLP, MLP_no_batch_bias, train
from eval import evaluate


def sweep(env_name,
          expert_path,
          output_dir,
          model_dir,
          default_config,
          traj_list,
          noise_list,
          prop_list,
          chunk_list,
          rng):
    stats_dir = os.path.join(output_dir, env_name)
    os.makedirs(stats_dir, exist_ok=True)
    model_dir = os.path.join(model_dir, env_name)
    os.makedirs(model_dir, exist_ok=True)

    print(f"Saving results to {stats_dir}")

    device = default_config['device']
    if device == 'mps':
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not "
                    "built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine.")
            device = 'cpu'
    if device == 'cuda':
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print (f"Using CUDA device: {device}")
        else:
            print ("CUDA device not found.")
            device = 'cpu'

    # grid over configurations
    for num_traj, noise_scale, chunk_len, prop_noised in itertools.product(traj_list, noise_list, chunk_list, prop_list):
        cfg = default_config.copy()
        cfg.update({
            'num_trajectories': num_traj,
            'noise_scale': noise_scale,
            'chunk_len': chunk_len,
            'prop_noised': prop_noised
        })
        if default_config['noised'] == 0:
            tag = f"{env_name}_traj{num_traj}_noise{noise_scale}_chunk{chunk_len}_prop{prop_noised}"
        else:
            tag = f"{env_name}_traj{num_traj}_noise{noise_scale}_noised_chunk{chunk_len}_prop{prop_noised}"
        print(f"Training model with config: {tag}")

        # 1) Collect trajectories
        trajectories = CollectTrajs(
            env_name,
            expert_path,
            noised=default_config['noised'],
            noise_scale=cfg['noise_scale'],
            prop_noised=cfg['prop_noised'],
            max_timesteps=cfg['max_timesteps'],
            num_trajectories=cfg['num_trajectories'],
            deterministic=True
        )
        # traj_file = os.path.join(output_dir, f"trajectories_{tag}.npz")
        # Save raw trajectories
        # np.savez(traj_file, **{k: trajectories for k, trajectories in [('traj', trajectories)]})

        # 2) Train model
        # infer dims
        sample_obs = np.array(trajectories[0]['obs'])
        obs_dim = sample_obs.shape[1]
        act_dim = np.array(trajectories[0]['acts'][0]).shape[0]
        
        if cfg['batch_bias'] == 1:
            model = MLP(obs_dim, act_dim, chunk_len)
        else:
            model = MLP_no_batch_bias(obs_dim, act_dim, chunk_len)

        # build a DataLoader from trajectories
        loader = TrajLoader(
            trajectories,
            chunk_len=chunk_len,
            batch_size=cfg['batch_size'],
            device=device
        )

        trained_model, train_losses = train(
            model,
            loader,
            device=device,
            lr=cfg['lr'],
            epochs=cfg['epochs']
        )
        model_file = os.path.join(model_dir, f"model_{tag}.pt")
        torch.save(trained_model.state_dict(), model_file)
        
        # Save training losses
        loss_file = os.path.join(stats_dir, f"train_losses_{tag}.npy")
        np.save(loss_file, train_losses)

        # 3) Evaluate
        trained_model.eval()
        results = evaluate(
            env_name,
            expert_path,
            trained_model,
            max_timesteps=cfg['max_timesteps'],
            num_traj=cfg['eval_num_traj'],
            store_instance_reward=cfg['store_instance_reward'],
            chunk_len=chunk_len,
            device=device,
            rng=rng
        )
        eval_file = os.path.join(stats_dir, f"eval_{tag}.npz")
        np.savez(eval_file, **results)

        # save config
        # with open(os.path.join(output_dir, f"config_{tag}.json"), 'w') as f:
        #     json.dump(cfg, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sweep behavior cloning over config grid.")
    parser.add_argument('--env', type=str, nargs='+', required=True, help='List of Gym environment IDs')
    parser.add_argument('--expert', type=str, default='experts', help='Path to expert policy directory')
    parser.add_argument('--output_dir', type=str, default='stats', help='Path for saved statistics')
    parser.add_argument('--model_dir', type=str, default='learned_models', help='Path for learned models')
    parser.add_argument('--num_traj', type=int, nargs='+', default=[5, 10, 20, 30, 40, 50], help='List of num_trajectories')
    parser.add_argument('--noised', type=int, default=0, help='Whether to record noisy actions')
    parser.add_argument('--noise_scale', type=float, nargs='+', default=[0.0, 0.001, 0.01, 0.05, 0.1, 0.5], help='List of expert noise scales')
    parser.add_argument('--prop_noised', type=float, nargs='+', default=[0.4], help='Lißst of prop_noised')
    parser.add_argument('--chunk_len', type=int, nargs='+', default=[1], help='List of chunk lengths')
    parser.add_argument('--max_timesteps', type=int, default=300, help='Maximum number of timesteps')
    parser.add_argument('--batch_bias', type=int, default=1, help='Batch norm and bias in model, default 1 (True)')
    parser.add_argument('--store_instance_reward', type=int, default=1, help='Record per time-step reward, default 1 (True)')
    args = parser.parse_args()

    # set up base numpy random Generator
    rng = np.random.default_rng(117)

    # default parameters
    default_config = {
        'max_timesteps': args.max_timesteps,
        'seed': None,
        'batch_size': 64,
        'num_workers': 0,
        'pin_memory': False,
        'lr': 0.001,
        'epochs': 4000,
        'device': 'cpu',
        'noised': args.noised,
        'store_instance_reward': args.store_instance_reward,
        # 'device': 'mps',
        # 'device': 'cuda',
        'eval_num_traj': 100,     # number of trajectories for evaluation
        'num_trajectories': args.num_traj,   # default trajectories for training
        'noise_scale': args.noise_scale,     # default expert action noise
        'prop_noised': args.prop_noised,     # default proportion of noise-injected trajectories
        'chunk_len': args.chunk_len,         # default action chunk length
        'batch_bias': args.batch_bias        # default batch norm and bias in model
    }

    # Run sweep for each environment
    for env_name in args.env:
        if env_name == 'hard_stable':
            expert_path = os.path.join(args.expert, 'hard_stable_perturb.pt')
            if not os.path.exists(expert_path):
                print(f"Warning: Expert model file {expert_path} does not exist. Will create new one.")
            print(f"\nRunning sweep for environment: {env_name}")
            sweep(
                env_name=env_name,
                expert_path=expert_path,
                output_dir=args.output_dir,
                model_dir=args.model_dir,
                default_config=default_config,
                traj_list=args.num_traj,
                noise_list=args.noise_scale,
                prop_list=args.prop_noised,
                chunk_list=args.chunk_len,
                rng=rng
            )
        else:
            expert_path = os.path.join(args.expert, f'{env_name}.zip')
            if not os.path.exists(expert_path):
                print(f"Warning: Expert policy file {expert_path} does not exist. Skipping {env_name}.")
                continue
            else:
                print(f"\nRunning sweep for environment: {env_name}")
                sweep(
                    env_name=env_name,
                    expert_path=expert_path,
                    output_dir=args.output_dir,
                    model_dir=args.model_dir,
                    default_config=default_config,
                    traj_list=args.num_traj,
                    noise_list=args.noise_scale,
                    prop_list=args.prop_noised,
                    chunk_list=args.chunk_len,
                    rng=rng
                )
