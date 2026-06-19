# stable-offline-rl

## CleanDiffuser DQL

DQL uses CleanDiffuser commit `05f17fc9dbeae7c19a5e264632c9ae9aaac5994e`. Install it without dependency resolution because CleanDiffuser's package metadata pins old versions of Gym, MuJoCo, NumPy, and Torch that are incompatible with the `mujocold` environment:

```bash
conda activate mujocold
python -m pip install --no-deps einops==0.8.1

cd /home/shekhe
git clone https://github.com/CleanDiffuserTeam/CleanDiffuser.git
git -C /home/shekhe/CleanDiffuser checkout 05f17fc9dbeae7c19a5e264632c9ae9aaac5994e
python -m pip install --editable /home/shekhe/CleanDiffuser --no-deps --no-build-isolation
```

The integration currently supports `HalfCheetah-v5`. It uses CleanDiffuser's five-step DDPM actor, twin critic, EMA updates, and 50-candidate inference defaults. Training length is `epoch * step_per_epoch`; the upstream two-million-step schedule corresponds to `--epoch 2000 --step-per-epoch 1000`.

CleanDiffuser's episodic reward-range normalization is disabled. Generated datasets sample individual transitions from collected trajectories, so they do not retain the complete episodes needed to calculate that normalization correctly. This keeps preprocessing consistent between generated and Minari datasets.

```bash
cd /home/shekhe/stable-offline-rl
python sweep.py \
  --env HalfCheetah-v5 \
  --dataset-source minari \
  --algos dql \
  --output-dir /home/shekhe/train_dir/stable_offline_rl \
  --device cuda \
  --epoch 2000 \
  --step-per-epoch 1000 \
  --batch-size 256 \
  --split-level episode \
  --eval
```
