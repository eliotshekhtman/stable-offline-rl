# stable-offline-rl

## Saved artifacts

Runs are independent of the working directory and are saved inside this repository:

```text
datasets/<environment>/<dataset-name>/<dataset-timestamp>/
trained/<environment>/<algorithm>_<dataset-name>/<training-timestamp>/
evals/<environment>/<algorithm>_<dataset-name>/<training-timestamp>/
evals/<environment>/plots/
```

Each training manifest records the dataset identity and all algorithm-relevant training arguments. Before training, `sweep.py` searches the timestamped variants under the corresponding run name and reuses the newest complete run with an identical training schema. Evaluation arguments are not part of that schema, so adding `--eval` or changing an evaluation setting reloads the existing model instead of retraining it. A different training schema creates another timestamped variant and leaves prior variants intact. Dataset reuse is automatic; `--output-dir`, `--reuse-datasets`, and `--overwrite` are no longer used.

Dataset splits are reused by the same rule. Paths in new metadata and manifests are absolute. Runs previously written under `/home/shekhe/train_dir` are not searched or modified.

`DATASET_SCHEMA_VERSION` and `TRAINING_SCHEMA_VERSION` in `sweep.py` invalidate cached artifacts when dataset conversion or training behavior changes without a corresponding CLI change. Increment the relevant value when making such a change; evaluation-only edits require neither increment.

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

The integration supports flat continuous-control environments, including Gymnasium MuJoCo tasks such as `HalfCheetah-v5` and `Humanoid-v5`. Actions are normalized to `[-1, 1]` inside DQL and converted back to the environment's native bounds for execution. It uses CleanDiffuser's five-step DDPM actor, twin critic, EMA updates, and 50-candidate inference defaults. Training length is `epoch * step_per_epoch`; the upstream two-million-step schedule corresponds to `--epoch 2000 --step-per-epoch 1000`.

Published CleanDiffuser defaults are used when available: Q-selection weight temperature 50 for HalfCheetah, 300 for Walker2d, 100 for Hopper medium/replay, and 8 for Hopper medium-expert, with `eta=1`. Other environments use 50 and record that it is a fallback. `--dql-weight-temperature` and `--dql-eta` override these choices.

`--dql-reward-normalization auto` applies CleanDiffuser-style episodic return-range scaling to the training episodes of Minari datasets split by episode. It leaves rewards unchanged for generated datasets and transition-level splits, which do not necessarily retain complete episodes. Every resolved DQL setting and its source are saved in `run_manifest.json`.

```bash
cd /home/shekhe/stable-offline-rl
python sweep.py \
  --env HalfCheetah-v5 \
  --dataset-source minari \
  --algos dql \
  --device cuda \
  --epoch 2000 \
  --step-per-epoch 1000 \
  --batch-size 256 \
  --split-level episode \
  --eval
```

## Evaluation over training

Every run saves policy checkpoints at approximately 0, 10, 20, ..., 100 percent of policy training. Milestones that fall in the same epoch are collapsed. Checkpoints live under `checkpoint/step_<gradient_step>/`; fixed model-based dynamics are stored once at step zero, while RAMBO saves its changing dynamics with each checkpoint.

Passing `--eval` runs both final-policy evaluation and checkpoint-history evaluation before plotting. In addition to reward and the model-based next-state/Jacobian metrics, evaluation reports:

- **Global stability:** trajectories from different reset states are phase-aligned by the offset minimizing second-half state distance. This alignment is an explicit heuristic for periodic motion.
- **Local stability:** held-out states are paired with small perturbations in reconstructible `qpos/qvel` coordinates, without phase alignment.
- **Empirical `(C, rho)`:** standardized distances are bounded at every observed timestep by `C * rho**t`. Values of `rho` are not constrained below one.
- **State and state-action OOD ratios:** mean rollout-to-training k-nearest-neighbor distance divided by the corresponding held-out-to-training distance. A ratio near one means the rollout is about as far from training data as held-out data is.

Dynamics mismatch and finite-difference Jacobian mismatch are evaluated only for the final model. Reward, stability, and OOD metrics are evaluated at every saved policy checkpoint. Raw arrays are saved under the matching `evals/<environment>/<run-name>/<training-timestamp>/` directory, and plots for the runs selected by the current sweep are written under `evals/<environment>/plots/`.
