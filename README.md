# stable-offline-rl

## Saved artifacts

Runs are independent of the working directory and are saved inside this repository:

```text
datasets/<environment>/<dataset-name>/<dataset-timestamp>/
trained/<environment>/<algorithm>_chunk<length>_<dataset-name>/<training-timestamp>/
evals/<environment>/<algorithm>_chunk<length>_<dataset-name>/<training-timestamp>/
evals/<environment>/plots/
```

Each training manifest records the dataset identity and all algorithm-relevant training arguments. Before training, `sweep.py` searches the timestamped variants under the corresponding run name and reuses the newest complete run with an identical training schema. Evaluation arguments are not part of that schema, so adding `--eval` or changing an evaluation setting reloads the existing model instead of retraining it. A different training schema creates another timestamped variant and leaves prior variants intact. Dataset reuse is automatic; `--output-dir`, `--reuse-datasets`, and `--overwrite` are no longer used.

Training normally displays one progress bar over policy-training epochs. Pass `--quiet` to suppress routine terminal output, including that bar. CSV, TensorBoard, checkpoint, and text-log artifacts are still written in either mode; warnings, exceptions, and tracebacks remain visible.

Dataset splits are reused by the same rule. Paths in new metadata and manifests are absolute. Runs previously written under `/home/shekhe/train_dir` are not searched or modified.

Premade-data sweeps use every matching dataset by default. Pass `--dataset mh` for one Robomimic dataset type, or `--dataset medium-v0` (equivalently its full ID, such as `mujoco/halfcheetah/medium-v0`) for one Minari dataset.

All datasets are split by complete episode. Minari episodes, robomimic demonstrations, and generated rollouts share the same ordered transition schema: every `episode_id` occupies one contiguous block, and consecutive entries within that block are consecutive environment transitions. Generated `--num-samples` values are minimum transition counts. Source proportions allocate whole trajectories, and complete trajectories are added until that minimum is reached. Metadata records requested proportions plus realized trajectory and transition counts and fractions.

Robomimic Lift uses continuing-task semantics: success annotations are not Bellman terminals, demonstration endpoints are timeouts, and model-generated transitions do not terminate at success. These datasets and runs use a `_continuing` tag so they cannot be confused with or reuse older terminal-Lift artifacts. Other environments retain their existing termination semantics.

Generated datasets can mix clean expert, noisy expert, and random-action episodes. Pass `--composition CLEAN_EXPERT NOISY_EXPERT`; random data fills the remaining proportion. Repeat the argument to sweep specified compositions without taking a Cartesian product between their two values. `--noise-scale` applies only to the noisy expert component, while clean expert actions use zero noise. The default composition is `1 0`.

MOBILE on `Reacher-v5` uses a shifted zero clamp for its critic targets. `--mobile-return-shift D` defaults to `30`: the critics are initialized with `+D`, each macro reward receives `(1 - macro_discount) * D`, and clamping at zero in shifted units is equivalent to a target floor of `-D` in the original units. The option is ignored by other environments, whose MOBILE behavior is unchanged.

`DATASET_SCHEMA_VERSION` and `TRAINING_SCHEMA_VERSION` in `sweep.py` invalidate cached artifacts when dataset conversion or training behavior changes without a corresponding CLI change. Increment the relevant value when making such a change; evaluation-only edits require neither increment.

## Action chunks

Pass one or more values to `--chunk-lengths` to train the Cartesian product of algorithms and chunk lengths. For example, `--algos bc cql --chunk-lengths 1 4 8` trains six policies. Length one is the ordinary one-action baseline. Every length has a separate run directory, manifest, evaluation, and plot label.

The saved datasets remain primitive one-step transitions and are reused across all chunk lengths. After the episode-level train/test split, both training and evaluation derive chunks identically from every source. For an episode with transitions indexed `0, ..., T - 1`, a length-`l` dataset contains the stride-one windows starting at `0, ..., T - l`. A window maps the starting observation to the flattened, time-major action vector `[a_t, ..., a_(t+l-1)]`, the observation after the last action, and the reward

```text
r_t + 0.99 r_(t+1) + ... + 0.99^(l-1) r_(t+l-1).
```

Windows never cross episode boundaries. Following robomimic's n-step convention, a chunk's terminal and timeout flags are the corresponding `any` over its window; those signals do not redefine the stored episode boundary. The policy backup discount is `0.99^l`. Episodes shorter than `l` contribute no windows; conversion fails clearly if an entire split has no valid window.

At execution time, all `l` predicted actions are applied open-loop. Execution stops early if the environment terminates or truncates, and reported episode returns remain the ordinary undiscounted sum of primitive rewards. Post-training contraction horizons and saved rollout timesteps are measured in primitive environment steps. Learned-dynamics `--rollout-length` and OOD state-action samples operate at chunk decision boundaries, so one learned-dynamics step represents up to `l` primitive steps.

```bash
python sweep.py \
  --env HalfCheetah-v5 \
  --dataset-source minari \
  --algos bc cql \
  --chunk-lengths 1 4 8 \
  --eval
```

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

`--dql-reward-normalization auto` applies CleanDiffuser-style episodic return-range scaling to Minari training episodes. It leaves rewards unchanged for generated and robomimic datasets. Every resolved DQL setting and its source is saved in `run_manifest.json`.

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
  --eval
```

## Evaluation over training

Every run saves policy checkpoints at approximately 0, 10, 20, ..., 100 percent of policy training. Milestones that fall in the same epoch are collapsed. Checkpoints live under `checkpoint/step_<gradient_step>/`; fixed model-based dynamics are stored once at step zero, while RAMBO saves its changing dynamics with each checkpoint.

Passing `--eval` evaluates every policy checkpoint and then plots the selected sweep runs. The 0%-90% checkpoints use `--checkpoint-eval-episodes` (default 20) for training-history monitoring. The final checkpoint uses a separate reset-seed stream and `--final-eval-episodes` (default 100) for reported performance and generated-data expert evaluation. The final policy and 100-percent checkpoint are one logical policy and share one rollout. Performance uses task-native units: success rate for Can, Lift, and ToolHang, final fingertip-target distance for Reacher, balance duration for InvertedDoublePendulum, and forward displacement for HalfCheetah. Robomimic performance evaluation stops on first success; Gymnasium tasks retain their native termination and truncation behavior.

Final-policy contraction pairs deterministic rollouts from the same simulator state. One copy receives a fixed-norm perturbation only in controlled-agent `qpos/qvel`; goals and manipulated objects are unchanged. Continuing-Lift pairs are rerun through success until the contraction horizon; other tasks reuse the unperturbed final-performance trajectory and retain their existing termination behavior. Distance is the Euclidean norm over Cartesian positions of the controlled agent's bodies: robot and gripper bodies for Robomimic, arm bodies excluding the target for Reacher, and all HalfCheetah bodies. Every coordinate is measured in meters; joint angles and velocities are excluded from the distance. Curves have no phase alignment, fitted `(C, rho)`, or support-fraction report. `--contraction-trajectories` cannot exceed `--final-eval-episodes`.

State and state-action OOD ratios are evaluated over checkpoints. Each is the mean rollout-to-training k-nearest-neighbor distance divided by the corresponding held-out-to-training distance, so a ratio near one means rollout samples are about as far from training data as held-out samples are.

For Robomimic datasets, final-policy plots compare task performance and contraction curves across action chunk lengths. For generated datasets, they compare those metrics against the realized fraction of complete trajectories collected from the noisy expert. A fixed random fraction gives the clean-expert/noisy-expert ablation; zero clean-expert fraction gives the random/noisy-expert ablation. Performance and OOD ratios are also plotted against training percent. Lines are exact means with shaded hierarchical-bootstrap 10th-90th percentile bands: seeds and episodes are resampled for performance, while seeds and trajectory pairs are resampled for contraction. A one-seed run still uses episode or trajectory-pair variability; milestone OOD bands require multiple seeds because only one OOD estimate is saved per seed and checkpoint. Raw arrays are saved under the matching `evals/<environment>/<run-name>/<training-timestamp>/` directory, and plots are written under `evals/<environment>/plots/`.

Pass `--reuse-eval` together with `--eval` to reuse a completed matching evaluation or matching per-checkpoint rollout caches. Without it, the run's evaluation directory is cleared before evaluation. Cached rollouts retain returns, task performance, decision-boundary observations and action chunks, initial simulator states, and primitive Cartesian position traces, so plotting and metric recomputation do not require another environment rollout.

The learned-dynamics and finite-difference Jacobian evaluation functions remain in `eval.py` for possible future use, but `--eval` does not invoke them or generate mismatch plots.
