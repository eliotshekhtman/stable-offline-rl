# stable-offline-rl

## Saved artifacts

Runs are independent of the working directory. `--storage-root` is the common
parent of all datasets, trained runs, and evaluations; it defaults to
`/data/shekhe/stable-offline-rl`:

```text
<storage-root>/datasets/<environment>/<dataset-name>/<dataset-timestamp>/
<storage-root>/trained/<environment>/<algorithm>_chunk<length>_<dataset-name>/<training-timestamp>/
<storage-root>/evals/<environment>/<algorithm>_chunk<length>_<dataset-name>/<training-timestamp>/
<storage-root>/evals/<environment>/plots/
```

Pass an absolute override such as `--storage-root /scratch/shekhe/stable-offline-rl`
when another server uses a different large-data filesystem. Sweep discovery and
writes use only the selected root; there is no fallback search of repository-local
`datasets/`, `trained/`, or `evals/`. The expert path and the Minari and Hugging
Face caches are independent of this option and remain unchanged.

Each training manifest records the dataset identity and all algorithm-relevant training arguments. Before training, `sweep.py` searches the timestamped variants under the corresponding run name and reuses the newest complete run with an identical training schema. Evaluation arguments are not part of that schema, so adding `--eval` or changing an evaluation setting reloads the existing model instead of retraining it. A different training schema creates another timestamped variant and leaves prior variants intact. Dataset reuse is automatic; `--output-dir`, `--reuse-datasets`, and `--overwrite` are no longer used.

Training normally displays one progress bar over policy-training epochs. Pass `--quiet` to suppress routine terminal output, including that bar. CSV, TensorBoard, checkpoint, and text-log artifacts are still written in either mode; warnings, exceptions, and tracebacks remain visible.

Dataset splits are reused by the same rule. Paths in new metadata and manifests are absolute. Runs previously written under `/home/shekhe/train_dir` are not searched or modified.

Premade-data sweeps use every matching dataset by default. Pass `--dataset mh` for one Robomimic dataset type, or `--dataset medium-v0` (equivalently its full ID, such as `mujoco/halfcheetah/medium-v0`) for one Minari dataset.

The supported tasks are exactly `Reacher-v5`, `HalfCheetah-v5`, `Lift`, and `Can`. Reacher and HalfCheetah support generated, Minari, and clean-Minari data; Lift and Can support Robomimic data. Other tasks and incompatible task/source combinations fail during argument validation before creating artifacts.

All datasets are split by complete episode. Minari episodes, robomimic demonstrations, and generated rollouts share the same ordered transition schema: every `episode_id` occupies one contiguous block, and consecutive entries within that block are consecutive environment transitions. Generated `--num-samples` values are minimum transition counts. Source proportions allocate whole trajectories, and complete trajectories are added until that minimum is reached. Metadata records requested proportions plus realized trajectory and transition counts and fractions.

Robomimic Lift uses continuing-task semantics: success annotations are not Bellman terminals, demonstration endpoints are timeouts, and model-generated transitions do not terminate at success. These datasets and runs use a `_continuing` tag so they cannot be confused with or reuse older terminal-Lift artifacts. Other environments retain their existing termination semantics.

Generated datasets can mix clean expert, noisy expert, and random-action episodes. Pass `--composition CLEAN_EXPERT NOISY_EXPERT`; random data fills the remaining proportion. Repeat the argument to sweep specified compositions without taking a Cartesian product between their two values. `--noise-scale` applies only to the noisy expert component, while clean expert actions use zero noise. The default composition is `1 0`.

Use `--dataset-source clean-minari --dataset DATASET` to ablate between clean expert trajectories and one Minari quality dataset. `--minari-fraction` defaults to `0 0.25 0.5 0.75 1`; fractions allocate complete trajectories, and Minari episodes are selected from a seeded permutation without replacement. A request that needs more Minari episodes or transitions than the source contains fails instead of duplicating data. The 0% endpoint reuses an existing generated-clean dataset and model when its environment, expert, horizon, sample count, seed, split, and training settings match; the irrelevant historical clean-dataset `noise_scale` may differ.

```bash
python sweep.py \
  --env Reacher-v5 \
  --dataset-source clean-minari \
  --dataset medium-v0 \
  --num-samples 500000 \
  --minari-fraction 0 0.25 0.5 0.75 1 \
  --algos bc iql \
  --eval
```

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

For MOPO, MOBILE, and COMBO, `--dynamics-chunk-mode direct` (the default) preserves the existing model that predicts one complete macro-transition from the flattened action chunk. With `--dynamics-chunk-mode recursive` and a chunk length greater than one, the learned model is instead fit to primitive one-step transitions. A synthetic macro-transition then applies the chunk's actions recursively and open-loop through that one-step model, discounts primitive rewards by the base discount within the chunk, and presents the resulting endpoint and macro reward to the unchanged macro policy and critics. `--rollout-length` remains a count of macro-transitions in both modes.

Length one is always canonicalized to the exact legacy direct implementation: no adapter is inserted, no mode field is added to the training schema, and existing length-one runs remain reusable and loadable. Existing higher-length manifests without a mode field also continue to mean direct dynamics. Recursive higher-length runs carry an explicit versioned mode block in their training schema and are plotted separately. Recursive model rollouts cost roughly one dynamics call per primitive action, so they are expected to be slower as chunk length grows.

```bash
python sweep.py \
  --env HalfCheetah-v5 \
  --dataset-source minari \
  --algos bc cql \
  --chunk-lengths 1 4 8 \
  --eval
```

## Evaluation over training

Every run saves policy checkpoints at approximately 0, 10, 20, ..., 100 percent of policy training. Milestones that fall in the same epoch are collapsed. Checkpoints live under `checkpoint/step_<gradient_step>/`; fixed model-based dynamics are stored once at step zero.

Passing `--eval` first validates the trained run, evaluates its policy checkpoints, and then plots the selected sweep runs. Training without `--eval` does not run validation or true-environment evaluation rollouts; training still constructs the task environment because policy and replay-buffer construction require its observation and action spaces. The 0%-90% checkpoints use `--checkpoint-eval-episodes` (default 20) for diagnostic training-history monitoring; set it to `0` to skip all non-final rollouts and retain only the rigorous final evaluation. The final checkpoint uses a separate reset-seed stream and `--final-eval-episodes` (default 100) for reported performance and generated-data expert evaluation. The final policy and 100-percent checkpoint are one logical policy and share one rollout. Performance uses task-native units: success rate for Can and Lift, final fingertip-target distance for Reacher, and forward displacement for HalfCheetah. Robomimic performance evaluation stops on first success; Gymnasium tasks retain their native termination and truncation behavior.

Final-policy contraction pairs deterministic rollouts from the same simulator state. One copy receives a fixed-norm perturbation only in controlled-agent `qpos/qvel`; goals and manipulated objects are unchanged. Continuing-Lift pairs are rerun through success until the contraction horizon; other tasks reuse the unperturbed final-performance trajectory and retain their existing termination behavior. Distance is the Euclidean norm over Cartesian positions of the controlled agent's bodies: robot and gripper bodies for Robomimic, arm bodies excluding the target for Reacher, and all HalfCheetah bodies. Every coordinate is measured in meters; joint angles and velocities are excluded from the distance. Curves have no phase alignment, fitted `(C, rho)`, or support-fraction report. `--contraction-trajectories` cannot exceed `--final-eval-episodes`.

State and state-action OOD ratios are evaluated over checkpoints. Each is the mean rollout-to-training k-nearest-neighbor distance divided by the corresponding held-out-to-training distance, so a ratio near one means rollout samples are about as far from training data as held-out samples are.

For Robomimic datasets, final-policy plots compare task performance and contraction curves across action chunk lengths. For generated datasets, they compare those metrics against the realized fraction of complete trajectories collected from the noisy expert. A fixed random fraction gives the clean-expert/noisy-expert ablation; zero clean-expert fraction gives the random/noisy-expert ablation. Clean-Minari sweeps plot the same metrics against the realized Minari trajectory fraction, separately for each Minari dataset, sample count, and chunk length. Each plotter reads only its own source-specific metadata; generating one family no longer deletes unrelated existing plots. Performance and OOD ratios are also plotted against training percent. Lines are exact means with shaded hierarchical-bootstrap 10th-90th percentile bands: seeds and episodes are resampled for performance, while seeds and trajectory pairs are resampled for contraction. A one-seed run still uses episode or trajectory-pair variability; milestone OOD bands require multiple seeds because only one OOD estimate is saved per seed and checkpoint. Raw arrays are saved under the matching `<storage-root>/evals/<environment>/<run-name>/<training-timestamp>/` directory, and plots are written under `<storage-root>/evals/<environment>/plots/`.

When plotting an environment's full result history, pass `python plot.py --root /data/shekhe/stable-offline-rl/evals/<environment> --cohort <cohort.json>` to select explicit series. Each `match` key is a dotted path within `training_schema`; matched parameters are appended to the plot label. For example, this cohort deliberately plots two MOBILE variants as `mobile (real ratio=0.00)` and `mobile (real ratio=0.50)`:

```json
{
  "version": 1,
  "series": [
    {"algo": "mobile", "match": {"model_based.real_ratio": 0.0}},
    {"algo": "mobile", "match": {"model_based.real_ratio": 0.5}},
    {"algo": "mopo", "match": {"epoch": 300}}
  ]
}
```

Algorithms may use different match parameters. Within each series, every plotted x-value must resolve to exactly one seed-averaged training configuration, and every non-axis training parameter must remain constant across x-values. An underspecified selector that mixes configurations therefore fails instead of combining or silently splitting them; select direct and recursive dynamics, different epochs, or other parameter variants as separate series. Automatically generated plots from one coherent sweep need no cohort file and retain their existing labels.

Pass `--reuse-eval` together with `--eval` to reuse a completed matching evaluation or matching per-checkpoint rollout caches. Without it, the run's evaluation directory is cleared before evaluation. New non-final rollout caches retain only returns, task performance, decision-boundary observations, and action chunks; contraction-only initial simulator states and primitive Cartesian position traces are retained only for the final policy. Existing larger rollout caches remain readable. Gym expert references are shared by task, expert path, episode budget, and rollout seed under `<storage-root>/evals/<environment>/_expert_cache/`; Robomimic references are shared by task and official PH source. Matching legacy per-run `expert.npz` caches are still reused and promoted to the shared cache.

With `--eval`, validation evaluates learned dynamics on held-out transitions for model-based runs. The unused finite-difference Jacobian and mismatch-plot implementations have been removed.
