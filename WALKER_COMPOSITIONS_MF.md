# IQL and TD3BC reference Walker composition sweeps

This is a separate, additive model-free entrypoint. The existing MOPO/MOBILE
launcher, workers, policy builders, OfflineRL-Kit libraries, environments,
schemas, and plotting code are not modified. Existing runs do not need to be
retrained, reevaluated, migrated, or updated.

## Dataset and experiment semantics

`walker_composition_mf.py` accepts `--algos iql td3bc` (both by default).
It imports the existing composition dataset helpers without changing them:

- Settings: `noise0.5`, `noise1.0`, and `clean-medium-v2`. The first and last are
  the default settings.
- Default clean/other fractions: `1 0`, `.75 .25`, `.5 .5`, `.25 .75`, `0 1`.
  Repeat `--composition CLEAN OTHER` to select a subset.
- The default raw transition budget is 999,995. Each source meets its quota
  while retaining complete trajectories; reference preprocessing and the
  holdout subsequently reduce training rows. This is not a promise of 999,995
  rows in the training split.
- Seed-specific collection, medium-v2 sampling without replacement, reference
  timeout conversion, and random whole-episode 20% holdout are identical to
  the existing composition workflow. Chunk length is fixed to one.
- Matching dataset specifications reuse the same cached train/test arrays
  across IQL, TD3BC, MOPO, and MOBILE. There is no new source pool or independent
  recollection for each algorithm. Training never uses held-out episodes.
- Collection and evaluation use the frozen reference MuJoCo 2.1 environment,
  verified model XML, foot friction 0.9/1.9, and 1,000-step horizon. This is not
  Gymnasium Walker2d-v5 or a Minari dataset.

Data collection locks remain shared. Training locks do not collide between
algorithms. Separate invocations of the same model-free algorithm can still
wait for an identical clean endpoint, as they should to avoid duplicate fits.

## Training

Training uses the current repository's unchanged `build_model_free_policy` and
OfflineRL-Kit's `MFPolicyTrainer`. There is no dynamics fitting, synthetic
replay, or environment simulation during policy updates. Rewards are not
rescaled. Model-based flags such as `--real-ratio`, `--rollout-length`, and
`--dynamics-max-epochs` are not accepted by this launcher.

Shared defaults are 3,000 epochs, 1,000 updates per epoch, batch size 256,
discount 0.99, and target-update coefficient 0.005. The algorithm-specific
defaults match the ordinary repository sweep, not a newly tuned Walker recipe:

| Algorithm | Arguments and defaults |
|---|---|
| IQL | `--iql-temperature 3`, `--iql-expectile 0.7`, `--iql-learning-rate 3e-4`, `--iql-lr-schedule cosine`, `--iql-hidden-dims 256 256` |
| TD3BC | `--td3bc-alpha 2.5`, `--td3bc-learning-rate 3e-4`, `--td3bc-hidden-dims 256 256` |

IQL's temperature multiplies the advantage in the exponential weight; it is
not a reciprocal temperature. Its cosine schedule applies to the actor and
uses the requested epoch count. `constant` is also supported.

TD3BC retains the existing target noise 0.2, noise clip 0.5, delayed actor
update frequency 2, and training-only observation normalization. Its exact
training mean and standard deviation (including the existing `+1e-3`) are
saved as non-trainable tensors in every checkpoint. Evaluation loads these
statistics rather than recomputing them or using test observations. Shared
dataset files are never normalized in place.

### Example sweep

GPU 0 below is an example allocation, not a claim that it is free. This command
is one sequential process: algorithms, compositions, settings, and seeds run
in sequence. To parallelize, launch separate commands with one seed/setting
and choose GPUs according to your available resources.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/home/shekhe/miniconda3/envs/mujocold/bin/python /home/shekhe/stable-offline-rl/walker_composition_mf.py \
  --settings noise0.5 clean-medium-v2 --algos iql td3bc \
  --seed 10000 10100 10200 10300 \
  --num-samples 999995 --test-fraction 0.2 --epoch 3000 --batch-size 256 \
  --checkpoint-eval-episodes 10 --final-eval-episodes 20 \
  --device cuda --quiet --eval --reuse-eval
```

Use `--composition 0.5 0.5` for a 50/50-only test. `--dry-run` prints the grid
and effective algorithm-specific configurations without creating files or
launching workers. Arguments for one algorithm do not affect the other's
training identity. Different epochs or algorithm parameters create distinct
runs; they never overwrite an existing trained policy.

## Evaluation and storage

The default `--storage-root` remains
`/data/shekhe/stable-offline-rl/walker-reference/compositions`:

| Artifact | Relative directory |
|---|---|
| Shared datasets and splits | `datasets/<dataset-id>/` |
| Existing MOPO/MOBILE runs, unchanged | `runs/<training-id>/` |
| New IQL/TD3BC runs | `model_free_runs/<training-id>/` |
| Evaluations for all four algorithms | `evals/<training-id>/<eval-id>/` |

Checkpoints are evaluated only after training, and only with `--eval`.
Positive `--checkpoint-eval-episodes` saves 10%, 20%, ..., 90% milestones;
zero disables these milestones. `--final-eval-episodes` controls final-policy
evaluation. Zero for both counts skips evaluation workers and environments
entirely (collection/preflight can still need an environment).

Inference matches each current policy's deterministic action selection.
The new evaluator emits the existing composition report format with the same
reset seeds, final seed offset of 1,000,000, physics verification, displacement,
raw return, normalized return, episode lengths, and expert-baseline cache.
Normalization is included in checkpoint hashes, not in seed-varying plot
configuration fields.

To evaluate model-free runs later without training:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/home/shekhe/miniconda3/envs/mujocold/bin/python /home/shekhe/stable-offline-rl/walker_composition_mf_eval.py \
  --root /data/shekhe/stable-offline-rl/walker-reference/compositions/model_free_runs \
  --checkpoint-eval-episodes 10 --final-eval-episodes 20 --reuse-eval
```

`--run-dir` selects a single run instead. If no intermediate checkpoints were
saved, use `--checkpoint-eval-episodes 0`; missing checkpoints cause an error,
not retraining. Failed training is recorded and is not automatically replaced
or resumed. Evaluation failure leaves completed training reusable.

For MOPO/MOBILE, continue using the existing `walker_composition_eval.py` with
its documented root ending in **`compositions/runs`**. Do not point that old
evaluator at the broader composition directory: it does not load model-free
architectures. The new model-free evaluator ignores model-based runs.

## Combined plots

The existing `walker_composition_plot.py` is unchanged. Two additional cohorts
select TD3BC, IQL, MOPO, and MOBILE:

- `walker_composition_cohorts/all_algorithms_clean_noisy.json`
- `walker_composition_cohorts/all_algorithms_clean_medium.json`

They select seeds 10000, 10100, 10200, 10300 and all five compositions, budget
999,995, holdout 0.2, 3,000 epochs, 1,000 steps/epoch, and batch 256. Evaluation
counts are 10 checkpoint episodes and 20 final episodes. IQL/TD3BC use the
defaults above. MOPO uses penalty 2.5 and either L5/B50,000 (noisy) or
L1/B250,000 (medium); MOBILE uses L5/B50,000, penalty 0.5, dynamics cap 30.
Both model-based series select real ratio 0.05, not the separate 0.5 tests.

```bash
/home/shekhe/miniconda3/envs/mujocold/bin/python /home/shekhe/stable-offline-rl/walker_composition_plot.py \
  --root /data/shekhe/stable-offline-rl/walker-reference/compositions/evals \
  --cohort /home/shekhe/stable-offline-rl/walker_composition_cohorts/all_algorithms_clean_noisy.json

/home/shekhe/miniconda3/envs/mujocold/bin/python /home/shekhe/stable-offline-rl/walker_composition_plot.py \
  --root /data/shekhe/stable-offline-rl/walker-reference/compositions/evals \
  --cohort /home/shekhe/stable-offline-rl/walker_composition_cohorts/all_algorithms_clean_medium.json
```

These cohorts are selections, not evidence that all requested runs already
exist: missing or ambiguous points fail explicitly. Adjust a cohort's series
filters if you deliberately use different parameters. To compare different
epoch counts between algorithms, move the epoch filter from the common
`match` object into each series. Keep compatible evaluation budgets and seeds.

Existing MOBILE-only cohorts and plot outputs remain valid. Additional MOBILE
real-ratio lines are still supported by separate series; 50/50-only runs cannot
supply a complete five-point line. Plot statistics, labels, and actual
trajectory-fraction x coordinates have not changed. The original hierarchical
bootstrap still uses 10,000 resamples and 10th/90th percentiles (80% intervals).

These combined plots compare the reference composition protocol only. They do
not mix in older fixed-mixture reference runs or Gymnasium/Minari Walker runs
with different collection/split semantics. Other tasks' runs and plotters are
unaffected and require no updates.

## Tests

All new tests use temporary outputs and require no GPU. Optional integration
tests use the already installed frozen reference environment; no dependency
installation or modification is performed.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 /home/shekhe/miniconda3/envs/mujocold/bin/python -B -m unittest discover \
  -s /home/shekhe/stable-offline-rl -p 'test_walker_composition_mf*.py'
```

Set `WALKER_COMPOSITION_INTEGRATION=1` to additionally run real reference
collection, tiny CPU training, cross-runtime inference, evaluation, cache
reuse, and mixed model-free/model-based plotting checks. Tiny smoke policies
validate the implementation; their scores do not establish learning quality.

### Implementation verification (2026-09-17)

- Full repository discovery: 355 tests, successful; eight opt-in integration
  checks were skipped in that default invocation and all passed when enabled
  separately.
- Existing-builder initialization and nine-update parity for both algorithms;
  IQL cosine/constant scheduling; exact same-runtime deterministic inference.
- Frozen-runtime deterministic actions agree within `1e-7`, including TD3BC
  normalization of float64 simulator observations.
- End-to-end temporary run: five collected reference datasets, ten tiny actual
  model-free fits, twenty checkpoint/final evaluation episodes, and four valid
  random model-based actor fixtures evaluated by the original evaluator.
  Twelve images rendered with the unchanged plotter; all three metric summaries
  and the shared expert cache were checked.
- Completed fits/evaluations were reused with training and worker launches
  forbidden. Dataset hashes and held-out episode separation were preserved.
- All 373 pre-existing source/configuration files captured across the main
  repository, modified learner, and reference setup remained byte-identical.
  All test outputs were temporary and removed; no GPU training was launched.
