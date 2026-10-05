# Reference Walker composition sweeps

This is an additional entrypoint. No existing source files, environments, expert
files, run schemas, caches, or runs are replaced or modified. In particular,
`sweep.py`, `walker_sweep.py`, their workers, and the existing plotter keep their
previous behavior. No package installation is required on this server.

## What is preserved, and what differs

`walker_composition.py` uses the existing current-learner implementation from
`walker_sweep.py`. Collection and evaluation run through the isolated
`/home/shekhe/walker-benchmark/benchmark-python` environment, with verified native
MuJoCo 2.1, original model XML, right/left foot friction 0.9/1.9, 1,000-step
horizon, reward rule, reference actor, and reference termination/preprocessing.
This is not Gymnasium Walker2d-v5 and does not mislabel D4RL data as Minari.

Supported families are `noise0.5`, `noise1.0`, and `clean-medium-v2`. The default
families are `noise0.5 clean-medium-v2`. Each defaults to clean/other compositions
`1 0`, `0.75 0.25`, `0.5 0.5`, `0.25 0.75`, `0 1`. Repeat `--composition CLEAN OTHER`
to request a subset instead. Chunk length is fixed to 1.

- Every seed and composition gets its own dataset. Generated components are
  collected afresh for that composition; there are **no reusable source pools**.
  Finished datasets are cached and shared across algorithms and exact reruns.
- Seeding follows ordinary collection: one `default_rng(seed)` drives clean then
  noisy trajectory resets and noise draws. Noise is added to deterministic expert
  actions as `Normal(0, noise_scale / sqrt(6))` per coordinate, followed by clipping
  to `[-1, 1]`. Shared seeds can therefore produce shared trajectory prefixes,
  just as in the ordinary collector; afresh does not imply independent RNG streams.
- Clean/medium selects complete medium episodes with a separate seeded random
  permutation, without replacement, then collects clean episodes with the seed.
  It concatenates **medium then clean**, like ordinary clean/Minari collection.
  The medium dataset itself is fixed; the seed changes its subset/order, not its
  original trajectories. The 100% medium endpoint contains the same complete
  episodes across seeds, with seed-dependent order and train/test split.
- `--num-samples 999995` is a **minimum raw transition budget before reference
  preprocessing and the outer split**, not a retained training-row target.
  Each source's quota is rounded upward independently and its final trajectory
  retained in full. Overshoots are recorded rather than trimmed.
- Medium-v2 has 999,995 transitions in 1,190 complete episodes; the five trailing
  incomplete rows are excluded. A requested medium quota above 999,995 fails
  before collection or training. No medium episodes are duplicated within a
  dataset. Different compositions may naturally select overlapping subsets.
- The reference converter is called once after assembly. It removes timeout rows
  and the global final row, attaching episode IDs before removal. Therefore the
  processed total is smaller than the raw budget. At the complete medium endpoint,
  it is 999,317 or 999,318 depending on whether the final selected row was already
  a timeout; this is the unchanged reference conversion rule.
- `--test-fraction 0.2` calls the existing random **whole-episode** splitter with
  the experiment seed. It does not stratify by source or rebalance transition
  counts. Held-out episodes enter neither real replay nor dynamics fitting.
  The dynamics model retains its separate internal validation split.

For the default budget, requested raw source quotas are:

| Clean / other | Clean minimum | Other minimum |
|---|---:|---:|
| 1 / 0 | 999,995 | 0 |
| 0.75 / 0.25 | 749,997 | 249,999 |
| 0.5 / 0.5 | 499,998 | 499,998 |
| 0.25 / 0.75 | 249,999 | 749,997 |
| 0 / 1 | 0 | 999,995 |

The identical 100%-clean endpoint is shared across families for matching seed and
parameters. Two families × five compositions × four seeds yield 40 plot points
but **36 distinct datasets/training runs per algorithm**. It is never counted
twice within a plot point.

## Training defaults

The defaults retain the existing reference recipe: MOBILE, 3,000 policy epochs,
1,000 updates/epoch, batch 256, real ratio 0.05, rollout length 5, rollout batch
50,000, penalty 0.5, actor LR 1e-4, critic LR 3e-4. Dynamics uses the existing
MOBILE cap of 30, or MOPO's uncapped early stopping. `--dynamics-max-epochs 0`
explicitly means uncapped. Other exposed training arguments match
`walker_sweep.py`; no objectives, entropy calculations, clamps, architectures,
or optimizers are changed. MOPO is also available through `--algos mopo` or
`--algos mobile mopo`.

This uses the **current learner**, not a promise of bitwise equivalence to the
frozen author runtime. The existing reference comparison's MOPO initialization
ordering and package-version caveats still apply.

## Commands

From any directory, the full requested MOBILE sweep is:

```bash
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/shekhe/miniconda3/envs/mujocold/bin/python \
  /home/shekhe/stable-offline-rl/walker_composition.py \
  --settings noise0.5 clean-medium-v2 \
  --algos mobile --seed 10000 10100 10200 10300 \
  --num-samples 999995 --test-fraction 0.2 \
  --epoch 3000 --device cuda --quiet \
  --checkpoint-eval-episodes 10 --final-eval-episodes 20 --eval --reuse-eval
```

This is **one sequential process**, not automatic GPU scheduling. To parallelize,
launch separate commands with one seed (and optionally one family) per command,
choosing GPUs yourself. Locks prevent concurrent commands from duplicating a
shared dataset or trained clean endpoint. The GPU number above is an example,
not a claim about its availability. Add `--dry-run` to inspect the exact grid and
quotas without creating files, opening an environment, or starting training.

`--eval` enables evaluation only **after training**, in a separate CPU process.
Without it, no policy/expert evaluation runs. Positive
`--checkpoint-eval-episodes` saves 10%, 20%, ..., 90% milestones during training
even if `--eval` is absent, allowing later evaluation. Zero saves no milestones.
The default final budget is 20 episodes. Zero for both counts skips evaluation
entirely. Collection/preflight still needs the reference environment.

To evaluate completed new runs later, without training:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/shekhe/miniconda3/envs/mujocold/bin/python \
  /home/shekhe/stable-offline-rl/walker_composition_eval.py \
  --root /data/shekhe/stable-offline-rl/walker-reference/compositions/runs \
  --checkpoint-eval-episodes 10 --final-eval-episodes 20 --reuse-eval
```

Use `--run-dir /absolute/path/to/run` to evaluate one run instead. Set
`--checkpoint-eval-episodes 0` for runs that did not save milestones; missing
earlier policies cannot be recreated from the final policy. This reports an
explicit error rather than silently dropping requested diagnostic evaluations
or retraining. Changing episode counts creates a separate evaluation cache.

## Plotting

Two supplied cohort files select the four seeds 10000/10100/10200/10300, all five
fractions, and the reference MOBILE parameters above. These are for the **new**
composition workflow only; older fixed-mixture runs do not have matching dataset
or evaluation semantics and are not folded into these plots.

```bash
/home/shekhe/miniconda3/envs/mujocold/bin/python \
  /home/shekhe/stable-offline-rl/walker_composition_plot.py \
  --root /data/shekhe/stable-offline-rl/walker-reference/compositions/evals \
  --cohort /home/shekhe/stable-offline-rl/walker_composition_cohorts/mobile_clean_noisy.json

/home/shekhe/miniconda3/envs/mujocold/bin/python \
  /home/shekhe/stable-offline-rl/walker_composition_plot.py \
  --root /data/shekhe/stable-offline-rl/walker-reference/compositions/evals \
  --cohort /home/shekhe/stable-offline-rl/walker_composition_cohorts/mobile_clean_medium.json
```

Outputs default to `compositions/evals/plots/<cohort-name>/<metric>/`:
`performance_vs_composition.png`, one
`performance_history_fraction_<fraction>.png` per composition, and
`plot_summary.json`. `--out /absolute/path` changes the destination.

The default y metric is **measured forward displacement in metres**, like the
ordinary Walker performance plots. `--metric raw_return` or
`--metric normalized_return` selects alternative recorded metrics. Displacement
is measured from simulator positions, never inferred from return. Final
evaluation resets use experiment seed + 1,000,000 + episode; intermediate
checkpoints use experiment seed + episode, matching ordinary evaluation.

Selection uses **requested transition fractions**. The x positions use mean
**actual trajectory fractions**, preserving the existing plotting convention.
They need not equal 0.25/0.5/0.75 when source episode lengths differ. Labels say
trajectories explicitly; metadata also records actual transition fractions.
Curves use equal weight per seed and the existing hierarchical bootstrap:
10,000 resamples, 10th/90th percentiles (**80% intervals**, not 95%).

Cohorts fail on missing points, ambiguous configurations, mixed physics, or
incompatible evaluations. If multiple evaluation budgets exist, add e.g.
`"eval_match": {"final_eval_episodes": 20, "checkpoint_eval_episodes": 10}`
at the cohort's top level. To add MOPO or another MOBILE parameter line, add a
`series` entry with its `algo` and `training_config.*` filters. Each series must
resolve to one seed-grouped training configuration; different algorithms may use
different epochs/parameters. There is no automatic picking of the best run.
No contraction/OOD evaluation is introduced by this workflow.

## Storage, integrity, and verification

All new artifacts default to
`/data/shekhe/stable-offline-rl/walker-reference/compositions/`, overridden by
`--storage-root`. Datasets live under `datasets/<identity>/`, trained policies
under `runs/<identity>/`, and evaluations under `evals/<training-id>/<eval-id>/`.
Datasets store train/test arrays once, source/episode provenance, quotas, actual
counts and fractions, full physics, and checksums. No raw source pools or
redundant unsplit dataset files are retained.

Dataset publication is atomic and creation is locked. Completed training is
reused only with matching dataset/training identity and checkpoint checksums.
Failures remain recorded and are not automatically overwritten or resumed.
Evaluation failures leave successful training reusable. Experts are evaluated
once per physics/seed/count/protocol combination and shared across runs.

Source fingerprints are recorded for auditing, not used to invalidate runs for
comment-only code changes. Future semantic changes to this new workflow must
explicitly version **its own** protocol or implement a targeted compatibility
check; they must not bump the unrelated ordinary sweep schema.

Run all regression and opt-in reference integration tests without generating
Python bytecode in the existing repositories:

```bash
cd /home/shekhe/stable-offline-rl
PYTHONDONTWRITEBYTECODE=1 WALKER_REFERENCE_TESTS=1 WALKER_COMPOSITION_INTEGRATION=1 \
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/shekhe/miniconda3/envs/mujocold/bin/python -m unittest discover -v
```

Integration tests use temporary datasets/policies and the reference environment.
They do not start full experiments or overwrite any existing run.

Implementation verification included the full existing/new regression suite,
reference actor/update parity, concurrent dataset/run reuse, and a real CPU sweep
covering both algorithms and both families at all five compositions: nine distinct
datasets and 18 tiny trained/evaluated policies. All policies had finite weights;
source quotas and disjoint episode splits were verified. Reuse of all 18 runs was
tested with collection, training, and evaluation subprocesses explicitly forbidden.
Both plot families rendered successfully for all three metrics. These were
pipeline smoke tests, not evidence of learned performance; their temporary
datasets, checkpoints, evaluations, and plots were removed afterward.
