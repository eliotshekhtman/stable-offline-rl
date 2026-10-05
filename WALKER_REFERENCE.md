# Isolated Walker reference sweeps

`walker_sweep.py` trains the **current modified OfflineRL-Kit learner** on the
legacy Walker benchmark data. `walker_reference.py` is its data/evaluation worker,
run through the existing `walker-benchmark/benchmark-python` interpreter.

This is a separate experiment entrypoint, not a change to `sweep.py`. Existing
Gymnasium/Minari Walker sweeps, other tasks, dataset caches, test splits, run
schemas and plotting behavior are unchanged. No packages need to be installed in
either environment. No expert files need to be replaced.

## Settings

| `--settings` value | Data |
|---|---|
| `medium-v2` | The canonical D4RL Walker2d-medium-v2 dataset, not a mixture |
| `clean-medium-v2` | Prepared 50/50 clean expert / canonical medium-v2 episode mixture |
| `noise0.5` | Prepared 50/50 clean / noise-injected expert, noise scale 0.5 |
| `noise1.0` | Prepared 50/50 clean / noise-injected expert, noise scale 1.0 |

The last three reuse the **same verified datasets** used by `run_mixture.py`.
In that older launcher `--setting medium-v2` means the mixture; here it is named
`clean-medium-v2` to avoid confusing it with the pure benchmark.

Default mixture manifest:
`/data/shekhe/walker-benchmark/mixtures/seed0_n500000/manifest.json`.
It records the original expert actor and reference-environment collection.
Dataset seed 0 stays fixed across training seeds. This entrypoint does not
recollect data, fetch policies or interpret an SB3 archive as a D4RL actor.
For new dataset seeds, prepare them with the existing isolated collector and pass
their `--manifest`; collection still runs in the reference environment.

The reference converter retains explicit next observations and true terminals,
and drops timeout transitions and the global final row. Pure medium-v2 therefore
has **999,322** usable transitions from 1,000,000 raw rows. Mixture sizes retain
their original small whole-trajectory overshoots; they are not silently resized.

## Isolation and comparison controls

- Training uses `mujocold`, the current `policies.py` builders, and the current
  `MBPolicyTrainer`. Chunk length is fixed to 1. No training environment is needed.
- Data conversion and policy evaluation use the pinned **MuJoCo 2.1 / Gym D4RL**
  environment, including right/left foot friction **0.9 / 1.9**, original body
  masses, reward rule and 1,000-step horizon. This is not Walker2d-v5 with just a
  friction override. Model rollouts use the reference Walker termination function.
- The worker verifies the frozen reference packages, source revisions, full model
  XML and dataset hashes before use, and checks mixture provenance against them.
  Run manifests record data hashes, source code hashes, physics,
  parameters, episode selections and actual training/test row counts.
- MOBILE uses the reference initialization order (actor, critics, dynamics),
  without altering the shared builder. Tests compare initial weights and one
  update against the frozen author entrypoint. MOPO uses the current actor/critic
  builder; its within-network initialization draw order differs from the author
  example, so identical MOPO seed numbers do not promise identical initial weights.
- Neither entropy calculations nor policy objectives are changed. MOBILE keeps
  its reference nonnegative target clamp with zero return shift.
- Evaluations are deterministic-policy rollouts in a **separate CPU process after
  training**. They cannot consume the training RNG stream or affect optimization.
  This differs from the author's every-epoch in-process evaluation and is not a
  claim of bitwise identical full training. Python package versions also remain
  those of the respective isolated environments.
- Results contain raw episode returns and D4RL normalized scores, **not** the
  forward-displacement metric of the ordinary Walker plots. Population standard
  deviations describe episodes within a checkpoint, not uncertainty across seeds.

### Optional outer test split

`--test-fraction 0` (default) uses all reference-converted rows.
`--test-fraction 0.2` calls the repository's existing whole-episode random splitter,
with the training seed. Conversion happens once before splitting: timeout rows
cannot be mistaken for new episode boundaries, and splitting does not introduce
additional final-row drops. The reference's partial trailing episode stays intact
as one group; it is not silently discarded.

The held-out episodes are excluded from **both real replay and dynamics fitting**.
They do not select checkpoints, change losses or provide early stopping. The
dynamics model still has its separate reference internal holdout
`min(20% of remaining training rows, 1000)` and patience of 5. Thus the outer split
changes available data, coverage and subsequent stochastic training, not the
learning objective. It is an empirical control, not an assumption of no effect.

The existing reference launchers also remain unchanged. To isolate the outer
split itself, compare two runs of this new entrypoint with all other options held
fixed. Comparing an old reference run directly to a split-enabled current run
would combine the split with the runtime/trainer differences described above.

## Defaults

Both algorithms default to the public medium-v2 recipe:
3,000 epochs, 1,000 updates/epoch, batch size 256, real ratio 0.05, rollout length 5,
50,000 rollout starts every 1,000 updates, model buffer capacity
`50000 * 5 * 5`, penalty 0.5, actor LR 1e-4, critic LR 3e-4,
gamma 0.99, tau 0.005, automatic entropy tuning with target -6 and LR 1e-4,
cosine actor LR schedule, and unnormalized rewards.

Dynamics defaults match each author example: **MOBILE cap 30; MOPO uncapped with
early stopping**. Use `--dynamics-max-epochs N` to override; `0` disables the cap.
These are intentionally not the defaults of the ordinary Minari sweeps.

By default there is only a final 10-episode evaluation.
`--checkpoint-eval-episodes N` saves and evaluates 10%, 20%, …, 90% milestones
after training, without evaluating the final policy twice. Final evaluation uses
`--final-eval-episodes` separately. Zero disables the corresponding evaluations;
zero for both skips the evaluation worker entirely. Data preflight still creates
one reference environment to verify physics. Every checkpoint uses the same
seeded evaluation stream for a given training seed.

## Commands

Run from `/home/shekhe`, using the current environment. For pure medium-v2:

```bash
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/shekhe/miniconda3/envs/mujocold/bin/python stable-offline-rl/walker_sweep.py \
  --settings medium-v2 --algos mobile --seed 1 2 \
  --test-fraction 0 --epoch 3000 --device cuda \
  --checkpoint-eval-episodes 10 --final-eval-episodes 20 --quiet
```

The same comparison with the optional 20% outer episode split:

```bash
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/shekhe/miniconda3/envs/mujocold/bin/python stable-offline-rl/walker_sweep.py \
  --settings medium-v2 --algos mobile --seed 1 2 \
  --test-fraction 0.2 --epoch 3000 --device cuda \
  --checkpoint-eval-episodes 10 --final-eval-episodes 20 --quiet
```

The three prepared mixtures, keeping the medium-v2 parameters fixed:

```bash
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/shekhe/miniconda3/envs/mujocold/bin/python stable-offline-rl/walker_sweep.py \
  --settings noise0.5 noise1.0 clean-medium-v2 --algos mobile --seed 1 2 \
  --test-fraction 0 --epoch 3000 --device cuda \
  --checkpoint-eval-episodes 10 --final-eval-episodes 20 --quiet
```

Each command runs its settings/seeds/algorithms **sequentially** in one process.
Use separate commands with one seed each if you want concurrent jobs. MOPO is
available through `--algos mopo` or `--algos mopo mobile`. For a medium-expert
parameter comparison explicitly add `--rollout-length 1 --penalty-coef 1.5`;
the script never changes parameters based on the dataset name.

Add `--dry-run` to print options without reading data or creating files.
`--smoke --device cpu` tests the pipeline on the first 8 episode groups with
2 dynamics and policy epochs, 8 updates/epoch and at most 2 evaluation episodes.
Smoke runs are explicitly labelled and stored separately; they are not results.

## Outputs and tests

Outputs default to `/data/shekhe/stable-offline-rl/walker-reference/runs/` in unique
timestamped directories; `--storage-root` overrides the root. No ordinary run
cache is reused, invalidated or overwritten. Rerunning a command starts new runs.
This isolated format is not automatically included in existing cohort plots.

Each run contains `run_manifest.json`, training CSVs under `record/`, policy and
dynamics checkpoints, and `evaluation.json` with individual returns, lengths and
normalized scores (when evaluation is enabled). Converted transfer files are
temporary on the selected storage volume and removed after loading.

```bash
cd /home/shekhe/stable-offline-rl
WALKER_REFERENCE_TESTS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/shekhe/miniconda3/envs/mujocold/bin/python -m unittest test_walker_sweep -v
```
