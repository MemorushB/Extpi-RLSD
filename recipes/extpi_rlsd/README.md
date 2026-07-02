# ExtPI-RLSD MVP Recipe

This recipe compares Qwen3-1.7B LoRA under:

- `closed_sft`: closed-model visible-solution SFT.
- `grpo`: math verifier GRPO.
- `opd_pg_qwen8b`: sampled-token OPD-PG/K1 with Qwen3-8B.
- `extpi_rlsd`: GRPO with privileged-trace RLSD reweighting.
- `extpi_rlsd_shuffled`: shuffled privileged-trace control.

The MVP includes reusable data tools, exact-token scorer helpers, worker-side
RLSD loss support, and a legacy PPO hook that materializes
`teacher_pi_log_probs` after rollout and before actor update. The current MVP
only supports `teacher_update_mode=base_no_adapter`. The actor loss fails fast
when `rlsd_enabled=True` and that tensor is missing, so there is no silent GRPO
fallback.

## Local Storage

Large files must live outside the repository:

```bash
bash scripts/setup_local_storage.sh
```

This creates `/data/users/rchen/extpi-rlsd/` and symlinks `data/`,
`checkpoints/`, `outputs/`, and `cache/` into the project.

## GPU Constraint

Single-card recipe scripts default to:

```bash
CUDA_VISIBLE_DEVICES=6
NPROC_PER_NODE=1
NGPUS_PER_NODE=1
```

They exit if another GPU is selected unless the scripts are intentionally
changed.

Multi-GPU entrypoints are separate `*_multi.sh` scripts. They source
`_env_multi.sh`, require `NGPUS_PER_NODE>=2`, and expect `CUDA_VISIBLE_DEVICES`
to be set intentionally.

## Data Flow

```bash
bash recipes/extpi_rlsd/scripts/00_prepare_dataset.sh \
  --eval_jsonl /path/to/aime24.jsonl \
  --eval_jsonl /path/to/aime25.jsonl \
  --eval_jsonl /path/to/hmmt25.jsonl
bash recipes/extpi_rlsd/scripts/01_generate_qwen8b_pi.sh
bash recipes/extpi_rlsd/scripts/01b_generate_recipient_uplift_completions.sh
bash recipes/extpi_rlsd/scripts/03_build_frontier.sh
```

ExtPI/UPOD defaults to `PI_TRACE_FIELD=qwen32b_pi_trace`. Set
`PI_TRACE_FIELD=qwen8b_pi_trace` only for legacy Qwen3-8B trace data.

Set `ALLOW_MISSING_EVAL_CONTAMINATION=1` only for local development runs that
intentionally skip eval contamination filtering.

Closed-teacher SFT data requires:

```bash
export CLOSED_TEACHER_BASE_URL=...
export CLOSED_TEACHER_API_KEY=...
export CLOSED_TEACHER_MODEL=...
bash recipes/extpi_rlsd/scripts/02_generate_closed_sft.sh
bash recipes/extpi_rlsd/scripts/02b_build_closed_sft_parquet.sh
```

## Smoke Commands

GRPO smoke:

```bash
TOTAL_TRAINING_STEPS=5 bash recipes/extpi_rlsd/scripts/run_grpo.sh
```

Direct sampled-token OPD-PG smoke:

```bash
TOTAL_TRAINING_STEPS=5 bash recipes/extpi_rlsd/scripts/run_opd_pg.sh
```

This baseline uses a single-card `inline_external_hf` teacher scorer for
sampled-token K1 OPD-PG and runs a tokenizer compatibility preflight. It does
not create a separate verl teacher resource pool. It is not a full-vocabulary
or top-k GKD baseline.

ExtPI-RLSD smoke:

```bash
TOTAL_TRAINING_STEPS=5 bash recipes/extpi_rlsd/scripts/run_extpi_rlsd.sh
```

The dedicated entry point is `verl.trainer.extpi_rlsd.main_extpi_rlsd`; it uses
the legacy PPO trainer because the MVP hook is implemented there.

Closed/off-policy SFT smoke:

```bash
SFT_EPOCHS=1 bash recipes/extpi_rlsd/scripts/run_closed_sft.sh
```

Closed SFT runs parquet preflight by default (`SFT_PREFLIGHT=1`) and uses
`MAX_SEQUENCE_LENGTH=8192` unless overridden.

## Hyperparameter Matrix

Run one UPOD tuning job from `experiments/extpi_rlsd/hparam_matrix.csv`:

```bash
RUN_ID=R1-01 bash recipes/extpi_rlsd/scripts/launch_hparam_matrix.sh
```

The launcher uses gpu6 by default, enables console + WandB + file logging, runs
matched-dev proxy eval only, and writes summaries under
`${EXTPI_DATA_ROOT}/outputs/hparams/`. Use `HPARAM_ENTRYPOINT=multi` only when
multiple GPUs are explicitly allocated.

## Multi-GPU Commands

GRPO and ExtPI-RLSD scale actor/rollout workers across the visible GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NGPUS_PER_NODE=4 \
SCALE_MODE=fixed \
TOTAL_TRAINING_STEPS=5 \
bash recipes/extpi_rlsd/scripts/run_extpi_rlsd_multi.sh

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NGPUS_PER_NODE=4 \
SCALE_MODE=fixed \
SFT_EPOCHS=1 \
bash recipes/extpi_rlsd/scripts/run_closed_sft_multi.sh
```

Direct OPD uses different backends by entrypoint:

- `run_opd_pg.sh`: single-card inline Qwen3-8B scorer.
- `run_opd_pg_multi_teacher_pool.sh`: multi-GPU verl distillation teacher pool.

Example 4-GPU OPD layout:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NGPUS_PER_NODE=3 \
TEACHER_NGPUS_PER_NODE=1 \
TOTAL_TRAINING_STEPS=5 \
bash recipes/extpi_rlsd/scripts/run_opd_pg_multi_teacher_pool.sh
```

Keep `SCALE_MODE=fixed` for single-card comparability. Use `SCALE_MODE=linear`
only for throughput experiments after fixed-scale smoke passes.
For closed SFT, `SCALE_MODE=fixed` defaults to global batch 8, and the multi-GPU
script checks that `SFT_TRAIN_BATCH_SIZE` is divisible by `NGPUS_PER_NODE * NNODES`.

Run the local contract tests without starting training:

```bash
pytest tests/extpi_rlsd
```

Evaluate one checkpoint and aggregate JSON outputs:

```bash
CHECKPOINT_PATH=/path/to/checkpoint \
RUN_NAME=extpi_rlsd \
CHECKPOINT_NAME=step_100 \
bash recipes/extpi_rlsd/scripts/run_eval_checkpoint.sh

bash recipes/extpi_rlsd/scripts/evaluate_all.sh
```

Matched-dev checkpoint selection defaults to Avg@4 with fixed prompt-seed pairs
`0,1,2,3` and `max_new_tokens=4096`.

Prepare and run the official AIME/HMMT evaluation pipeline:

```bash
bash recipes/extpi_rlsd/scripts/04_prepare_official_eval_data.sh

MODEL_PATH=/path/to/base_model \
CHECKPOINT_PATH=/path/to/checkpoint \
RUN_NAME=extpi_rlsd \
CHECKPOINT_NAME=step_100 \
bash recipes/extpi_rlsd/scripts/run_official_math_eval.sh
```

This writes normalized `aime24`, `aime25`, and `hmmt25` eval files under
`${EXTPI_DATA_ROOT}/datasets/eval/official_math` and writes per-dataset JSON
results plus `summary.csv` / `summary.md` under
`${EXTPI_DATA_ROOT}/outputs/eval_official/${RUN_NAME}`. Official math eval
defaults to OPSD-compatible prompts, `Avg@12`, fixed seeds `0..11`,
`max_new_tokens=38912`, and non-thinking Qwen3 generation unless
`EVAL_ENABLE_THINKING=1` is set.

## WandB Logging

All training entrypoints use `TRAINER_LOGGER`, which defaults to
`["console","wandb"]`, and set `WANDB_MODE=online` when WandB is enabled and no
mode was provided by the caller.

```bash
WANDB_ENTITY=... \
TOTAL_TRAINING_STEPS=5 \
bash recipes/extpi_rlsd/scripts/run_extpi_rlsd.sh
```

Set `WANDB_MODE=offline` when running without online credentials. Set
`EXTPI_ENABLE_WANDB=0` or `TRAINER_LOGGER='["console"]'` to disable WandB for a
run. `WANDB_DIR` defaults to
`/data/users/rchen/extpi-rlsd/outputs/wandb`.

## Public Claims Boundary

Use the label `OPD-PG (sampled-token K1)` for the direct distillation baseline.
Do not claim full-vocabulary or top-k GKD results unless those baselines are
actually run.
