# ExtPI-RLSD

ExtPI-RLSD is a research MVP built on a pinned upstream `volcengine/verl`
commit:

```text
cbd7f9f462c0230f4f6161462b5b294c9d55d453
```

The repository keeps the vendored `verl` Python training stack plus the
ExtPI-RLSD recipe, tools, and tests. Upstream documentation, Docker files,
examples, CI workflows, and unrelated tests are intentionally removed so this
checkout is focused on the ExtPI-RLSD experiments.

## Methods

The recipe compares Qwen3-1.7B LoRA under single-GPU smoke entrypoints and
separate multi-GPU scale-out entrypoints:

- `closed_sft`: closed-model visible-solution SFT.
- `grpo`: math-verifier GRPO.
- `opd_pg_qwen8b`: Baseline 2, sampled-token OPD-PG/K1 with Qwen3-8B.
- `extpi_rlsd`: GRPO with privileged-trace RLSD reweighting.
- `extpi_rlsd_shuffled`: shuffled privileged-trace control.

The ExtPI-RLSD hook builds scorer-only PI prompts, appends the original student
response token IDs without decode/re-tokenize, scores them under the
base-no-adapter self-teacher, and writes `teacher_pi_log_probs` into the actor
update batch. The MVP only supports `teacher_update_mode=base_no_adapter`. The
actor loss fails fast when RLSD is enabled and that tensor is missing.

## Repository Layout

```text
recipes/extpi_rlsd/       Recipe configs, prompts, rewards, and run scripts
tools/extpi_rlsd/         OPSD preparation, PI cache, split, manifest, and report tools
tests/extpi_rlsd/         Local contract tests for ExtPI-RLSD
verl/trainer/extpi_rlsd/  RLSD math, scorer helpers, manifest, and trainer hook
docs/                     Project implementation plan only
scripts/                  Local storage setup helper
```

Large datasets, model weights, caches, checkpoints, and outputs must live under
`/data/users/rchen/extpi-rlsd/`; the project keeps local symlinks only.

## Setup

Create local storage symlinks:

```bash
bash scripts/setup_local_storage.sh
```

Install the required training dependencies outside this repository. The current
environment must provide `torch`, `ray`, `transformers`, `peft`, `vllm`,
`math-verify`, and the regular verl runtime dependencies.

## GPU Constraint

Single-card recipe scripts default to gpu6 only:

```bash
CUDA_VISIBLE_DEVICES=6
NPROC_PER_NODE=1
NGPUS_PER_NODE=1
```

The scripts exit if a different GPU is selected unless the script is
intentionally edited.

Multi-GPU scripts are opt-in and use `_env_multi.sh`, which does not apply the
gpu6 guard. Set `CUDA_VISIBLE_DEVICES` explicitly before using them.

## Data Flow

Prepare OPSD, generate verified Qwen3-8B PI traces, attach recipient uplift
statistics, and build frontier splits:

```bash
bash recipes/extpi_rlsd/scripts/00_prepare_dataset.sh \
  --eval_jsonl /path/to/aime24.jsonl \
  --eval_jsonl /path/to/aime25.jsonl \
  --eval_jsonl /path/to/hmmt25.jsonl
bash recipes/extpi_rlsd/scripts/01_generate_qwen8b_pi.sh
bash recipes/extpi_rlsd/scripts/01b_generate_recipient_uplift_completions.sh
bash recipes/extpi_rlsd/scripts/03_build_frontier.sh
```

For local development only, `ALLOW_MISSING_EVAL_CONTAMINATION=1` lets
`00_prepare_dataset.sh` run without eval contamination files. Official splits
must not use that override.

Closed-teacher SFT data uses an OpenAI-compatible endpoint:

```bash
export CLOSED_TEACHER_BASE_URL=...
export CLOSED_TEACHER_API_KEY=...
export CLOSED_TEACHER_MODEL=...
bash recipes/extpi_rlsd/scripts/02_generate_closed_sft.sh
bash recipes/extpi_rlsd/scripts/02b_build_closed_sft_parquet.sh
```

## Run Entrypoints

These commands are the intended single-card smoke entrypoints once data and
dependencies are ready. They do not bypass the gpu6 guard.

```bash
TOTAL_TRAINING_STEPS=5 bash recipes/extpi_rlsd/scripts/run_grpo.sh
TOTAL_TRAINING_STEPS=5 bash recipes/extpi_rlsd/scripts/run_opd_pg.sh
TOTAL_TRAINING_STEPS=5 bash recipes/extpi_rlsd/scripts/run_extpi_rlsd.sh
TOTAL_TRAINING_STEPS=5 bash recipes/extpi_rlsd/scripts/run_extpi_rlsd_shuffled.sh
SFT_EPOCHS=1 bash recipes/extpi_rlsd/scripts/run_closed_sft.sh
```

`run_opd_pg.sh` is the Baseline 2 entrypoint. It uses a single-card
`inline_external_hf` teacher scorer, does not create a separate verl teacher
resource pool, and performs a tokenizer compatibility preflight before
training. `run_extpi_rlsd.sh` uses
`verl.trainer.extpi_rlsd.main_extpi_rlsd`, which runs the legacy PPO trainer
hook that materializes `teacher_pi_log_probs`.

`run_closed_sft.sh` is the off-policy SFT baseline entrypoint. It runs parquet
preflight by default (`SFT_PREFLIGHT=1`) and defaults to
`MAX_SEQUENCE_LENGTH=8192`.

## Multi-GPU Entrypoints

Use these only when multiple GPUs are intentionally allocated:

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

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NGPUS_PER_NODE=3 \
TEACHER_NGPUS_PER_NODE=1 \
TOTAL_TRAINING_STEPS=5 \
bash recipes/extpi_rlsd/scripts/run_opd_pg_multi_teacher_pool.sh
```

`SCALE_MODE=fixed` keeps the research batch at 8 prompts/update for
single-card comparability. `SCALE_MODE=linear` sets the default train batch to
`8 * NGPUS_PER_NODE * NNODES` for throughput experiments.
For closed SFT, `SCALE_MODE=fixed` likewise defaults to global batch 8, and
`run_closed_sft_multi.sh` requires that batch to divide the world size.

OPD backend choice is deliberate:

- `run_opd_pg.sh`: single-card `inline_external_hf` scorer in the trainer actor.
- `run_opd_pg_multi_teacher_pool.sh`: verl distillation teacher pool managed by
  Ray resource pools.

For a 4-GPU node, a practical OPD layout is actor/rollout on 3 GPUs and Qwen3-8B
teacher on 1 GPU. If the teacher needs tensor parallelism, use actor/rollout on
2 GPUs with `TEACHER_NGPUS_PER_NODE=2 TEACHER_TP_SIZE=2`.

Evaluate one checkpoint and aggregate result JSON files:

```bash
CHECKPOINT_PATH=/path/to/checkpoint \
RUN_NAME=extpi_rlsd \
CHECKPOINT_NAME=step_100 \
bash recipes/extpi_rlsd/scripts/run_eval_checkpoint.sh

bash recipes/extpi_rlsd/scripts/evaluate_all.sh
```

The matched-dev evaluator defaults to Avg@4 with fixed prompt-seed pairs
`0,1,2,3` and `max_new_tokens=4096`. Override with `EVAL_SEEDS` only when a run
manifest records the change.

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

Training scripts default to console + WandB logging with online syncing:

```bash
TRAINER_LOGGER='["console","wandb"]'
WANDB_MODE=online
```

Use any single-card or multi-GPU training entrypoint directly:

```bash
WANDB_ENTITY=... \
TOTAL_TRAINING_STEPS=5 \
bash recipes/extpi_rlsd/scripts/run_extpi_rlsd.sh
```

For machines without WandB credentials, set `WANDB_MODE=offline` or disable
WandB for that run:

```bash
EXTPI_ENABLE_WANDB=0 bash recipes/extpi_rlsd/scripts/run_grpo.sh
```

You can also pass the verl logger list directly:

```bash
TRAINER_LOGGER='["console"]' bash recipes/extpi_rlsd/scripts/run_grpo.sh
```

Local WandB files are written under
`/data/users/rchen/extpi-rlsd/outputs/wandb` by default through `WANDB_DIR`, so
they stay outside the Git checkout. Credentials such as `WANDB_API_KEY` must be
provided through the environment and are never written to the run manifest.

## Tests

Run local contract tests without starting training:

```bash
python3 -m pytest tests/extpi_rlsd -q
python3 -m ruff check \
  verl/trainer/extpi_rlsd \
  tools/extpi_rlsd \
  recipes/extpi_rlsd/rewards \
  tests/extpi_rlsd \
  verl/workers/config/actor.py \
  verl/workers/utils/losses.py \
  verl/trainer/ppo/ray_trainer.py
```

Entrypoint parsing checks:

```bash
python3 -m verl.trainer.main_ppo --help
python3 -m verl.trainer.extpi_rlsd.main_extpi_rlsd --help
python3 -m verl.trainer.sft_trainer --help
```

## GitHub Sync

The private GitHub remote is:

```text
git@github.com:MemorushB/Extpi-RLSD.git
```

Code, configs, templates, tests, and lightweight manifests are committed.
Datasets, model weights, caches, checkpoints, `wandb`, Ray outputs, and local
run artifacts remain ignored.
