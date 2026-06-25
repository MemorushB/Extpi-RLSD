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

The recipe compares Qwen3-1.7B LoRA under a single-GPU setup:

- `closed_sft`: closed-model visible-solution SFT.
- `grpo`: math-verifier GRPO.
- `opd_pg_qwen8b`: Baseline 2, sampled-token OPD-PG/K1 with Qwen3-8B.
- `extpi_rlsd`: GRPO with privileged-trace RLSD reweighting.
- `extpi_rlsd_shuffled`: shuffled privileged-trace control.

The ExtPI-RLSD hook builds scorer-only PI prompts, appends the original student
response token IDs without decode/re-tokenize, scores them under the
base-no-adapter self-teacher, and writes `teacher_pi_log_probs` into the actor
update batch. The actor loss fails fast when RLSD is enabled and that tensor is
missing.

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

Recipe scripts default to gpu6 only:

```bash
CUDA_VISIBLE_DEVICES=6
NPROC_PER_NODE=1
NGPUS_PER_NODE=1
```

The scripts exit if a different GPU is selected unless the script is
intentionally edited.

## Data Flow

Prepare OPSD, generate verified Qwen3-8B PI traces, attach recipient uplift
statistics, and build frontier splits:

```bash
bash recipes/extpi_rlsd/scripts/00_prepare_dataset.sh
bash recipes/extpi_rlsd/scripts/01_generate_qwen8b_pi.sh
python3 tools/extpi_rlsd/compute_recipient_uplift.py \
  --input_jsonl /data/users/rchen/extpi-rlsd/datasets/opsd_clean/all_clean_qwen8b_pi.jsonl \
  --plain_jsonl /path/to/plain_completions.jsonl \
  --pi_jsonl /path/to/pi_completions.jsonl \
  --output_jsonl /data/users/rchen/extpi-rlsd/datasets/opsd_clean/all_clean_qwen8b_pi_screened.jsonl
bash recipes/extpi_rlsd/scripts/03_build_frontier.sh
```

Closed-teacher SFT data uses an OpenAI-compatible endpoint:

```bash
export CLOSED_TEACHER_BASE_URL=...
export CLOSED_TEACHER_API_KEY=...
export CLOSED_TEACHER_MODEL=...
bash recipes/extpi_rlsd/scripts/02_generate_closed_sft.sh
```

## Run Entrypoints

These commands are the intended smoke entrypoints once data and dependencies are
ready. They do not bypass the gpu6 guard.

```bash
TOTAL_TRAINING_STEPS=5 bash recipes/extpi_rlsd/scripts/run_grpo.sh
TOTAL_TRAINING_STEPS=5 bash recipes/extpi_rlsd/scripts/run_opd_pg.sh
TOTAL_TRAINING_STEPS=5 bash recipes/extpi_rlsd/scripts/run_extpi_rlsd.sh
TOTAL_TRAINING_STEPS=5 bash recipes/extpi_rlsd/scripts/run_extpi_rlsd_shuffled.sh
```

`run_opd_pg.sh` is the Baseline 2 entrypoint. `run_extpi_rlsd.sh` uses
`verl.trainer.extpi_rlsd.main_extpi_rlsd`, which runs the legacy PPO trainer
hook that materializes `teacher_pi_log_probs`.

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
