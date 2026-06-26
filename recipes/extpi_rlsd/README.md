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

Recipe scripts default to:

```bash
CUDA_VISIBLE_DEVICES=6
NPROC_PER_NODE=1
NGPUS_PER_NODE=1
```

They exit if another GPU is selected unless the scripts are intentionally
changed.

## Data Flow

```bash
bash recipes/extpi_rlsd/scripts/00_prepare_dataset.sh \
  --eval_jsonl /path/to/aime24.jsonl \
  --eval_jsonl /path/to/aime25.jsonl \
  --eval_jsonl /path/to/hmmt25.jsonl
bash recipes/extpi_rlsd/scripts/01_generate_qwen8b_pi.sh
# Generate plain/PI recipient completion JSONL files externally, then attach uplift:
python3 tools/extpi_rlsd/compute_recipient_uplift.py \
  --input_jsonl /data/users/rchen/extpi-rlsd/datasets/opsd_clean/all_clean_qwen8b_pi.jsonl \
  --plain_jsonl /path/to/plain_completions.jsonl \
  --pi_jsonl /path/to/pi_completions.jsonl \
  --output_jsonl /data/users/rchen/extpi-rlsd/datasets/opsd_clean/all_clean_qwen8b_pi_screened.jsonl
bash recipes/extpi_rlsd/scripts/03_build_frontier.sh
```

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

This baseline uses verl's distillation teacher-service path for sampled-token
K1 OPD-PG and runs a tokenizer compatibility preflight. It is not a
full-vocabulary or top-k GKD baseline.

ExtPI-RLSD smoke:

```bash
TOTAL_TRAINING_STEPS=5 bash recipes/extpi_rlsd/scripts/run_extpi_rlsd.sh
```

The dedicated entry point is `verl.trainer.extpi_rlsd.main_extpi_rlsd`; it uses
the legacy PPO trainer because the MVP hook is implemented there.

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

## Public Claims Boundary

Use the label `OPD-PG (sampled-token K1)` for the direct distillation baseline.
Do not claim full-vocabulary or top-k GKD results unless those baselines are
actually run.
