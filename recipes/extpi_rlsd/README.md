# ExtPI-RLSD MVP Recipe

This recipe compares Qwen3-1.7B LoRA under:

- `closed_sft`: closed-model visible-solution SFT.
- `grpo`: math verifier GRPO.
- `opd_pg_qwen8b`: sampled-token OPD-PG/K1 with Qwen3-8B.
- `extpi_rlsd`: GRPO with privileged-trace RLSD reweighting.
- `extpi_rlsd_shuffled`: shuffled privileged-trace control.

The current MVP includes the reusable data tools, exact-token scorer helpers,
worker-side RLSD loss support, and safety tests. The remaining online hook is
to materialize `teacher_pi_log_probs` after rollout and before actor update.
The actor loss already fails fast when `rlsd_enabled=True` and that tensor is
missing, so there is no silent GRPO fallback.

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
bash recipes/extpi_rlsd/scripts/00_prepare_dataset.sh
bash recipes/extpi_rlsd/scripts/01_generate_qwen8b_pi.sh
bash recipes/extpi_rlsd/scripts/03_build_frontier.sh
```

Closed-teacher SFT data requires:

```bash
export CLOSED_TEACHER_BASE_URL=...
export CLOSED_TEACHER_API_KEY=...
export CLOSED_TEACHER_MODEL=...
bash recipes/extpi_rlsd/scripts/02_generate_closed_sft.sh
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

ExtPI-RLSD currently requires an inline scorer hook that writes
`teacher_pi_log_probs` into the update batch. The loss contract and tests are in
place:

```bash
pytest tests/extpi_rlsd
```

## Public Claims Boundary

Use the label `OPD-PG (sampled-token K1)` for the direct distillation baseline.
Do not claim full-vocabulary or top-k GKD results unless those baselines are
actually run.
