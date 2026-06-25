# ExtPI-RLSD Overlay

This repository is pinned to upstream `volcengine/verl` commit
`cbd7f9f462c0230f4f6161462b5b294c9d55d453` and adds an ExtPI-RLSD research MVP
under `recipes/extpi_rlsd`, `tools/extpi_rlsd`, and `verl/trainer/extpi_rlsd`.

The project default is single-GPU execution on `gpu6`. Large datasets, model
weights, caches, checkpoints, and outputs are stored under
`/data/users/rchen/extpi-rlsd/` and symlinked into the checkout.

Start with:

```bash
bash scripts/setup_local_storage.sh
pytest tests/extpi_rlsd
```

See `recipes/extpi_rlsd/README.md` for the run sequence.
