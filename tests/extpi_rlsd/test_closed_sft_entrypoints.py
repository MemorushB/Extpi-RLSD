from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "recipes" / "extpi_rlsd" / "scripts"


def _read_script(name: str) -> str:
    return (SCRIPT_DIR / name).read_text()


def test_single_closed_sft_entrypoint_uses_torchrun_and_model_path():
    text = _read_script("run_closed_sft.sh")

    assert "torchrun --standalone" in text
    assert "-m verl.trainer.sft_trainer" in text
    assert 'model.path="${MODEL_PATH}"' in text
    assert "model.partial_pretrain" not in text
    assert 'trainer.n_gpus_per_node="${NGPUS_PER_NODE}"' in text
    assert 'trainer.nnodes="${NNODES:-1}"' in text


def test_multi_closed_sft_entrypoint_uses_multi_env_and_torchrun():
    path = SCRIPT_DIR / "run_closed_sft_multi.sh"
    assert path.exists()
    text = path.read_text()

    assert 'source "${SCRIPT_DIR}/_env_multi.sh"' in text
    assert "torchrun --standalone" in text
    assert "-m verl.trainer.sft_trainer" in text
    assert 'trainer.n_gpus_per_node="${NGPUS_PER_NODE}"' in text
    assert 'trainer.nnodes="${NNODES}"' in text
    assert 'model.path="${MODEL_PATH}"' in text
    assert "model.partial_pretrain" not in text
    assert "WORLD_SIZE=$((NGPUS_PER_NODE * NNODES))" in text
    assert "SFT_TRAIN_BATCH_SIZE must be divisible by WORLD_SIZE" in text


def test_closed_sft_entrypoints_run_parquet_preflight():
    for script_name in ("run_closed_sft.sh", "run_closed_sft_multi.sh"):
        text = _read_script(script_name)
        assert "tools/extpi_rlsd/inspect_sft_parquet.py" in text
        assert "--fail_on_overlong" in text
        assert 'SFT_PREFLIGHT="${SFT_PREFLIGHT:-1}"' in text
