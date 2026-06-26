import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "recipes" / "extpi_rlsd" / "scripts"


def _read_script(name: str) -> str:
    return (SCRIPT_DIR / name).read_text()


def test_single_card_opd_entrypoint_stays_inline():
    text = _read_script("run_opd_pg.sh")
    assert "direct_opd_teacher_backend=inline_external_hf" in text
    assert "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES" in text
    assert "distillation.enabled=True" not in text


def test_recipe_entrypoints_use_configurable_trainer_logger():
    script_names = (
        "run_grpo.sh",
        "run_extpi_rlsd.sh",
        "run_grpo_multi.sh",
        "run_extpi_rlsd_multi.sh",
        "run_closed_sft.sh",
    )
    for script_name in script_names:
        text = _read_script(script_name)
        assert 'trainer.logger="${TRAINER_LOGGER}"' in text
        assert "trainer.logger='[\"console\"]'" not in text


def test_multi_opd_entrypoint_uses_teacher_pool():
    text = _read_script("run_opd_pg_multi_teacher_pool.sh")
    assert "distillation.enabled=True" in text
    assert "distillation.teacher_models.teacher_model.model_path" in text
    assert "direct_opd_teacher_backend=inline_external_hf" not in text
    assert "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1" not in text


def test_env_multi_rejects_single_gpu():
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "6", "NGPUS_PER_NODE": "1", "NPROC_PER_NODE": "1"})
    result = subprocess.run(
        ["bash", "-lc", f"source {SCRIPT_DIR / '_env_multi.sh'}"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "multi scripts expect >=2 GPUs" in result.stderr


def test_env_multi_unsets_single_card_ray_override():
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "NGPUS_PER_NODE": "2",
            "NPROC_PER_NODE": "2",
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
        }
    )
    cmd = (
        f"source {SCRIPT_DIR / '_env_multi.sh'}; "
        'printf "%s:%s:%s" "$NGPUS_PER_NODE" "$NPROC_PER_NODE" '
        '"${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES-unset}"'
    )
    result = subprocess.run(["bash", "-lc", cmd], env=env, capture_output=True, text=True, check=False)

    assert result.returncode == 0
    assert result.stdout == "2:2:unset"


def test_env_single_card_enables_wandb_logger(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "6",
            "NPROC_PER_NODE": "1",
            "NGPUS_PER_NODE": "1",
            "EXTPI_ENABLE_WANDB": "1",
            "EXTPI_DATA_ROOT": str(tmp_path),
        }
    )
    cmd = (
        f"source {SCRIPT_DIR / '_env.sh'}; "
        'printf "%s:%s" "$TRAINER_LOGGER" "$WANDB_DIR"'
    )
    result = subprocess.run(["bash", "-lc", cmd], env=env, capture_output=True, text=True, check=False)

    assert result.returncode == 0
    logger_value, wandb_dir = result.stdout.split(":", 1)
    assert logger_value == '["console","wandb"]'
    assert wandb_dir == str(tmp_path / "outputs" / "wandb")
    assert (tmp_path / "outputs" / "wandb").is_dir()
