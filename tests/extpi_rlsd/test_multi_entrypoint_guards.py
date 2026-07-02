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
        "run_closed_sft_multi.sh",
    )
    for script_name in script_names:
        text = _read_script(script_name)
        assert 'trainer.logger="${TRAINER_LOGGER}"' in text
        assert "trainer.logger='[\"console\"]'" not in text


def test_actor_yaml_has_extpi_policy_loss_keys():
    text = (ROOT / "verl" / "trainer" / "config" / "actor" / "actor.yaml").read_text()
    for key in (
        "rlsd_enabled",
        "rlsd_lambda",
        "rlsd_lambda_warmup_steps",
        "rlsd_lambda_decay_steps",
        "rlsd_reweight_clip_range",
        "rlsd_negative_only",
        "rlsd_token_mask",
        "rlsd_delta_student_logp_source",
        "opd_pg_enabled",
        "opd_logratio_clip_abs",
    ):
        assert key in text


def test_online_scripts_set_rollout_logprob_microbatch():
    for script_name in ("run_grpo.sh", "run_extpi_rlsd.sh", "run_grpo_multi.sh", "run_extpi_rlsd_multi.sh"):
        text = _read_script(script_name)
        assert "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu" in text


def test_online_scripts_set_current_and_legacy_reward_paths():
    for script_name in ("run_grpo.sh", "run_extpi_rlsd.sh", "run_grpo_multi.sh", "run_extpi_rlsd_multi.sh"):
        text = _read_script(script_name)
        assert "custom_reward_function.path=recipes/extpi_rlsd/rewards/math_verify_reward.py" in text
        assert "custom_reward_function.name=compute_score" in text
        assert "reward.custom_reward_function.path=recipes/extpi_rlsd/rewards/math_verify_reward.py" in text
        assert "reward.custom_reward_function.name=compute_score" in text


def test_compute_ref_log_prob_assigns_temperature():
    text = (ROOT / "verl" / "trainer" / "ppo" / "ray_trainer.py").read_text()
    assert '"temperature": rollout_config.temperature' in text


def test_multi_opd_entrypoint_uses_teacher_pool():
    text = _read_script("run_opd_pg_multi_teacher_pool.sh")
    assert "distillation.enabled=True" in text
    assert "distillation.teacher_models.teacher_model.model_path" in text
    assert "direct_opd_teacher_backend=inline_external_hf" not in text
    assert "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1" not in text


def test_env_multi_has_no_gpu6_guard_and_unsets_single_card_ray_override():
    text = _read_script("_env_multi.sh")
    assert 'CUDA_VISIBLE_DEVICES}" != "6"' not in text
    assert "unset RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES" in text


def test_extpi_multi_allows_configured_teacher_modes():
    text = _read_script("run_extpi_rlsd_multi.sh")
    assert 'TEACHER_UPDATE_MODE="${TEACHER_UPDATE_MODE:-base_no_adapter}"' in text
    assert "current_no_grad" in text
    assert "periodic_snapshot" in text
    assert 'actor_rollout_ref.model.extpi_teacher_adapter="${EXTPI_TEACHER_ADAPTER}"' in text


def test_extpi_scripts_expose_teacher_modes_and_lambda_schedule():
    for script_name in ("run_extpi_rlsd.sh", "run_extpi_rlsd_multi.sh"):
        text = _read_script(script_name)
        assert 'RLSD_LAMBDA_WARMUP_STEPS="${RLSD_LAMBDA_WARMUP_STEPS:-0}"' in text
        assert 'RLSD_LAMBDA_DECAY_STEPS="${RLSD_LAMBDA_DECAY_STEPS:-0}"' in text
        assert 'TEACHER_SYNC_INTERVAL="${TEACHER_SYNC_INTERVAL:-20}"' in text
        assert 'actor_rollout_ref.actor.policy_loss.rlsd_lambda_warmup_steps="${RLSD_LAMBDA_WARMUP_STEPS}"' in text
        assert 'actor_rollout_ref.actor.policy_loss.rlsd_lambda_decay_steps="${RLSD_LAMBDA_DECAY_STEPS}"' in text
        assert 'actor_rollout_ref.actor.policy_loss.rlsd_token_mask="${RLSD_TOKEN_MASK}"' in text
        assert '+extpi_rlsd.teacher_sync_interval="${TEACHER_SYNC_INTERVAL}"' in text
        assert '+extpi_rlsd.pi_trace_field="${PI_TRACE_FIELD}"' in text


def test_hparam_launcher_runs_one_id_with_file_logger():
    text = _read_script("launch_hparam_matrix.sh")
    assert "RUN_ID" in text
    assert "export-env" in text
    assert 'TRAINER_LOGGER="${TRAINER_LOGGER:-[\\"console\\",\\"wandb\\",\\"file\\"]}"' in text
    assert "run_official_math_eval.sh" not in text


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


def test_env_single_card_enables_wandb_logger_by_default(tmp_path):
    env = os.environ.copy()
    env.pop("WANDB_MODE", None)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "6",
            "NPROC_PER_NODE": "1",
            "NGPUS_PER_NODE": "1",
            "EXTPI_DATA_ROOT": str(tmp_path),
        }
    )
    cmd = (
        f"source {SCRIPT_DIR / '_env.sh'}; "
        'printf "%s:%s:%s" "$TRAINER_LOGGER" "$WANDB_DIR" "$WANDB_MODE"'
    )
    result = subprocess.run(["bash", "-lc", cmd], env=env, capture_output=True, text=True, check=False)

    assert result.returncode == 0
    logger_value, wandb_dir, wandb_mode = result.stdout.split(":", 2)
    assert logger_value == '["console","wandb"]'
    assert wandb_dir == str(tmp_path / "outputs" / "wandb")
    assert wandb_mode == "online"
    assert (tmp_path / "outputs" / "wandb").is_dir()


def test_env_single_card_preserves_explicit_wandb_mode(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "6",
            "NPROC_PER_NODE": "1",
            "NGPUS_PER_NODE": "1",
            "EXTPI_DATA_ROOT": str(tmp_path),
            "WANDB_MODE": "offline",
        }
    )
    cmd = f"source {SCRIPT_DIR / '_env.sh'}; printf \"%s\" \"$WANDB_MODE\""
    result = subprocess.run(["bash", "-lc", cmd], env=env, capture_output=True, text=True, check=False)

    assert result.returncode == 0
    assert result.stdout == "offline"


def test_env_single_card_can_disable_default_wandb(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "6",
            "NPROC_PER_NODE": "1",
            "NGPUS_PER_NODE": "1",
            "EXTPI_ENABLE_WANDB": "0",
            "EXTPI_DATA_ROOT": str(tmp_path),
        }
    )
    cmd = f"source {SCRIPT_DIR / '_env.sh'}; printf \"%s\" \"$TRAINER_LOGGER\""
    result = subprocess.run(["bash", "-lc", cmd], env=env, capture_output=True, text=True, check=False)

    assert result.returncode == 0
    assert result.stdout == '["console"]'
    assert not (tmp_path / "outputs" / "wandb").exists()


def test_env_multi_enables_wandb_logger_by_default(tmp_path):
    env = os.environ.copy()
    env.pop("WANDB_MODE", None)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "NPROC_PER_NODE": "2",
            "NGPUS_PER_NODE": "2",
            "EXTPI_DATA_ROOT": str(tmp_path),
        }
    )
    cmd = (
        f"source {SCRIPT_DIR / '_env_multi.sh'}; "
        'printf "%s:%s:%s" "$TRAINER_LOGGER" "$WANDB_DIR" "$WANDB_MODE"'
    )
    result = subprocess.run(["bash", "-lc", cmd], env=env, capture_output=True, text=True, check=False)

    assert result.returncode == 0
    logger_value, wandb_dir, wandb_mode = result.stdout.split(":", 2)
    assert logger_value == '["console","wandb"]'
    assert wandb_dir == str(tmp_path / "outputs" / "wandb")
    assert wandb_mode == "online"
    assert (tmp_path / "outputs" / "wandb").is_dir()


def test_env_single_card_unsets_cuda_allocator_by_default(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "6",
            "NPROC_PER_NODE": "1",
            "NGPUS_PER_NODE": "1",
            "EXTPI_ENABLE_WANDB": "0",
            "EXTPI_DATA_ROOT": str(tmp_path),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    cmd = f"source {SCRIPT_DIR / '_env.sh'}; printf \"%s\" \"${{PYTORCH_CUDA_ALLOC_CONF-unset}}\""
    result = subprocess.run(["bash", "-lc", cmd], env=env, capture_output=True, text=True, check=False)

    assert result.returncode == 0
    assert result.stdout == "unset"


def test_env_single_card_allows_forced_cuda_allocator(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "6",
            "NPROC_PER_NODE": "1",
            "NGPUS_PER_NODE": "1",
            "EXTPI_ENABLE_WANDB": "0",
            "EXTPI_DATA_ROOT": str(tmp_path),
            "EXTPI_FORCE_PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
        }
    )
    cmd = f"source {SCRIPT_DIR / '_env.sh'}; printf \"%s\" \"$PYTORCH_CUDA_ALLOC_CONF\""
    result = subprocess.run(["bash", "-lc", cmd], env=env, capture_output=True, text=True, check=False)

    assert result.returncode == 0
    assert result.stdout == "max_split_size_mb:128"


def test_env_multi_unsets_cuda_allocator_by_default(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "NPROC_PER_NODE": "2",
            "NGPUS_PER_NODE": "2",
            "EXTPI_ENABLE_WANDB": "0",
            "EXTPI_DATA_ROOT": str(tmp_path),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    cmd = f"source {SCRIPT_DIR / '_env_multi.sh'}; printf \"%s\" \"${{PYTORCH_CUDA_ALLOC_CONF-unset}}\""
    result = subprocess.run(["bash", "-lc", cmd], env=env, capture_output=True, text=True, check=False)

    assert result.returncode == 0
    assert result.stdout == "unset"


def test_env_multi_allows_forced_cuda_allocator(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "NPROC_PER_NODE": "2",
            "NGPUS_PER_NODE": "2",
            "EXTPI_ENABLE_WANDB": "0",
            "EXTPI_DATA_ROOT": str(tmp_path),
            "EXTPI_FORCE_PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
        }
    )
    cmd = f"source {SCRIPT_DIR / '_env_multi.sh'}; printf \"%s\" \"$PYTORCH_CUDA_ALLOC_CONF\""
    result = subprocess.run(["bash", "-lc", cmd], env=env, capture_output=True, text=True, check=False)

    assert result.returncode == 0
    assert result.stdout == "max_split_size_mb:128"
