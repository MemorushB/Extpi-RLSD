from pathlib import Path


def test_run_opd_does_not_pass_ray_env_via_hydra_int_override():
    text = Path("recipes/extpi_rlsd/scripts/run_opd_pg.sh").read_text()
    assert (
        "ray_kwargs.ray_init.runtime_env.env_vars.RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1"
        not in text
    )
    assert "export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES" in text


def test_ray_runtime_env_forwards_opd_cuda_env_as_string():
    text = Path("verl/trainer/constants_ppo.py").read_text()
    assert "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES" in text
    assert "str(value)" in text
