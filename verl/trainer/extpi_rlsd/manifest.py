"""Run manifest collection for reproducible ExtPI-RLSD runs."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    """Hash a file in chunks."""

    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], cwd: str | Path | None = None) -> str | None:
    try:
        return subprocess.check_output(command, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _json_hash(payload: Any) -> str:
    return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _torch_cuda_metadata() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"error": repr(exc)}

    payload: dict[str, Any] = {
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        payload["devices"] = [
            {
                "index": idx,
                "name": torch.cuda.get_device_name(idx),
                "capability": torch.cuda.get_device_capability(idx),
            }
            for idx in range(torch.cuda.device_count())
        ]
    return payload


def _model_metadata(model_path: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"path": model_path}
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        payload["revision"] = getattr(config, "_commit_hash", None)
        payload["model_type"] = getattr(config, "model_type", None)
    except Exception as exc:
        payload["config_error"] = repr(exc)
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        special_tokens = getattr(tokenizer, "special_tokens_map", {})
        chat_template = getattr(tokenizer, "chat_template", None)
        tokenizer_payload = {
            "name_or_path": getattr(tokenizer, "name_or_path", None),
            "revision": getattr(tokenizer, "_commit_hash", None),
            "vocab_size": getattr(tokenizer, "vocab_size", None),
            "bos_token_id": getattr(tokenizer, "bos_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "special_tokens_map": special_tokens,
            "chat_template_sha256": sha256(chat_template.encode("utf-8")).hexdigest() if chat_template else None,
            "compatibility_hash": _json_hash(
                {
                    "vocab_size": getattr(tokenizer, "vocab_size", None),
                    "bos_token_id": getattr(tokenizer, "bos_token_id", None),
                    "eos_token_id": getattr(tokenizer, "eos_token_id", None),
                    "pad_token_id": getattr(tokenizer, "pad_token_id", None),
                    "special_tokens_map": special_tokens,
                    "chat_template": chat_template,
                }
            ),
        }
        payload["tokenizer"] = tokenizer_payload
    except Exception as exc:
        payload["tokenizer_error"] = repr(exc)
    return payload


def collect_run_manifest(
    *,
    config: dict[str, Any],
    output_path: str | Path | None = None,
    dataset_manifest_path: str | Path | None = None,
    model_revisions: dict[str, str | None] | None = None,
    model_paths: dict[str, str] | None = None,
    seed: int | None = None,
    cwd: str | Path = ".",
) -> dict[str, Any]:
    """Collect run provenance into a JSON-serializable manifest."""

    cwd = Path(cwd)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": _run(["git", "rev-parse", "HEAD"], cwd=cwd),
            "dirty_state": _run(["git", "status", "--short"], cwd=cwd),
        },
        "packages": {
            "verl": _package_version("verl"),
            "transformers": _package_version("transformers"),
            "vllm": _package_version("vllm"),
            "torch": _package_version("torch"),
            "peft": _package_version("peft"),
        },
        "system": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "nvidia_smi": _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
            "nvidia_smi_gpu_query": _run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,uuid,driver_version,memory.total",
                    "--format=csv,noheader,nounits",
                ]
            ),
            "cuda": _torch_cuda_metadata(),
        },
        "model_revisions": model_revisions or {},
        "models": {name: _model_metadata(path) for name, path in (model_paths or {}).items()},
        "seed": seed,
        "config": config,
    }
    if dataset_manifest_path:
        dataset_manifest_path = Path(dataset_manifest_path)
        manifest["dataset_manifest"] = {
            "path": str(dataset_manifest_path),
            "sha256": sha256_file(dataset_manifest_path) if dataset_manifest_path.exists() else None,
        }
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
