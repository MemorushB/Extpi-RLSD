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


def collect_run_manifest(
    *,
    config: dict[str, Any],
    output_path: str | Path | None = None,
    dataset_manifest_path: str | Path | None = None,
    model_revisions: dict[str, str | None] | None = None,
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
        },
        "model_revisions": model_revisions or {},
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
