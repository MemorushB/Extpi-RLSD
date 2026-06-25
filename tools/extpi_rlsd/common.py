"""Shared utility functions for ExtPI-RLSD scripts."""

from __future__ import annotations

import json
import os
import re
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

DATA_ROOT = Path(os.environ.get("EXTPI_DATA_ROOT", "/data/users/rchen/extpi-rlsd"))


def enforce_gpu6() -> None:
    """Require scripts to run on the project-approved single GPU."""

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in {"6", None, ""}:
        raise SystemExit(f"ExtPI-RLSD scripts may only use gpu6 by default; got CUDA_VISIBLE_DEVICES={visible!r}")
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def normalize_problem(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def stable_problem_id(problem: str) -> str:
    return sha256(normalize_problem(problem).encode("utf-8")).hexdigest()


def compact_text(text: str) -> str:
    return re.sub(r"[\W_]+", "", str(text).lower())


def ngram_set(text: str, n: int = 5) -> set[str]:
    tokens = re.findall(r"\w+", str(text).lower())
    if len(tokens) < n:
        return set(tokens)
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
