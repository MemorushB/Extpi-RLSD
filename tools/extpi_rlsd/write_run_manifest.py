#!/usr/bin/env python3
"""Write a lightweight run manifest for an ExtPI-RLSD command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verl.trainer.extpi_rlsd.manifest import collect_run_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_json", default=None)
    parser.add_argument("--dataset_manifest", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = {}
    if args.config_json:
        config = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
    collect_run_manifest(
        config=config,
        dataset_manifest_path=args.dataset_manifest,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
