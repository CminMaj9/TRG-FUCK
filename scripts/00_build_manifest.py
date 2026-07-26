#!/usr/bin/env python3
"""Step 00: rebuild and strictly validate the cohort from raw NIfTI files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tprcl.data import build_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data.example.yaml",
        help="Dataset rules and frozen expected counts",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Override data_root from the config, for example the Windows data folder",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "step00_manifest",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a failure exit code for structural errors or count mismatches",
    )
    args = parser.parse_args()

    outputs = build_manifest(
        args.config,
        args.output_dir,
        data_root=args.data_root,
        strict=args.strict,
    )
    report = json.loads(outputs["report"].read_text(encoding="utf-8"))

    print("=" * 72)
    print("Step 00 - Dataset Finalization")
    print("=" * 72)
    for name, value in report["counts"].items():
        print(f"{name:24s}: {value}")
    print(f"{'errors':24s}: {report['errors']}")
    print(f"{'warnings':24s}: {report['warnings']}")
    print(f"{'report':24s}: {outputs['report']}")


if __name__ == "__main__":
    main()
