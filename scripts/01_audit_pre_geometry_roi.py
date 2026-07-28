#!/usr/bin/env python3
"""Step 01: audit Pre CT geometry and basic tumor-burden variables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tprcl.data.pre_audit import audit_pre_cohort  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "step00_manifest"
        / "preonly_labeled_cohort.csv",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "step01_pre_audit",
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    outputs = audit_pre_cohort(
        args.manifest, args.data_root, args.output_dir, strict=args.strict
    )
    report = json.loads(outputs["report"].read_text(encoding="utf-8"))
    print("=" * 72)
    print("Step 01 - Pre CT Geometry and ROI Audit")
    print("=" * 72)
    for name in (
        "manifest_cases",
        "audited_cases",
        "failed_cases",
        "geometry_mismatch_cases",
        "binary_label_counts",
        "errors",
    ):
        print(f"{name:28s}: {report[name]}")
    print(f"{'report':28s}: {outputs['report']}")


if __name__ == "__main__":
    main()
