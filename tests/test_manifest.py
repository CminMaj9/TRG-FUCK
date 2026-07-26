from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from tprcl.data.manifest import (
    FilenameRules,
    ManifestValidationError,
    build_manifest,
    normalized_file_key,
    parse_filename,
    strip_nifti_suffix,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _make_dataset(
    root: Path,
    *,
    trg_values: list[int | None],
    add_unassigned: bool = True,
) -> None:
    for index, trg in enumerate(trg_values, start=1):
        label = f"-TRG{trg}" if trg is not None else ""
        for suffix in ("_000", "_001"):
            stem = f"CASE{index:03d}{label}{suffix}"
            _touch(root / "imagesTr" / f"{stem}_0000.nii.gz")
            _touch(root / "labelsTr" / f"{stem}.nii.gz")
            _touch(root / "labelsTr_old" / f"{stem}.nii.gz")
    if add_unassigned:
        _touch(root / "imagesTr" / "EXTRA-TRG3_0000.nii.gz")
        _touch(root / "labelsTr" / "EXTRA-TRG3.nii.gz")


def _write_config(root: Path, expected: dict[str, object]) -> Path:
    config = {
        "schema_version": 1,
        "data_root": str(root),
        "directories": {
            "images": "imagesTr",
            "masks": "labelsTr",
            "old_masks": "labelsTr_old",
        },
        "filename_rules": {
            "image_modality_suffix": "_0000",
            "pre_suffix": "_000",
            "post_suffix": "_001",
        },
        "analysis": {"mask_source": "labelsTr", "responder_trg": [0, 1]},
        "expected": expected,
    }
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_strip_nifti_suffix_is_case_insensitive() -> None:
    assert strip_nifti_suffix("case.NII.GZ") == "case"
    assert strip_nifti_suffix("case.nii") == "case"
    with pytest.raises(ValueError, match="Not a NIfTI"):
        strip_nifti_suffix("case.csv")


def test_image_and_mask_names_share_one_pairing_key() -> None:
    assert normalized_file_key("CASE-TRG2_000_0000.nii.gz", "_0000") == (
        normalized_file_key("CASE-TRG2_000.nii.gz", "_0000")
    )


def test_parser_uses_only_explicit_timepoint_rules() -> None:
    rules = FilenameRules()
    pre = parse_filename("CASE001-TRG0_000_0000", rules)
    post = parse_filename("CASE001-TRG0_001", rules)
    unknown = parse_filename("CASE001-TRG0", rules)

    assert (pre.patient_key, pre.timepoint, pre.trg) == ("case001", "pre", 0)
    assert (post.patient_key, post.timepoint, post.trg) == ("case001", "post", 0)
    assert unknown.timepoint is None
    assert "unassigned_timepoint" in unknown.problems


def test_manifest_builds_labeled_preonly_cohort(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _make_dataset(data_root, trg_values=[0, 1, 2, 3, None])
    expected = {
        "image_files": 11,
        "mask_files": 11,
        "old_mask_files": 10,
        "image_mask_pairs": 11,
        "assigned_timepoints": 10,
        "unassigned_timepoints": 1,
        "complete_patients": 5,
        "labeled_patients": 4,
        "unlabeled_patients": 1,
        "trg_counts": {"0": 1, "1": 1, "2": 1, "3": 1},
        "binary_counts": {"0": 2, "1": 2},
    }
    outputs = build_manifest(
        _write_config(data_root, expected),
        tmp_path / "outputs",
        strict=True,
    )

    report = json.loads(outputs["report"].read_text(encoding="utf-8"))
    timepoints = pd.read_csv(outputs["timepoints"])
    preonly = pd.read_csv(outputs["preonly"])
    unassigned = pd.read_csv(outputs["unassigned"])

    assert report["counts"] == expected
    assert report["errors"] == 0
    assert len(timepoints) == 10
    assert set(timepoints["timepoint"]) == {"pre", "post"}
    assert len(preonly) == 4
    assert "timepoint" not in preonly.columns
    assert preonly["image_relpath"].str.contains(r"_000_0000\.nii\.gz$").all()
    assert set(preonly["binary_label"]) == {0, 1}
    assert len(unassigned) == 1


def test_strict_mode_rejects_inconsistent_longitudinal_trg(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _touch(data_root / "imagesTr" / "CASE001-TRG0_000_0000.nii.gz")
    _touch(data_root / "labelsTr" / "CASE001-TRG0_000.nii.gz")
    _touch(data_root / "imagesTr" / "CASE001-TRG3_001_0000.nii.gz")
    _touch(data_root / "labelsTr" / "CASE001-TRG3_001.nii.gz")
    config_path = _write_config(data_root, {})

    with pytest.raises(ManifestValidationError, match="Strict manifest"):
        build_manifest(config_path, tmp_path / "outputs", strict=True)

    issues = pd.read_csv(tmp_path / "outputs" / "manifest_issues.csv")
    assert "inconsistent_longitudinal_trg" in set(issues["code"])


def test_strict_mode_rejects_expected_count_mismatch(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _make_dataset(data_root, trg_values=[0], add_unassigned=False)
    config_path = _write_config(data_root, {"complete_patients": 999})

    with pytest.raises(ManifestValidationError, match="Strict manifest"):
        build_manifest(config_path, tmp_path / "outputs", strict=True)

    report = json.loads(
        (tmp_path / "outputs" / "manifest_report.json").read_text(encoding="utf-8")
    )
    assert report["expected_mismatches"] == [
        "complete_patients: actual=1, expected=999"
    ]
