"""Build an auditable cohort manifest directly from raw NIfTI filenames."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

NIFTI_SUFFIXES = (".nii.gz", ".nii")
TRG_PATTERN = re.compile(r"(?i)(?<![A-Za-z0-9])TRG[_-]?([0-9]+)(?![A-Za-z0-9])")
EXPLICIT_TIMEPOINT_PATTERN = re.compile(
    r"(?i)(?<![A-Za-z0-9])(pre|post)(?![A-Za-z0-9])"
)

TIMEPOINT_COLUMNS = [
    "research_id",
    "timepoint",
    "trg",
    "binary_label",
    "image_relpath",
    "mask_relpath",
    "old_mask_relpath",
    "analysis_mask_source",
]
PATIENT_COLUMNS = [
    "research_id",
    "trg",
    "binary_label",
    "pre_image_relpath",
    "pre_mask_relpath",
    "post_image_relpath",
    "post_mask_relpath",
    "analysis_mask_source",
]
PREONLY_COLUMNS = [
    "research_id",
    "trg",
    "binary_label",
    "image_relpath",
    "mask_relpath",
    "analysis_mask_source",
]
UNASSIGNED_COLUMNS = [
    "file_key",
    "trg",
    "image_relpath",
    "mask_relpath",
    "old_mask_relpath",
    "reason",
]
ISSUE_COLUMNS = ["severity", "code", "file_key", "details"]


class ManifestValidationError(RuntimeError):
    """Raised when strict manifest validation finds an inconsistency."""


@dataclass(frozen=True)
class FilenameRules:
    """Filename suffixes used by the current dataset."""

    image_modality_suffix: str = "_0000"
    pre_suffix: str = "_000"
    post_suffix: str = "_001"


@dataclass(frozen=True)
class ParsedName:
    """Information parsed from one normalized NIfTI stem."""

    source_stem: str
    patient_key: str | None
    timepoint: str | None
    trg: int | None
    problems: tuple[str, ...]


def strip_nifti_suffix(path: str | Path) -> str:
    """Return a filename without its case-insensitive NIfTI suffix."""

    name = Path(path).name
    lowered = name.casefold()
    for suffix in NIFTI_SUFFIXES:
        if lowered.endswith(suffix):
            return name[: -len(suffix)]
    raise ValueError(f"Not a NIfTI filename: {name}")


def discover_nifti_files(directory: Path) -> list[Path]:
    """Recursively discover NIfTI files in deterministic order."""

    if not directory.is_dir():
        raise FileNotFoundError(f"NIfTI directory not found: {directory}")
    return sorted(
        path.resolve()
        for path in directory.rglob("*")
        if path.is_file() and path.name.casefold().endswith(NIFTI_SUFFIXES)
    )


def normalized_file_key(path: str | Path, image_modality_suffix: str) -> str:
    """Normalize an image or mask name to the shared pairing key."""

    stem = strip_nifti_suffix(path)
    if image_modality_suffix and stem.casefold().endswith(
        image_modality_suffix.casefold()
    ):
        stem = stem[: -len(image_modality_suffix)]
    return stem.casefold()


def parse_filename(stem: str, rules: FilenameRules) -> ParsedName:
    """Parse patient, timepoint and TRG without guessing unknown timepoints."""

    source_stem = stem
    if rules.image_modality_suffix and stem.casefold().endswith(
        rules.image_modality_suffix.casefold()
    ):
        stem = stem[: -len(rules.image_modality_suffix)]

    problems: list[str] = []
    suffix_timepoint: str | None = None
    suffix_to_remove = ""
    if rules.pre_suffix and stem.casefold().endswith(rules.pre_suffix.casefold()):
        suffix_timepoint = "pre"
        suffix_to_remove = rules.pre_suffix
    if rules.post_suffix and stem.casefold().endswith(rules.post_suffix.casefold()):
        if suffix_timepoint is not None:
            problems.append("ambiguous_timepoint_suffix")
        suffix_timepoint = "post"
        suffix_to_remove = rules.post_suffix

    explicit_timepoints = {
        match.group(1).casefold() for match in EXPLICIT_TIMEPOINT_PATTERN.finditer(stem)
    }
    if len(explicit_timepoints) > 1:
        problems.append("ambiguous_explicit_timepoint")
        explicit_timepoint = None
    else:
        explicit_timepoint = next(iter(explicit_timepoints), None)

    if (
        suffix_timepoint is not None
        and explicit_timepoint is not None
        and suffix_timepoint != explicit_timepoint
    ):
        problems.append("conflicting_timepoint_markers")
        timepoint = None
    else:
        timepoint = suffix_timepoint or explicit_timepoint
    if timepoint is None and not any(
        problem.startswith(("ambiguous", "conflicting")) for problem in problems
    ):
        problems.append("unassigned_timepoint")

    trg_values = {int(match.group(1)) for match in TRG_PATTERN.finditer(stem)}
    if any(value not in {0, 1, 2, 3} for value in trg_values):
        problems.append("invalid_trg")
        trg = None
    elif len(trg_values) > 1:
        problems.append("ambiguous_trg")
        trg = None
    else:
        trg = next(iter(trg_values), None)

    patient_stem = stem
    if suffix_to_remove:
        patient_stem = patient_stem[: -len(suffix_to_remove)]
    patient_stem = EXPLICIT_TIMEPOINT_PATTERN.sub("-", patient_stem)
    patient_stem = TRG_PATTERN.sub("-", patient_stem)
    patient_stem = re.sub(r"[\s_-]+", "-", patient_stem).strip("-")
    patient_key = patient_stem.casefold() or None
    if patient_key is None:
        problems.append("empty_patient_key")

    return ParsedName(
        source_stem=source_stem,
        patient_key=patient_key,
        timepoint=timepoint,
        trg=trg,
        problems=tuple(problems),
    )


def _load_config(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")
    if config.get("schema_version") != 1:
        raise ValueError("Only config schema_version=1 is supported")
    return config


def _resolve_data_root(config: dict[str, Any], override: str | Path | None) -> Path:
    raw = str(override) if override is not None else str(config.get("data_root", ""))
    expanded = os.path.expandvars(os.path.expanduser(raw))
    if not expanded or "$" in expanded or "%" in expanded:
        raise ValueError(
            "Data root is unresolved. Pass --data-root or set GC_TRG_DATA_ROOT."
        )
    root = Path(expanded).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Data root not found: {root}")
    return root


def _relative(path: Path | None, root: Path) -> str:
    if path is None:
        return ""
    return path.relative_to(root).as_posix()


def _index_files(
    paths: list[Path],
    source: str,
    rules: FilenameRules,
    issues: list[dict[str, str]],
) -> dict[str, Path]:
    grouped: dict[str, list[Path]] = {}
    for path in paths:
        key = normalized_file_key(path, rules.image_modality_suffix)
        grouped.setdefault(key, []).append(path)

    index: dict[str, Path] = {}
    for key, matches in sorted(grouped.items()):
        if len(matches) == 1:
            index[key] = matches[0]
        else:
            issues.append(
                {
                    "severity": "error",
                    "code": "duplicate_normalized_file_key",
                    "file_key": key,
                    "details": f"{source}: {[str(path) for path in matches]}",
                }
            )
    return index


def _distribution(series: pd.Series) -> dict[str, int]:
    numeric = pd.to_numeric(series, errors="coerce").dropna().astype(int)
    return {
        str(key): int(value)
        for key, value in numeric.value_counts().sort_index().items()
    }


def _expected_mismatches(actual: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        if key not in actual:
            mismatches.append(f"Unknown expected count: {key}")
            continue
        actual_value = actual[key]
        if isinstance(expected_value, dict):
            normalized_expected = {
                str(item_key): int(item_value)
                for item_key, item_value in expected_value.items()
            }
            if actual_value != normalized_expected:
                mismatches.append(
                    f"{key}: actual={actual_value}, expected={normalized_expected}"
                )
        elif int(actual_value) != int(expected_value):
            mismatches.append(
                f"{key}: actual={actual_value}, expected={expected_value}"
            )
    return mismatches


def _write_csv(rows: list[dict[str, Any]], columns: list[str], path: Path) -> None:
    frame = pd.DataFrame(rows, columns=columns)
    for column in ("trg", "binary_label"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(
                "Int64"
            )
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def build_manifest(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    data_root: str | Path | None = None,
    strict: bool = False,
) -> dict[str, Path]:
    """Build and validate the current longitudinal and Pre-only cohorts."""

    config_path = Path(config_path).resolve()
    config = _load_config(config_path)
    root = _resolve_data_root(config, data_root)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    directories = config.get("directories", {})
    images_dir = root / str(directories.get("images", "imagesTr"))
    masks_dir = root / str(directories.get("masks", "labelsTr"))
    old_masks_name = directories.get("old_masks")
    old_masks_dir = root / str(old_masks_name) if old_masks_name else None

    filename_config = config.get("filename_rules", {})
    rules = FilenameRules(
        image_modality_suffix=str(
            filename_config.get("image_modality_suffix", "_0000")
        ),
        pre_suffix=str(filename_config.get("pre_suffix", "_000")),
        post_suffix=str(filename_config.get("post_suffix", "_001")),
    )
    analysis = config.get("analysis", {})
    mask_source = str(analysis.get("mask_source", "labelsTr"))
    responder_trg = {int(value) for value in analysis.get("responder_trg", [0, 1])}
    if not responder_trg <= {0, 1, 2, 3}:
        raise ValueError("analysis.responder_trg must contain only TRG values 0-3")

    issues: list[dict[str, str]] = []
    image_paths = discover_nifti_files(images_dir)
    mask_paths = discover_nifti_files(masks_dir)
    old_mask_paths = (
        discover_nifti_files(old_masks_dir)
        if old_masks_dir is not None and old_masks_dir.is_dir()
        else []
    )
    image_index = _index_files(image_paths, "images", rules, issues)
    mask_index = _index_files(mask_paths, "masks", rules, issues)
    old_mask_index = _index_files(old_mask_paths, "old_masks", rules, issues)

    paired_rows: list[dict[str, Any]] = []
    unassigned_rows: list[dict[str, Any]] = []
    all_keys = sorted(set(image_index) | set(mask_index))
    for key in all_keys:
        image_path = image_index.get(key)
        mask_path = mask_index.get(key)
        old_mask_path = old_mask_index.get(key)
        if image_path is None or mask_path is None:
            issues.append(
                {
                    "severity": "error",
                    "code": "missing_image" if image_path is None else "missing_mask",
                    "file_key": key,
                    "details": str(mask_path or image_path),
                }
            )
            continue

        parsed = parse_filename(strip_nifti_suffix(image_path), rules)
        fatal_parse_problems = [
            problem for problem in parsed.problems if problem != "unassigned_timepoint"
        ]
        for problem in parsed.problems:
            issues.append(
                {
                    "severity": (
                        "warning" if problem == "unassigned_timepoint" else "error"
                    ),
                    "code": problem,
                    "file_key": key,
                    "details": parsed.source_stem,
                }
            )

        row = {
            "file_key": key,
            "patient_key": parsed.patient_key,
            "timepoint": parsed.timepoint,
            "trg": parsed.trg,
            "image_relpath": _relative(image_path, root),
            "mask_relpath": _relative(mask_path, root),
            "old_mask_relpath": _relative(old_mask_path, root),
        }
        if parsed.timepoint is None or parsed.patient_key is None:
            unassigned_rows.append(
                {
                    "file_key": key,
                    "trg": parsed.trg,
                    "image_relpath": row["image_relpath"],
                    "mask_relpath": row["mask_relpath"],
                    "old_mask_relpath": row["old_mask_relpath"],
                    "reason": ";".join(parsed.problems) or "unassigned_timepoint",
                }
            )
        elif not fatal_parse_problems:
            paired_rows.append(row)

    patient_groups: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in paired_rows:
        grouped.setdefault(str(row["patient_key"]), []).append(row)

    for patient_key, rows in sorted(grouped.items()):
        pre_rows = [row for row in rows if row["timepoint"] == "pre"]
        post_rows = [row for row in rows if row["timepoint"] == "post"]
        if len(pre_rows) != 1 or len(post_rows) != 1:
            issues.append(
                {
                    "severity": "error",
                    "code": "incomplete_or_duplicate_longitudinal_pair",
                    "file_key": patient_key,
                    "details": f"pre={len(pre_rows)}, post={len(post_rows)}",
                }
            )
            continue

        trg_values = {row["trg"] for row in rows if row["trg"] is not None}
        if len(trg_values) > 1:
            issues.append(
                {
                    "severity": "error",
                    "code": "inconsistent_longitudinal_trg",
                    "file_key": patient_key,
                    "details": str(sorted(trg_values)),
                }
            )
            continue

        trg = next(iter(trg_values), None)
        patient_groups.append(
            {
                "patient_key": patient_key,
                "trg": trg,
                "binary_label": (
                    int(trg in responder_trg) if trg is not None else None
                ),
                "pre": pre_rows[0],
                "post": post_rows[0],
            }
        )

    timepoint_rows: list[dict[str, Any]] = []
    patient_rows: list[dict[str, Any]] = []
    preonly_rows: list[dict[str, Any]] = []
    for index, patient in enumerate(patient_groups, start=1):
        research_id = f"GC-{index:04d}"
        trg = patient["trg"]
        binary_label = patient["binary_label"]
        for timepoint in ("pre", "post"):
            source = patient[timepoint]
            timepoint_rows.append(
                {
                    "research_id": research_id,
                    "timepoint": timepoint,
                    "trg": trg,
                    "binary_label": binary_label,
                    "image_relpath": source["image_relpath"],
                    "mask_relpath": source["mask_relpath"],
                    "old_mask_relpath": source["old_mask_relpath"],
                    "analysis_mask_source": mask_source,
                }
            )

        pre = patient["pre"]
        post = patient["post"]
        patient_rows.append(
            {
                "research_id": research_id,
                "trg": trg,
                "binary_label": binary_label,
                "pre_image_relpath": pre["image_relpath"],
                "pre_mask_relpath": pre["mask_relpath"],
                "post_image_relpath": post["image_relpath"],
                "post_mask_relpath": post["mask_relpath"],
                "analysis_mask_source": mask_source,
            }
        )
        if trg is not None:
            preonly_rows.append(
                {
                    "research_id": research_id,
                    "trg": trg,
                    "binary_label": binary_label,
                    "image_relpath": pre["image_relpath"],
                    "mask_relpath": pre["mask_relpath"],
                    "analysis_mask_source": mask_source,
                }
            )

    patient_frame = pd.DataFrame(patient_rows, columns=PATIENT_COLUMNS)
    counts: dict[str, Any] = {
        "image_files": len(image_paths),
        "mask_files": len(mask_paths),
        "old_mask_files": len(old_mask_paths),
        "image_mask_pairs": sum(
            key in image_index and key in mask_index for key in all_keys
        ),
        "assigned_timepoints": len(paired_rows),
        "unassigned_timepoints": len(unassigned_rows),
        "complete_patients": len(patient_rows),
        "labeled_patients": len(preonly_rows),
        "unlabeled_patients": len(patient_rows) - len(preonly_rows),
        "trg_counts": _distribution(patient_frame.get("trg", pd.Series(dtype=int))),
        "binary_counts": _distribution(
            patient_frame.get("binary_label", pd.Series(dtype=int))
        ),
    }
    mismatches = _expected_mismatches(counts, config.get("expected", {}))
    for mismatch in mismatches:
        issues.append(
            {
                "severity": "error",
                "code": "expected_count_mismatch",
                "file_key": "",
                "details": mismatch,
            }
        )

    outputs = {
        "timepoints": output_dir / "timepoint_manifest.csv",
        "patients": output_dir / "patient_manifest.csv",
        "preonly": output_dir / "preonly_labeled_cohort.csv",
        "unassigned": output_dir / "unassigned_timepoints.csv",
        "issues": output_dir / "manifest_issues.csv",
        "report": output_dir / "manifest_report.json",
    }
    _write_csv(timepoint_rows, TIMEPOINT_COLUMNS, outputs["timepoints"])
    _write_csv(patient_rows, PATIENT_COLUMNS, outputs["patients"])
    _write_csv(preonly_rows, PREONLY_COLUMNS, outputs["preonly"])
    _write_csv(unassigned_rows, UNASSIGNED_COLUMNS, outputs["unassigned"])
    _write_csv(issues, ISSUE_COLUMNS, outputs["issues"])

    issue_frame = pd.DataFrame(issues, columns=ISSUE_COLUMNS)
    error_count = int(issue_frame["severity"].eq("error").sum())
    warning_count = int(issue_frame["severity"].eq("warning").sum())
    report = {
        "schema_version": 1,
        "data_root": str(root),
        "analysis_definition": {
            "timepoint": "pre",
            "mask_source": mask_source,
            "responder_trg": sorted(responder_trg),
            "non_responder_trg": sorted({0, 1, 2, 3} - responder_trg),
        },
        "legacy_manifests_used": False,
        "counts": counts,
        "expected_mismatches": mismatches,
        "errors": error_count,
        "warnings": warning_count,
        "strict": strict,
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    outputs["report"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if strict and error_count:
        raise ManifestValidationError(
            f"Strict manifest validation failed with {error_count} error(s). "
            f"See {outputs['issues']} and {outputs['report']}."
        )
    return outputs
