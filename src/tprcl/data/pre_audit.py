"""Geometry and tumor-burden audit for the frozen Pre-only cohort."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

AUDIT_COLUMNS = [
    "research_id",
    "trg",
    "binary_label",
    "size_x",
    "size_y",
    "size_z",
    "spacing_x_mm",
    "spacing_y_mm",
    "spacing_z_mm",
    "voxel_volume_mm3",
    "mask_voxels",
    "tumor_volume_mm3",
    "tumor_volume_ml",
    "bbox_x_mm",
    "bbox_y_mm",
    "bbox_z_mm",
    "bbox_diagonal_mm",
    "foreground_labels",
    "image_mask_geometry_match",
]
ISSUE_COLUMNS = ["severity", "code", "research_id", "details"]


class PreAuditError(RuntimeError):
    """Raised when strict Pre audit validation finds an invalid case."""


def _triplet(values: tuple[Any, ...], name: str) -> tuple[Any, Any, Any]:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    return values


def geometry_matches(
    image: Any,
    mask: Any,
    *,
    coordinate_tolerance: float = 1e-5,
    direction_tolerance: float = 1e-6,
) -> tuple[bool, list[str]]:
    """Compare the physical geometry of an image and its mask."""

    mismatches: list[str] = []
    if tuple(image.GetSize()) != tuple(mask.GetSize()):
        mismatches.append(f"size: image={image.GetSize()}, mask={mask.GetSize()}")
    for name, image_values, mask_values, tolerance in (
        ("spacing", image.GetSpacing(), mask.GetSpacing(), coordinate_tolerance),
        ("origin", image.GetOrigin(), mask.GetOrigin(), coordinate_tolerance),
        ("direction", image.GetDirection(), mask.GetDirection(), direction_tolerance),
    ):
        if not np.allclose(
            np.asarray(image_values, dtype=float),
            np.asarray(mask_values, dtype=float),
            rtol=0.0,
            atol=tolerance,
        ):
            mismatches.append(
                f"{name}: image={tuple(image_values)}, mask={tuple(mask_values)}"
            )
    return not mismatches, mismatches


def summarize_mask(
    mask_array: np.ndarray,
    spacing_xyz: tuple[float, float, float],
) -> dict[str, Any]:
    """Compute label-independent tumor burden in physical units."""

    spacing_xyz = _triplet(tuple(float(value) for value in spacing_xyz), "spacing")
    if not all(math.isfinite(value) and value > 0 for value in spacing_xyz):
        raise ValueError(f"Spacing must be positive and finite: {spacing_xyz}")
    if mask_array.ndim != 3:
        raise ValueError(f"Mask array must be 3D, got shape={mask_array.shape}")
    foreground = mask_array != 0
    coordinates_zyx = np.argwhere(foreground)
    if coordinates_zyx.size == 0:
        raise ValueError("Mask has no foreground voxels")

    mask_voxels = int(foreground.sum())
    voxel_volume_mm3 = float(np.prod(spacing_xyz))
    bbox_voxels_zyx = coordinates_zyx.max(axis=0) - coordinates_zyx.min(axis=0) + 1
    bbox_mm_xyz = bbox_voxels_zyx[::-1] * np.asarray(spacing_xyz)
    labels = np.unique(mask_array[foreground])
    return {
        "voxel_volume_mm3": voxel_volume_mm3,
        "mask_voxels": mask_voxels,
        "tumor_volume_mm3": mask_voxels * voxel_volume_mm3,
        "tumor_volume_ml": mask_voxels * voxel_volume_mm3 / 1000.0,
        "bbox_x_mm": float(bbox_mm_xyz[0]),
        "bbox_y_mm": float(bbox_mm_xyz[1]),
        "bbox_z_mm": float(bbox_mm_xyz[2]),
        "bbox_diagonal_mm": float(np.linalg.norm(bbox_mm_xyz)),
        "foreground_labels": "|".join(str(value) for value in labels),
    }


def _read_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"research_id", "trg", "binary_label", "image_relpath", "mask_relpath"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Manifest contains no cases")
    if frame["research_id"].duplicated().any():
        duplicates = sorted(
            frame.loc[frame["research_id"].duplicated(False), "research_id"].unique()
        )
        raise ValueError(f"Duplicate research_id values: {duplicates}")
    return frame


def audit_pre_cohort(
    manifest_path: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    *,
    strict: bool = False,
) -> dict[str, Path]:
    """Audit every frozen labeled Pre case and write reproducible tables."""

    try:
        import SimpleITK as sitk
    except ImportError as error:
        raise RuntimeError(
            'SimpleITK is required. Install with: python -m pip install -e ".[dev]"'
        ) from error

    manifest_path = Path(manifest_path).resolve()
    data_root = Path(data_root).resolve()
    output_dir = Path(output_dir).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest(manifest_path)

    rows: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []
    for source in manifest.to_dict(orient="records"):
        research_id = str(source["research_id"])
        image_path = data_root / str(source["image_relpath"])
        mask_path = data_root / str(source["mask_relpath"])
        try:
            if not image_path.is_file():
                raise FileNotFoundError(f"Image not found: {image_path}")
            if not mask_path.is_file():
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            image = sitk.ReadImage(str(image_path))
            mask = sitk.ReadImage(str(mask_path))
            geometry_match, geometry_details = geometry_matches(image, mask)
            if not geometry_match:
                issues.append(
                    {
                        "severity": "error",
                        "code": "image_mask_geometry_mismatch",
                        "research_id": research_id,
                        "details": "; ".join(geometry_details),
                    }
                )
            size_xyz = _triplet(tuple(int(value) for value in image.GetSize()), "size")
            spacing_xyz = _triplet(
                tuple(float(value) for value in image.GetSpacing()), "spacing"
            )
            burden = summarize_mask(sitk.GetArrayFromImage(mask), spacing_xyz)
            rows.append(
                {
                    "research_id": research_id,
                    "trg": int(source["trg"]),
                    "binary_label": int(source["binary_label"]),
                    "size_x": size_xyz[0],
                    "size_y": size_xyz[1],
                    "size_z": size_xyz[2],
                    "spacing_x_mm": spacing_xyz[0],
                    "spacing_y_mm": spacing_xyz[1],
                    "spacing_z_mm": spacing_xyz[2],
                    **burden,
                    "image_mask_geometry_match": geometry_match,
                }
            )
        except Exception as error:
            issues.append(
                {
                    "severity": "error",
                    "code": "case_audit_failed",
                    "research_id": research_id,
                    "details": f"{type(error).__name__}: {error}",
                }
            )

    audit_frame = pd.DataFrame(rows, columns=AUDIT_COLUMNS)
    issue_frame = pd.DataFrame(issues, columns=ISSUE_COLUMNS)
    outputs = {
        "audit": output_dir / "pre_geometry_roi_audit.csv",
        "issues": output_dir / "pre_geometry_roi_issues.csv",
        "report": output_dir / "pre_geometry_roi_report.json",
    }
    audit_frame.to_csv(outputs["audit"], index=False, encoding="utf-8-sig")
    issue_frame.to_csv(outputs["issues"], index=False, encoding="utf-8-sig")

    error_count = int(issue_frame["severity"].eq("error").sum())
    label_counts = {
        str(key): int(value)
        for key, value in (
            audit_frame["binary_label"].value_counts().sort_index().items()
        )
    }
    report = {
        "schema_version": 1,
        "analysis_timepoint": "pre",
        "manifest": str(manifest_path),
        "data_root": str(data_root),
        "manifest_cases": int(len(manifest)),
        "audited_cases": int(len(audit_frame)),
        "failed_cases": int(len(manifest) - len(audit_frame)),
        "geometry_mismatch_cases": int(
            (~audit_frame["image_mask_geometry_match"]).sum()
        ),
        "binary_label_counts": label_counts,
        "errors": error_count,
        "strict": strict,
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    outputs["report"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if strict and error_count:
        raise PreAuditError(
            f"Strict Pre audit failed with {error_count} error(s). "
            f"See {outputs['issues']} and {outputs['report']}."
        )
    return outputs
