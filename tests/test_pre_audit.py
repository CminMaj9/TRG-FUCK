from __future__ import annotations

import numpy as np
import pytest

from tprcl.data.pre_audit import geometry_matches, summarize_mask


class _Geometry:
    def __init__(
        self,
        *,
        size: tuple[int, int, int] = (10, 20, 30),
        spacing: tuple[float, float, float] = (1.0, 1.0, 2.0),
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        direction: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    ) -> None:
        self._size = size
        self._spacing = spacing
        self._origin = origin
        self._direction = direction

    def GetSize(self) -> tuple[int, int, int]:
        return self._size

    def GetSpacing(self) -> tuple[float, float, float]:
        return self._spacing

    def GetOrigin(self) -> tuple[float, float, float]:
        return self._origin

    def GetDirection(self) -> tuple[float, ...]:
        return self._direction


def test_geometry_match_reports_physical_mismatch() -> None:
    matched, details = geometry_matches(_Geometry(), _Geometry())
    assert matched
    assert details == []
    matched, details = geometry_matches(_Geometry(), _Geometry(spacing=(1.0, 1.0, 2.5)))
    assert not matched
    assert any(detail.startswith("spacing:") for detail in details)


def test_summarize_mask_uses_physical_units_and_zyx_array_order() -> None:
    mask = np.zeros((5, 6, 7), dtype=np.uint8)
    mask[1:4, 2:6, 3:5] = 1
    result = summarize_mask(mask, spacing_xyz=(0.5, 1.0, 2.0))
    assert result["mask_voxels"] == 24
    assert result["voxel_volume_mm3"] == pytest.approx(1.0)
    assert result["tumor_volume_ml"] == pytest.approx(0.024)
    assert result["bbox_x_mm"] == pytest.approx(1.0)
    assert result["bbox_y_mm"] == pytest.approx(4.0)
    assert result["bbox_z_mm"] == pytest.approx(6.0)
    assert result["foreground_labels"] == "1"


@pytest.mark.parametrize(
    ("mask", "spacing", "message"),
    [
        (np.zeros((2, 2, 2)), (1.0, 1.0, 1.0), "no foreground"),
        (np.ones((2, 2)), (1.0, 1.0, 1.0), "must be 3D"),
        (np.ones((2, 2, 2)), (1.0, 0.0, 1.0), "positive and finite"),
    ],
)
def test_summarize_mask_rejects_invalid_inputs(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        summarize_mask(mask, spacing)
