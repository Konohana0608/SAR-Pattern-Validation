from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk

from sar_pattern_validation.errors import CsvFormatError
from sar_pattern_validation.image_loader import (
    SARImageLoader,
    is_registration_mask_sufficient,
)
from sar_pattern_validation.utils import image_extent_mm

from .helpers import gaussian_2d, make_rect_grid, write_sar_csv


def _build_loader(
    measured_csv: str, reference_csv: str, *, resample_resolution=None
) -> SARImageLoader:
    """Helper: construct SARImageLoader with pipeline defaults."""
    return SARImageLoader(
        measured_path=measured_csv,
        reference_path=reference_csv,
        resample_resolution=resample_resolution,
        show_plot=False,
        warn=True,
    )


def test_builds_linear_and_db_images(tmp_csv_pair):
    """
    Sanity check: the loader builds both linear and dB images for measured + reference.
    Linear images are used for gamma/masks; dB images are used for MI registration.
    """
    measured_csv, reference_csv = tmp_csv_pair
    loader = _build_loader(measured_csv, reference_csv, resample_resolution=None)

    # Linear images exist
    assert isinstance(loader.measured_image_linear, sitk.Image)
    assert isinstance(loader.reference_image_linear, sitk.Image)

    # dB images exist and are returned in (reference, measured) order
    reference_db, measured_db = loader.get_images()
    assert isinstance(reference_db, sitk.Image)
    assert isinstance(measured_db, sitk.Image)

    # Images are 2D
    meas_lin = sitk.GetArrayFromImage(loader.measured_image_linear)
    ref_lin = sitk.GetArrayFromImage(loader.reference_image_linear)
    assert meas_lin.ndim == 2
    assert ref_lin.ndim == 2

    # Spacing must be finite and positive
    for s in (
        *loader.measured_image_linear.GetSpacing(),
        *loader.reference_image_linear.GetSpacing(),
    ):
        assert np.isfinite(s)
        assert s > 0


def test_resampling_produces_uniform_spacing_and_nontrivial_content(tmp_csv_pair):
    """
    With resample_resolution set, both images are interpolated onto a uniform grid.
    We validate externally observable behavior (spacing + finite arrays + non-zero region),
    not private attributes.
    """
    measured_csv, reference_csv = tmp_csv_pair
    loader = _build_loader(measured_csv, reference_csv, resample_resolution=0.003)

    mx, my = loader.measured_image_linear.GetSpacing()
    rx, ry = loader.reference_image_linear.GetSpacing()

    # Uniform spacing should match the request (within small tolerance)
    assert np.isclose(mx, 0.003, rtol=5e-3)
    assert np.isclose(my, 0.003, rtol=5e-3)
    assert np.isclose(rx, 0.003, rtol=5e-3)
    assert np.isclose(ry, 0.003, rtol=5e-3)

    meas_lin = sitk.GetArrayFromImage(loader.measured_image_linear)
    ref_lin = sitk.GetArrayFromImage(loader.reference_image_linear)

    assert np.isfinite(meas_lin).all()
    assert np.isfinite(ref_lin).all()

    # Some valid signal should exist
    assert np.nanmax(meas_lin) > 0
    assert np.nanmax(ref_lin) > 0


def test_peak_normalization_is_unit_peak(tmp_csv_pair):
    """
    The loader normalizes each linear image by its (masked) peak.
    Therefore, max(linear) should be ~1.0 for each image.
    """
    measured_csv, reference_csv = tmp_csv_pair
    loader = _build_loader(measured_csv, reference_csv, resample_resolution=None)

    meas_lin = sitk.GetArrayFromImage(loader.measured_image_linear)
    ref_lin = sitk.GetArrayFromImage(loader.reference_image_linear)

    assert np.isclose(np.nanmax(meas_lin), 1.0, atol=1e-6)
    assert np.isclose(np.nanmax(ref_lin), 1.0, atol=1e-6)


def test_db_images_are_finite(tmp_csv_pair):
    """
    The dB conversion clamps before log10, so there should be no ±inf.
    """
    measured_csv, reference_csv = tmp_csv_pair
    loader = _build_loader(measured_csv, reference_csv, resample_resolution=None)

    reference_db, measured_db = loader.get_images()
    ref_db = sitk.GetArrayFromImage(reference_db)
    meas_db = sitk.GetArrayFromImage(measured_db)

    assert np.isfinite(ref_db).all()
    assert np.isfinite(meas_db).all()


def test_metric_masks_are_binary_and_match_geometry(tmp_csv_pair):
    """
    make_metric_masks() returns absolute masks (uint8) aligned to the corresponding
    linear images (same size/spacing/origin).
    """
    measured_csv, reference_csv = tmp_csv_pair
    loader = _build_loader(measured_csv, reference_csv, resample_resolution=None)

    measured_mask_u8, reference_mask_u8 = loader.make_metric_masks()
    assert isinstance(measured_mask_u8, sitk.Image)
    assert isinstance(reference_mask_u8, sitk.Image)

    meas_mask = sitk.GetArrayFromImage(measured_mask_u8)
    ref_mask = sitk.GetArrayFromImage(reference_mask_u8)

    meas_lin = sitk.GetArrayFromImage(loader.measured_image_linear)
    ref_lin = sitk.GetArrayFromImage(loader.reference_image_linear)

    assert meas_mask.shape == meas_lin.shape
    assert ref_mask.shape == ref_lin.shape

    assert set(np.unique(meas_mask)).issubset({0, 1})
    assert set(np.unique(ref_mask)).issubset({0, 1})

    # For synthetic data, we expect at least some included pixels.
    assert np.any(meas_mask == 1)
    assert np.any(ref_mask == 1)

    # Geometry consistency: masks should copy information from the linear images
    assert measured_mask_u8.GetSpacing() == loader.measured_image_linear.GetSpacing()
    assert measured_mask_u8.GetOrigin() == loader.measured_image_linear.GetOrigin()
    assert reference_mask_u8.GetSpacing() == loader.reference_image_linear.GetSpacing()
    assert reference_mask_u8.GetOrigin() == loader.reference_image_linear.GetOrigin()


def test_support_masks_are_binary_and_cover_at_least_metric_masks(tmp_csv_pair):
    measured_csv, reference_csv = tmp_csv_pair
    loader = _build_loader(measured_csv, reference_csv, resample_resolution=None)

    measured_mask_u8, reference_mask_u8 = loader.make_metric_masks()
    measured_support_u8, reference_support_u8 = loader.make_support_masks()

    meas_mask = sitk.GetArrayFromImage(measured_mask_u8).astype(bool)
    ref_mask = sitk.GetArrayFromImage(reference_mask_u8).astype(bool)
    meas_support = sitk.GetArrayFromImage(measured_support_u8).astype(bool)
    ref_support = sitk.GetArrayFromImage(reference_support_u8).astype(bool)

    assert np.all(meas_mask <= meas_support)
    assert np.all(ref_mask <= ref_support)
    assert measured_support_u8.GetSpacing() == loader.measured_image_linear.GetSpacing()
    assert measured_support_u8.GetOrigin() == loader.measured_image_linear.GetOrigin()
    assert (
        reference_support_u8.GetSpacing() == loader.reference_image_linear.GetSpacing()
    )
    assert reference_support_u8.GetOrigin() == loader.reference_image_linear.GetOrigin()


def test_registration_mask_sufficiency_rejects_sparse_and_thin_masks() -> None:
    sparse = np.zeros((32, 32), dtype=np.uint8)
    sparse[:7, :9] = 1  # 63 active pixels -> below the minimum count threshold
    assert not is_registration_mask_sufficient(sparse)

    thin = np.zeros((32, 32), dtype=np.uint8)
    thin[10:26, 12:16] = 1  # Adequate count, but only 4 px wide
    assert not is_registration_mask_sufficient(thin)

    dense = np.zeros((32, 32), dtype=np.uint8)
    dense[8:18, 10:20] = 1
    assert is_registration_mask_sufficient(dense)


def test_loader_uses_supported_signal_when_metric_mask_is_insufficient() -> None:
    project_root = Path(__file__).resolve().parents[1]
    measured_csv = (
        project_root / "data/measurements/D1950_Flat HSL_10 mm_4 dBm_10g_14.csv"
    )
    reference_csv = project_root / "data/database/dipole_1950MHz_Flat_10mm_10g.csv"

    loader = SARImageLoader(
        measured_path=str(measured_csv),
        reference_path=str(reference_csv),
        power_level_dbm=4.0,
        noise_floor_wkg=0.05,
        show_plot=False,
        warn=True,
    )

    assert loader.measured_metric_mask_sufficient is False
    assert loader.measured_peak > 1e-6

    _, measured_db = loader.get_images()
    measured_db_arr = sitk.GetArrayFromImage(measured_db)

    assert np.isfinite(measured_db_arr).all()
    assert np.ptp(measured_db_arr) > 1.0
    assert float(np.nanmax(measured_db_arr)) <= 1e-6


def test_loader_converts_mm_headers_to_meter_spacing(tmp_path):
    """
    Documented millimeter-style headers should be converted to metric spacing.
    """
    measured_csv = tmp_path / "measured.csv"
    reference_csv = tmp_path / "reference.csv"

    df = pd.DataFrame(
        {
            "x_mm": [0.0, 1.0, 0.0, 1.0],
            "y_mm": [0.0, 0.0, 1.0, 1.0],
            "sar": [1.0, 2.0, 3.0, 4.0],
        }
    )
    df.to_csv(measured_csv, index=False)
    df.to_csv(reference_csv, index=False)

    loader = _build_loader(str(measured_csv), str(reference_csv))

    mx, my = loader.measured_image_linear.GetSpacing()
    ox, oy = loader.measured_image_linear.GetOrigin()

    assert np.isclose(mx, 0.001, atol=1e-12)
    assert np.isclose(my, 0.001, atol=1e-12)
    assert np.isclose(ox, 0.0, atol=1e-12)
    assert np.isclose(oy, 0.0, atol=1e-12)


def test_loader_treats_bare_xy_headers_as_meters(tmp_path):
    """
    Bare coordinate headers default to meters unless millimeters are explicit.
    """
    measured_csv = tmp_path / "measured.csv"
    reference_csv = tmp_path / "reference.csv"

    df = pd.DataFrame(
        {
            "x": [100.0, 101.0, 100.0, 101.0],
            "y": [200.0, 200.0, 201.0, 201.0],
            "sar": [1.0, 2.0, 3.0, 4.0],
        }
    )
    df.to_csv(measured_csv, index=False)
    df.to_csv(reference_csv, index=False)

    loader = _build_loader(str(measured_csv), str(reference_csv))

    img = loader.measured_image_linear
    sx, sy = img.GetSpacing()
    ox, oy = img.GetOrigin()

    assert np.isclose(sx, 1.0, atol=1e-12)
    assert np.isclose(sy, 1.0, atol=1e-12)
    assert np.isclose(ox, 100.0, atol=1e-12)
    assert np.isclose(oy, 200.0, atol=1e-12)


def test_loader_rejects_non_coordinate_headers_that_only_start_with_x_or_y(tmp_path):
    measured_csv = tmp_path / "measured.csv"
    reference_csv = tmp_path / "reference.csv"

    df = pd.DataFrame(
        {
            "xpos": [0.0, 1.0, 0.0, 1.0],
            "ypos": [0.0, 0.0, 1.0, 1.0],
            "sar": [1.0, 2.0, 3.0, 4.0],
        }
    )
    df.to_csv(measured_csv, index=False)
    df.to_csv(reference_csv, index=False)

    with pytest.raises(CsvFormatError, match="recognizable x/y coordinate columns"):
        _build_loader(str(measured_csv), str(reference_csv))


def test_loader_rejects_ambiguous_coordinate_columns(tmp_path):
    measured_csv = tmp_path / "measured.csv"
    reference_csv = tmp_path / "reference.csv"

    df = pd.DataFrame(
        {
            "x": [0.0, 1.0, 0.0, 1.0],
            "x_mm": [0.0, 1.0, 0.0, 1.0],
            "y": [0.0, 0.0, 1.0, 1.0],
            "sar": [1.0, 2.0, 3.0, 4.0],
        }
    )
    df.to_csv(measured_csv, index=False)
    df.to_csv(reference_csv, index=False)

    with pytest.raises(CsvFormatError, match="Ambiguous x-coordinate column"):
        _build_loader(str(measured_csv), str(reference_csv))


def test_loader_preserves_world_coordinates_from_csv(tmp_path):
    """
    The loader should preserve the physical coordinate of the first grid sample
    instead of recentering the image around zero.
    """
    x_m = np.array([0.100, 0.101], dtype=float)
    y_m = np.array([0.200, 0.201], dtype=float)
    _, _, Z = gaussian_2d(x_m, y_m, x0=0.1005, y0=0.2005, sx=0.005, sy=0.005)

    measured_csv = tmp_path / "measured.csv"
    reference_csv = tmp_path / "reference.csv"
    write_sar_csv(measured_csv, x_m, y_m, Z)
    pd.read_csv(measured_csv).to_csv(reference_csv, index=False)

    loader = _build_loader(str(measured_csv), str(reference_csv))

    img = loader.measured_image_linear
    sx, sy = img.GetSpacing()
    ox, oy = img.GetOrigin()

    assert np.isclose(sx, 0.001, atol=1e-12)
    assert np.isclose(sy, 0.001, atol=1e-12)
    assert np.isclose(ox, x_m.min(), atol=1e-12)
    assert np.isclose(oy, y_m.min(), atol=1e-12)

    extent = image_extent_mm(img)
    assert np.allclose(extent, [99.5, 101.5, 199.5, 201.5], atol=1e-9)


def test_resample_to_1mm_spacing_preserves_axis_origin_and_unit_peak(tmp_path):
    """
    When resampling to 1 mm (0.001 m), spacing should match the request,
    the origin should match the minimum physical coordinate, and unit-peak
    normalization should be preserved.
    """
    x_m, y_m = make_rect_grid(step=0.001)
    _, _, Z = gaussian_2d(x_m, y_m, x0=0.01, y0=-0.02, sx=0.02, sy=0.03, peak=1.0)

    measured_csv = tmp_path / "measured.csv"
    reference_csv = tmp_path / "reference.csv"
    write_sar_csv(measured_csv, x_m, y_m, Z)
    pd.read_csv(measured_csv).to_csv(reference_csv, index=False)

    loader = _build_loader(
        str(measured_csv), str(reference_csv), resample_resolution=0.001
    )

    mx, my = loader.measured_image_linear.GetSpacing()
    rx, ry = loader.reference_image_linear.GetSpacing()

    assert np.isclose(mx, 0.001, rtol=5e-3)
    assert np.isclose(my, 0.001, rtol=5e-3)
    assert np.isclose(rx, 0.001, rtol=5e-3)
    assert np.isclose(ry, 0.001, rtol=5e-3)

    for img in (loader.measured_image_linear, loader.reference_image_linear):
        ox, oy = img.GetOrigin()

        assert np.isclose(ox, x_m.min(), rtol=0, atol=1e-12)
        assert np.isclose(oy, y_m.min(), rtol=0, atol=1e-12)

    meas_lin = sitk.GetArrayFromImage(loader.measured_image_linear)
    ref_lin = sitk.GetArrayFromImage(loader.reference_image_linear)

    assert np.isclose(np.nanmax(meas_lin), 1.0, atol=1e-6)
    assert np.isclose(np.nanmax(ref_lin), 1.0, atol=1e-6)
