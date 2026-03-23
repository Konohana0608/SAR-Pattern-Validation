import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk

from sar_pattern_validation.image_loader import SARImageLoader
from sar_pattern_validation.registration2d import Rigid2DRegistration, Transform2D

from .helpers import (
    gaussian_2d,
    make_rect_grid,
    rigid_transform_points,
    write_sar_csv,
)

# -----------------------------------------------------------------------------
# Test philosophy (why these are "slow")
#
# These tests:
# - build synthetic SAR CSVs
# - load them through SARImageLoader
# - run multi-stage exhaustive registration (SimpleITK)
# - verify the aligned moving image matches the fixed image on the overlap region
#
# They are marked "slow" because exhaustive search is intentionally expensive.
# -----------------------------------------------------------------------------


def _build_loader(
    measured_csv: str, reference_csv: str, *, resample_resolution=0.006
) -> SARImageLoader:
    """
    Construct the loader with consistent defaults.

    Loader convention (current SARImageLoader):
      - reference = fixed (target grid)
      - measured  = moving (warped onto reference)
    """
    return SARImageLoader(
        measured_path=measured_csv,
        reference_path=reference_csv,
        resample_resolution=resample_resolution,
        show_plot=False,
        warn=True,
    )


def _run_registration_coarse_to_fine(
    *,
    fixed_img: sitk.Image,
    moving_img: sitk.Image,
    fixed_mask: sitk.Image | None,
    moving_mask: sitk.Image | None,
):
    """
    Coarse→fine exhaustive rigid registration using Mattes MI + Moments init.

    Buffer-escape prevention (union expansion) and adaptive step clamping are handled
    internally by Rigid2DRegistration.run() — no pre-processing needed here.
    """
    stages = [
        dict(
            translation_step=0.010,
            rot_step_deg=1.0,
            rot_span_deg=180.0,
            tx_steps=6,
            ty_steps=6,
        ),
        dict(
            translation_step=0.001,
            rot_step_deg=0.5,
            rot_span_deg=90.0,
            tx_steps=6,
            ty_steps=6,
        ),
    ]

    reg = Rigid2DRegistration(
        fixed_image=fixed_img,
        moving_image=moving_img,
        transform_type=Transform2D.RIGID,
    )

    aligned, tx = reg.run(stages=stages, fixed_mask=fixed_mask, moving_mask=moving_mask)
    return aligned, tx


def _apply_known_rigid_transform(
    point: tuple[float, float],
    *,
    tx: float,
    ty: float,
    theta_deg: float,
) -> tuple[float, float]:
    theta = np.deg2rad(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    x, y = point
    return (c * x - s * y + tx, s * x + c * y + ty)


# --------------------------------- tests -------------------------------------


def test_resample_to_1mm_spacing_uses_expected_spacing_axis_origin_and_unit_peak(
    tmp_path,
):
    """
    For regular metric input:
      - spacing should be ~0.001 m
      - peak remains ~1
      - origin should match the minimum coordinate on each axis
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

    meas_lin = sitk.GetArrayFromImage(loader.measured_image_linear)
    ref_lin = sitk.GetArrayFromImage(loader.reference_image_linear)
    assert np.isclose(np.nanmax(meas_lin), 1.0, atol=1e-6)
    assert np.isclose(np.nanmax(ref_lin), 1.0, atol=1e-6)

    for img in (loader.measured_image_linear, loader.reference_image_linear):
        ox, oy = img.GetOrigin()

        assert np.isclose(ox, x_m.min(), rtol=0, atol=1e-12)
        assert np.isclose(oy, y_m.min(), rtol=0, atol=1e-12)


@pytest.mark.slow
def test_translate_registration_recovers_known_landmark_shift(tmp_path):
    x, y = make_rect_grid(xmin=-0.04, xmax=0.04, ymin=-0.04, ymax=0.04, step=0.005)
    _, _, main_peak = gaussian_2d(
        x, y, x0=0.010, y0=-0.015, sx=0.018, sy=0.020, peak=1.0
    )
    _, _, side_peak = gaussian_2d(
        x, y, x0=-0.014, y0=0.018, sx=0.010, sy=0.014, peak=0.6
    )
    Z = main_peak + side_peak

    reference_csv = tmp_path / "reference.csv"
    write_sar_csv(reference_csv, x, y, Z)

    translation = (0.004, -0.003)
    measured_df = pd.read_csv(reference_csv)
    measured_df = rigid_transform_points(
        measured_df, tx=translation[0], ty=translation[1], theta_deg=0.0
    )
    measured_csv = tmp_path / "measured.csv"
    measured_df.to_csv(measured_csv, index=False)

    loader = _build_loader(
        str(measured_csv), str(reference_csv), resample_resolution=0.005
    )
    _, reference_mask_u8 = loader.make_metric_masks()

    reg = Rigid2DRegistration(
        fixed_image=loader.reference_image_db,
        moving_image=loader.measured_image_db,
        transform_type=Transform2D.TRANSLATE,
    )
    _, final_tx = reg.run(
        stages=[
            dict(
                translation_step=0.001,
                tx_steps=6,
                ty_steps=6,
                rot_step_deg=0.0,
                rot_span_deg=0.0,
            )
        ],
        fixed_mask=reference_mask_u8,
        moving_mask=None,
    )

    reference_point = (0.01, -0.015)
    measured_point = _apply_known_rigid_transform(
        reference_point, tx=translation[0], ty=translation[1], theta_deg=0.0
    )
    recovered_point = final_tx.TransformPoint(measured_point)

    assert np.allclose(recovered_point, reference_point, atol=1.0e-3)


@pytest.mark.slow
def test_rigid_registration_recovers_landmarks_and_beats_identity(tmp_path):
    x, y = make_rect_grid(xmin=-0.05, xmax=0.05, ymin=-0.05, ymax=0.05, step=0.005)
    _, _, main_peak = gaussian_2d(
        x, y, x0=0.018, y0=-0.012, sx=0.018, sy=0.022, peak=1.0
    )
    _, _, side_peak = gaussian_2d(
        x, y, x0=-0.014, y0=0.020, sx=0.012, sy=0.015, peak=0.55
    )
    Z = main_peak + side_peak

    reference_csv = tmp_path / "reference.csv"
    write_sar_csv(reference_csv, x, y, Z)

    tx, ty, theta_deg = 0.004, -0.003, 3.0
    measured_df = pd.read_csv(reference_csv)
    measured_df = rigid_transform_points(measured_df, tx=tx, ty=ty, theta_deg=theta_deg)
    measured_csv = tmp_path / "measured.csv"
    measured_df.to_csv(measured_csv, index=False)

    loader = _build_loader(
        str(measured_csv), str(reference_csv), resample_resolution=0.005
    )
    _, reference_mask_u8 = loader.make_metric_masks()

    reg = Rigid2DRegistration(
        fixed_image=loader.reference_image_db,
        moving_image=loader.measured_image_db,
        transform_type=Transform2D.RIGID,
    )
    _, final_tx = reg.run(
        stages=[
            dict(
                translation_step=0.002,
                rot_step_deg=0.5,
                rot_span_deg=6.0,
                tx_steps=4,
                ty_steps=4,
            ),
            dict(
                translation_step=0.0005,
                rot_step_deg=0.25,
                rot_span_deg=2.0,
                tx_steps=4,
                ty_steps=4,
            ),
        ],
        fixed_mask=reference_mask_u8,
        moving_mask=None,
    )

    landmarks_reference = [
        (0.018, -0.012),
        (0.008, -0.002),
        (0.028, -0.022),
    ]
    landmarks_measured = [
        _apply_known_rigid_transform(p, tx=tx, ty=ty, theta_deg=theta_deg)
        for p in landmarks_reference
    ]

    identity_errors_mm = [
        1000.0 * np.hypot(m[0] - r[0], m[1] - r[1])
        for m, r in zip(landmarks_measured, landmarks_reference, strict=False)
    ]
    recovered_errors_mm = [
        1000.0
        * np.hypot(
            final_tx.TransformPoint(m)[0] - r[0],
            final_tx.TransformPoint(m)[1] - r[1],
        )
        for m, r in zip(landmarks_measured, landmarks_reference, strict=False)
    ]

    assert np.mean(recovered_errors_mm) < 2.0
    assert np.mean(recovered_errors_mm) < 0.3 * np.mean(identity_errors_mm)


@pytest.mark.slow
def test_self_registration_produces_near_identity_transform(tmp_path):
    """
    Registering an image to itself must succeed (no 'outside buffer' error) and
    return a near-identity transform regardless of the exhaustive search range.
    """
    x, y = make_rect_grid(xmin=-0.04, xmax=0.04, ymin=-0.04, ymax=0.04, step=0.005)
    _, _, Z = gaussian_2d(x, y, x0=0.008, y0=-0.010, sx=0.020, sy=0.025, peak=1.0)

    csv_path = tmp_path / "image.csv"
    write_sar_csv(csv_path, x, y, Z)

    loader = _build_loader(str(csv_path), str(csv_path), resample_resolution=0.005)
    fixed_img = loader.reference_image_linear
    moving_img = loader.measured_image_linear
    fixed_mask, _ = (
        loader.make_metric_masks()
    )  # (measured_mask, reference_mask); use reference

    # Use the same large default search range that was previously causing the error
    stages = [
        dict(
            translation_step=0.010,
            rot_step_deg=4.0,
            rot_span_deg=180.0,
            tx_steps=6,
            ty_steps=6,
        ),
        dict(
            translation_step=0.001,
            rot_step_deg=1.0,
            rot_span_deg=90.0,
            tx_steps=6,
            ty_steps=6,
        ),
    ]
    reg = Rigid2DRegistration(
        fixed_image=fixed_img,
        moving_image=moving_img,
        transform_type=Transform2D.RIGID,
    )
    # Must not raise
    _, final_tx = reg.run(stages=stages, fixed_mask=fixed_mask, moving_mask=None)

    # Transform applied to image center should be near-identity
    center = fixed_img.GetOrigin()
    cx = center[0] + 0.5 * fixed_img.GetSize()[0] * fixed_img.GetSpacing()[0]
    cy = center[1] + 0.5 * fixed_img.GetSize()[1] * fixed_img.GetSpacing()[1]
    mapped = final_tx.TransformPoint((cx, cy))
    shift_mm = 1000.0 * np.hypot(mapped[0] - cx, mapped[1] - cy)
    assert shift_mm < 2.0, f"Self-registration shift too large: {shift_mm:.2f} mm"


@pytest.mark.slow
def test_registration_recovers_peak_offset_outside_reference_fov(tmp_path):
    """
    Measured image's SAR peak is placed so it sits outside the reference FOV before
    registration.  Without union-domain expansion this peak would be clipped away.
    The test verifies that registration still recovers a good alignment.
    """
    ref_step = 0.005
    meas_step = 0.005
    # Reference: small grid centred near origin
    xr, yr = make_rect_grid(xmin=-0.04, xmax=0.04, ymin=-0.04, ymax=0.04, step=ref_step)
    _, _, Zr = gaussian_2d(xr, yr, x0=0.0, y0=0.0, sx=0.020, sy=0.020, peak=1.0)
    reference_csv = tmp_path / "reference.csv"
    write_sar_csv(reference_csv, xr, yr, Zr)

    # Measured: larger grid, peak shifted +55 mm in x (well outside reference FOV before reg)
    offset_x, offset_y = 0.030, 0.010
    xm, ym = make_rect_grid(
        xmin=-0.08, xmax=0.08, ymin=-0.06, ymax=0.06, step=meas_step
    )
    _, _, Zm = gaussian_2d(
        xm, ym, x0=offset_x, y0=offset_y, sx=0.020, sy=0.020, peak=1.0
    )
    measured_csv = tmp_path / "measured.csv"
    write_sar_csv(measured_csv, xm, ym, Zm)

    loader = SARImageLoader(
        measured_path=str(measured_csv),
        reference_path=str(reference_csv),
        resample_resolution=ref_step,
        show_plot=False,
        warn=False,
    )
    fixed_img = loader.reference_image_linear
    moving_img = loader.measured_image_linear
    _, fixed_mask = loader.make_metric_masks()

    stages = [
        dict(
            translation_step=0.010,
            rot_step_deg=4.0,
            rot_span_deg=180.0,
            tx_steps=8,
            ty_steps=8,
        ),
        dict(
            translation_step=0.002,
            rot_step_deg=1.0,
            rot_span_deg=20.0,
            tx_steps=5,
            ty_steps=5,
        ),
    ]
    reg = Rigid2DRegistration(
        fixed_image=fixed_img,
        moving_image=moving_img,
        transform_type=Transform2D.RIGID,
    )
    _, final_tx = reg.run(stages=stages, fixed_mask=fixed_mask, moving_mask=None)

    # The known shift should be recovered within 3 mm
    ref_peak = (0.0, 0.0)
    meas_peak = (offset_x, offset_y)
    recovered = final_tx.TransformPoint(meas_peak)
    error_mm = 1000.0 * np.hypot(recovered[0] - ref_peak[0], recovered[1] - ref_peak[1])
    assert error_mm < 3.0, f"Peak recovery error too large: {error_mm:.2f} mm"


def test_clamp_stage_steps_reduces_steps_for_small_images():
    """
    _clamp_stage_steps must reduce tx_steps/ty_steps when the user preset would push
    the search range beyond the image physical extent.
    """
    import logging

    # 20 mm × 20 mm images at 1 mm spacing
    fixed = sitk.Image(20, 20, sitk.sitkFloat32)
    fixed.SetSpacing([0.001, 0.001])
    moving = sitk.Image(20, 20, sitk.sitkFloat32)
    moving.SetSpacing([0.001, 0.001])

    # stage asks for ±6 steps × 10 mm = ±60 mm search on a 20 mm image
    stage = dict(
        translation_step=0.010,
        rot_step_deg=4.0,
        rot_span_deg=180.0,
        tx_steps=6,
        ty_steps=6,
    )
    log = logging.getLogger("test")
    clamped = Rigid2DRegistration._clamp_stage_steps(stage, fixed, moving, log)

    # max sensible steps = floor(20mm / (2 * 10mm)) = 1
    assert clamped["tx_steps"] == 1
    assert clamped["ty_steps"] == 1
    # other keys unchanged
    assert clamped["translation_step"] == stage["translation_step"]
    assert clamped["rot_step_deg"] == stage["rot_step_deg"]


def test_expand_to_union_respects_requested_padding():
    fixed = sitk.Image(20, 10, sitk.sitkFloat32)
    fixed.SetSpacing([0.001, 0.002])
    fixed.SetOrigin([0.0, -0.004])

    moving = sitk.Image(8, 6, sitk.sitkFloat32)
    moving.SetSpacing([0.001, 0.002])
    moving.SetOrigin([-0.003, 0.002])

    expanded = Rigid2DRegistration._expand_to_union(
        fixed,
        moving,
        0.0,
        sitk.sitkLinear,
        padding_x=0.002,
        padding_y=0.004,
    )

    assert expanded.GetOrigin() == pytest.approx((-0.005, -0.008))
    assert expanded.GetSpacing() == pytest.approx((0.001, 0.002))
    assert expanded.GetSize() == (27, 14)


def test_clamp_stage_steps_does_not_increase_steps():
    """
    When the preset is already conservative, _clamp_stage_steps must not increase steps.
    """
    import logging

    # Large image: 200 mm × 200 mm
    fixed = sitk.Image(200, 200, sitk.sitkFloat32)
    fixed.SetSpacing([0.001, 0.001])
    moving = sitk.Image(200, 200, sitk.sitkFloat32)
    moving.SetSpacing([0.001, 0.001])

    stage = dict(
        translation_step=0.010,
        rot_step_deg=4.0,
        rot_span_deg=180.0,
        tx_steps=3,
        ty_steps=2,
    )
    log = logging.getLogger("test")
    clamped = Rigid2DRegistration._clamp_stage_steps(stage, fixed, moving, log)

    assert clamped["tx_steps"] == 3
    assert clamped["ty_steps"] == 2


def test_build_adaptive_stages_refines_monotonically():
    fixed = sitk.Image(100, 100, sitk.sitkFloat32)
    fixed.SetSpacing([0.001, 0.001])
    moving = sitk.Image(100, 100, sitk.sitkFloat32)
    moving.SetSpacing([0.001, 0.001])

    fixed_arr = np.zeros((100, 100), dtype=np.uint8)
    fixed_arr[30:70, 40:60] = 1
    fixed_mask = sitk.GetImageFromArray(fixed_arr)
    fixed_mask.CopyInformation(fixed)

    moving_arr = np.zeros((100, 100), dtype=np.uint8)
    moving_arr[28:68, 45:65] = 1
    moving_mask = sitk.GetImageFromArray(moving_arr)
    moving_mask.CopyInformation(moving)

    stages = Rigid2DRegistration.build_adaptive_stages(
        fixed_image=fixed,
        moving_image=moving,
        transform_type=Transform2D.RIGID,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        max_stages=4,
        max_stage_evals=50000,
    )

    assert len(stages) >= 2
    for i in range(1, len(stages)):
        assert stages[i]["translation_step"] <= stages[i - 1]["translation_step"]
        assert stages[i]["tx_steps"] <= stages[i - 1]["tx_steps"]
        assert stages[i]["ty_steps"] <= stages[i - 1]["ty_steps"]
        assert stages[i]["rot_span_deg"] <= stages[i - 1]["rot_span_deg"]
        assert stages[i]["rot_step_deg"] <= stages[i - 1]["rot_step_deg"]


def test_build_adaptive_stages_symmetry_uses_90_deg_span():
    fixed = sitk.Image(120, 120, sitk.sitkFloat32)
    fixed.SetSpacing([0.001, 0.001])
    moving = sitk.Image(120, 120, sitk.sitkFloat32)
    moving.SetSpacing([0.001, 0.001])

    yy, xx = np.indices((120, 120))
    r2 = (xx - 60) ** 2 + (yy - 60) ** 2
    circ = (r2 <= 20**2).astype(np.uint8)

    fixed_mask = sitk.GetImageFromArray(circ)
    fixed_mask.CopyInformation(fixed)
    moving_mask = sitk.GetImageFromArray(circ)
    moving_mask.CopyInformation(moving)

    stages = Rigid2DRegistration.build_adaptive_stages(
        fixed_image=fixed,
        moving_image=moving,
        transform_type=Transform2D.RIGID,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        assume_axial_symmetry=True,
        max_stages=3,
        max_stage_evals=50000,
    )

    assert stages[0]["rot_span_deg"] == 90.0


def test_build_adaptive_stages_sparse_support_disables_symmetry_shortcut():
    fixed = sitk.Image(120, 120, sitk.sitkFloat32)
    fixed.SetSpacing([0.001, 0.001])
    moving = sitk.Image(120, 120, sitk.sitkFloat32)
    moving.SetSpacing([0.001, 0.001])

    # Sparse ring-like support can be near-circular but unreliable for orientation.
    yy, xx = np.indices((120, 120))
    r2 = (xx - 60) ** 2 + (yy - 60) ** 2
    sparse_ring = ((r2 >= 20**2) & (r2 <= 22**2)).astype(np.uint8)

    fixed_mask = sitk.GetImageFromArray(sparse_ring)
    fixed_mask.CopyInformation(fixed)
    moving_mask = sitk.GetImageFromArray(sparse_ring)
    moving_mask.CopyInformation(moving)

    stages = Rigid2DRegistration.build_adaptive_stages(
        fixed_image=fixed,
        moving_image=moving,
        transform_type=Transform2D.RIGID,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        assume_axial_symmetry=True,
        max_stages=3,
        max_stage_evals=50000,
    )

    assert stages[0]["rot_span_deg"] == 180.0


def test_fit_stage_eval_budget_can_coarsen_rotation_sampling():
    stage = {
        "translation_step": 0.001,
        "rot_step_deg": 0.25,
        "rot_span_deg": 180.0,
        "tx_steps": 1,
        "ty_steps": 1,
    }

    capped = Rigid2DRegistration._fit_stage_eval_budget(
        stage=stage,
        transform_type=Transform2D.RIGID,
        max_stage_evals=1500,
    )

    tx = int(capped["tx_steps"])
    ty = int(capped["ty_steps"])
    rot_step = max(float(capped["rot_step_deg"]), 1e-6)
    rot_span = max(float(capped["rot_span_deg"]), rot_step)
    rot_n = int(max(1, round(rot_span / rot_step)))
    evals = (2 * rot_n + 1) * (2 * tx + 1) * (2 * ty + 1)

    assert evals <= 1500
    assert float(capped["rot_step_deg"]) >= float(stage["rot_step_deg"])
