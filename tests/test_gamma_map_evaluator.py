# tests/test_gamma_map_evaluator.py

import numpy as np
import pytest
import SimpleITK as sitk

from sar_pattern_validation.gamma_eval import GammaMapEvaluator

# ------------------------- helpers -------------------------


def _sitk_from_array(
    arr: np.ndarray,
    *,
    spacing_m: tuple[float, float] = (0.001, 0.001),
    centered_origin: bool = True,
) -> sitk.Image:
    """
    Create a 2D float32 SimpleITK image from a numpy array [H, W].

    spacing is (sx, sy) in meters.
    origin is centered if centered_origin=True, i.e. origin = -0.5 * size * spacing.
    """
    assert arr.ndim == 2
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    sx, sy = spacing_m
    img.SetSpacing([float(sx), float(sy)])

    if centered_origin:
        w, h = img.GetSize()
        ox = -0.5 * w * sx
        oy = -0.5 * h * sy
        img.SetOrigin([float(ox), float(oy)])
    return img


def _identity_transform() -> sitk.Transform:
    # Use a 2D Euler transform (rigid) with identity parameters
    tx = sitk.Euler2DTransform()
    tx.SetAngle(0.0)
    tx.SetTranslation((0.0, 0.0))
    return tx


def _u8_mask_from_bool(mask: np.ndarray, like: sitk.Image) -> sitk.Image:
    """Make a uint8 SITK mask image with the same geometry as `like`."""
    m = sitk.GetImageFromArray(mask.astype(np.uint8))
    m.CopyInformation(like)
    return m


# ------------------------- tests -------------------------


def test_gamma_dose_only_uniform_fields_is_analytic():
    """
    Analytic dose-only gamma:
      reference = constant 1
      measured  = constant (1 + delta)
      spatial term can be 0 (same pixel)
      => gamma = |delta| / dose_tol everywhere (capped by gamma_cap)

    This strongly verifies the ΔSAR / ΔD term.
    """
    h, w = 64, 80
    reference = np.ones((h, w), dtype=np.float32)

    delta = 0.05
    measured = np.ones((h, w), dtype=np.float32) * (1.0 + delta)

    dose_percent = 5.0
    dose_tol = dose_percent / 100.0  # because peak-normalized

    ref_img = _sitk_from_array(reference, spacing_m=(0.001, 0.001))
    meas_img = _sitk_from_array(measured, spacing_m=(0.001, 0.001))

    evaluator = GammaMapEvaluator(
        reference_sar_linear=ref_img,
        measured_sar_linear=meas_img,
        measured_to_reference_transform=_identity_transform(),
        dose_to_agreement_percent=dose_percent,
        distance_to_agreement_mm=2.0,
        gamma_cap=10.0,  # avoid cap interfering
    )
    evaluator.compute()

    g = evaluator.gamma_map
    assert g is not None
    assert np.isfinite(g).all()

    expected = abs(delta) / dose_tol
    # should be uniform
    assert np.allclose(g, expected, rtol=0, atol=1e-6)

    # pass rate: expected == 1.0 exactly here (0.05/0.05)
    assert evaluator.pass_rate_percent == pytest.approx(100.0, abs=1e-6)


def test_gamma_distance_only_single_peak_shift_is_analytic():
    """
    Analytic distance-only gamma at the *measured peak location*:

      reference: single peak at center
      measured:  same peak shifted by +shift_pixels in x
      identity transform -> measured peak stays shifted on reference grid.

    For the shifted peak pixel, best match is the reference peak (dose term 0),
    so gamma = shift_pixels / dta_pixels (assuming dta_pixels >= shift_pixels).
    """
    h, w = 101, 101
    reference = np.zeros((h, w), dtype=np.float32)
    measured = np.zeros((h, w), dtype=np.float32)

    r0, c0 = h // 2, w // 2
    shift_pixels = 2

    reference[r0, c0] = 1.0
    measured[r0, c0 + shift_pixels] = 1.0

    dta_mm = 4.0  # spacing 1mm => dta_pixels = 4
    dta_pixels = 4

    ref_img = _sitk_from_array(reference, spacing_m=(0.001, 0.001))
    meas_img = _sitk_from_array(measured, spacing_m=(0.001, 0.001))

    evaluator = GammaMapEvaluator(
        reference_sar_linear=ref_img,
        measured_sar_linear=meas_img,
        measured_to_reference_transform=_identity_transform(),
        dose_to_agreement_percent=1.0,
        distance_to_agreement_mm=dta_mm,
        gamma_cap=10.0,
    )
    evaluator.compute()
    g = evaluator.gamma_map
    assert g is not None

    expected = shift_pixels / float(dta_pixels)
    got = float(g[r0, c0 + shift_pixels])
    assert got == pytest.approx(expected, abs=1e-6)


def test_mm_to_pixels_rounding_for_non_integer_distance():
    """
    Non-integer mm distances should be converted to the correct integer pixel
    radius via rounding.
    """
    # Use a spacing where the conversion is non-integer in pixels.
    # spacing = 0.9 mm = 0.0009 m
    spacing_m = (0.0009, 0.0009)
    dummy = _sitk_from_array(np.zeros((10, 10), dtype=np.float32), spacing_m=spacing_m)

    # distance_mm = 2.0 mm => radius_m=0.002, /0.0009=2.222.. => round -> 2 pixels
    dta_mm = 2.0
    dta_pix = GammaMapEvaluator._mm_to_pixels(dta_mm, dummy)
    assert dta_pix == 2

    # distance_mm = 2.6 mm => 0.0026/0.0009=2.888.. => round -> 3 pixels
    dta_mm2 = 2.6
    dta_pix2 = GammaMapEvaluator._mm_to_pixels(dta_mm2, dummy)
    assert dta_pix2 == 3

    assert dta_pix2 > dta_pix


def test_gamma_distance_case_between_one_and_cap_is_analytic():
    """
    When the best spatial match lies beyond one DTA but within gamma_cap * DTA,
    gamma should still be found correctly instead of saturating at the cap.
    """
    h, w = 101, 101
    reference = np.zeros((h, w), dtype=np.float32)
    measured = np.zeros((h, w), dtype=np.float32)

    r0, c0 = h // 2, w // 2
    shift_pixels = 3

    reference[r0, c0] = 1.0
    measured[r0, c0 + shift_pixels] = 1.0

    ref_img = _sitk_from_array(reference, spacing_m=(0.001, 0.001))
    meas_img = _sitk_from_array(measured, spacing_m=(0.001, 0.001))

    evaluator = GammaMapEvaluator(
        reference_sar_linear=ref_img,
        measured_sar_linear=meas_img,
        measured_to_reference_transform=_identity_transform(),
        dose_to_agreement_percent=0.0001,
        distance_to_agreement_mm=2.0,
        gamma_cap=2.0,
    )
    evaluator.compute()

    g = evaluator.gamma_map
    assert g is not None
    assert float(g[r0, c0 + shift_pixels]) == pytest.approx(1.5, abs=1e-6)


def test_gamma_distance_case_beyond_cap_is_capped():
    """
    If the nearest exact spatial match lies beyond gamma_cap * DTA, gamma should
    saturate at gamma_cap.
    """
    h, w = 101, 101
    reference = np.zeros((h, w), dtype=np.float32)
    measured = np.zeros((h, w), dtype=np.float32)

    r0, c0 = h // 2, w // 2
    shift_pixels = 5

    reference[r0, c0] = 1.0
    measured[r0, c0 + shift_pixels] = 1.0

    ref_img = _sitk_from_array(reference, spacing_m=(0.001, 0.001))
    meas_img = _sitk_from_array(measured, spacing_m=(0.001, 0.001))

    evaluator = GammaMapEvaluator(
        reference_sar_linear=ref_img,
        measured_sar_linear=meas_img,
        measured_to_reference_transform=_identity_transform(),
        dose_to_agreement_percent=0.0001,
        distance_to_agreement_mm=2.0,
        gamma_cap=2.0,
    )
    evaluator.compute()

    g = evaluator.gamma_map
    assert g is not None
    assert float(g[r0, c0 + shift_pixels]) == pytest.approx(2.0, abs=1e-6)


def test_evaluation_mask_reference_only_is_used_as_is():
    """
    If only reference_mask_u8 is provided, evaluation_mask should match it exactly.
    """
    h, w = 50, 60
    ref = np.random.RandomState(0).rand(h, w).astype(np.float32)
    meas = np.random.RandomState(1).rand(h, w).astype(np.float32)

    ref_img = _sitk_from_array(ref, spacing_m=(0.001, 0.001))
    meas_img = _sitk_from_array(meas, spacing_m=(0.001, 0.001))

    # make a simple rectangular ROI
    mask = np.zeros((h, w), dtype=bool)
    mask[10:30, 12:40] = True
    ref_mask_u8 = _u8_mask_from_bool(mask, ref_img)

    evaluator = GammaMapEvaluator(
        reference_sar_linear=ref_img,
        measured_sar_linear=meas_img,
        measured_to_reference_transform=_identity_transform(),
        dose_to_agreement_percent=5.0,
        distance_to_agreement_mm=2.0,
    )
    evaluator.reference_mask_u8 = ref_mask_u8
    evaluator.compute()

    assert evaluator.evaluation_mask is not None
    assert evaluator.evaluated_pixel_count == int(mask.sum())
    assert np.array_equal(evaluator.evaluation_mask, mask)


def test_evaluation_mask_intersection_reference_and_resampled_measured():
    """
    If both masks are provided:
      evaluation_mask = reference_mask ∩ resampled(measured_mask)

    Here we use identity transform so resampling does not change mask geometry.
    """
    h, w = 64, 64
    ref = np.ones((h, w), dtype=np.float32)
    meas = np.ones((h, w), dtype=np.float32)

    ref_img = _sitk_from_array(ref, spacing_m=(0.001, 0.001))
    meas_img = _sitk_from_array(meas, spacing_m=(0.001, 0.001))

    ref_mask = np.zeros((h, w), dtype=bool)
    ref_mask[10:50, 10:50] = True

    meas_mask = np.zeros((h, w), dtype=bool)
    meas_mask[30:60, 30:60] = True

    ref_mask_u8 = _u8_mask_from_bool(ref_mask, ref_img)
    meas_mask_u8 = _u8_mask_from_bool(meas_mask, meas_img)

    evaluator = GammaMapEvaluator(
        reference_sar_linear=ref_img,
        measured_sar_linear=meas_img,
        measured_to_reference_transform=_identity_transform(),
        dose_to_agreement_percent=5.0,
        distance_to_agreement_mm=2.0,
    )
    evaluator.reference_mask_u8 = ref_mask_u8
    evaluator.measured_mask_u8 = meas_mask_u8
    evaluator.compute()

    expected = ref_mask & meas_mask
    assert evaluator.evaluation_mask is not None
    assert np.array_equal(evaluator.evaluation_mask, expected)
    assert evaluator.evaluated_pixel_count == int(expected.sum())
