from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from sar_pattern_validation.plotting import plot_gamma_results
from sar_pattern_validation.utils import image_extent_mm
from sar_pattern_validation.workflow_config import PlottingConfig


class GammaMapEvaluator:
    """
    Gamma evaluation for 2D SAR maps (paper-aligned, linear space)

    Inputs
    ------
    reference_sar_linear :
        sitk.Image, peak-normalized linear SAR on the reference grid.

    measured_sar_linear :
        sitk.Image, peak-normalized linear SAR on the measured grid.

    measured_to_reference_transform :
        sitk.Transform mapping measured coordinates -> reference coordinates.
        This is the same mapping used to build the overlay (measured warped onto reference).

    Gamma definition (abstract)
    ---------------------------
    Each evaluated point (measured after registration) is compared to all reference points
    within a search neighborhood to find the minimum:

        Γ^2 = (Δr / Δd)^2 + (ΔSAR / ΔD)^2

    where:
      Δd : distance-to-agreement (mm)
      ΔD : dose-to-agreement (fraction of peak, because inputs are peak-normalized)

    Because both inputs are peak-normalized (peak = 1):
        ΔD_linear = dose_to_agreement_percent / 100

    ROI / masking
    -------------
    The abstract masks values before normalization based on an absolute threshold.
    Those masks are supplied here (optional) and are used only to define the evaluation region.

    On the reference grid:
        evaluation_roi = reference_mask  ∩  resampled(measured_mask)

    If no masks are provided, evaluation covers all pixels.
    """

    def __init__(
        self,
        *,
        reference_sar_linear: sitk.Image,
        measured_sar_linear: sitk.Image,
        measured_to_reference_transform: sitk.Transform,
        dose_to_agreement_percent: float = 5.0,  # ΔD in % of peak (peak-normalized inputs)
        distance_to_agreement_mm: float = 2.0,  # Δd in mm
        gamma_cap: float = 2.0,
    ):
        self.reference_sar_linear = sitk.Cast(reference_sar_linear, sitk.sitkFloat32)
        self.measured_sar_linear = sitk.Cast(measured_sar_linear, sitk.sitkFloat32)
        self.measured_to_reference_transform = measured_to_reference_transform

        self.dose_to_agreement_percent = float(dose_to_agreement_percent)
        self.distance_to_agreement_mm = float(distance_to_agreement_mm)
        self.gamma_cap = float(gamma_cap)

        self.reference_mask_u8: sitk.Image | None = None
        self.measured_mask_u8: sitk.Image | None = None

        self.gamma_map: np.ndarray | None = None
        self.evaluation_mask: np.ndarray | None = None
        self.pass_rate_percent: float | None = None
        self.evaluated_pixel_count: int | None = None
        self.passed_pixel_count: int | None = None
        self.failed_pixel_count: int | None = None

    # ------------------------------ public ------------------------------

    def compute(self) -> None:
        """
        Compute gamma on the reference grid.

        Output fields
        -------------
        gamma_map :
            float32 array (H,W) with NaN outside evaluation_mask (if masks provided).

        evaluation_mask :
            bool array (H,W) indicating where gamma is evaluated.

        pass_rate_percent :
            100 * (# pixels with gamma <= 1) / (# evaluated pixels)
        """
        measured_on_reference = self._resample_measured_onto_reference()

        reference_arr = sitk.GetArrayFromImage(self.reference_sar_linear).astype(
            np.float32
        )
        measured_arr = sitk.GetArrayFromImage(measured_on_reference).astype(np.float32)

        distance_to_agreement_pixels = self._mm_to_pixels(
            self.distance_to_agreement_mm, self.reference_sar_linear
        )
        dose_to_agreement_fraction = max(self.dose_to_agreement_percent / 100.0, 1e-12)

        evaluation_mask = self._build_evaluation_mask_on_reference()
        gamma = self._gamma_2d_peak_normalized(
            reference=reference_arr,
            evaluation=measured_arr,
            distance_to_agreement_pixels=distance_to_agreement_pixels,
            dose_to_agreement_fraction=dose_to_agreement_fraction,
            gamma_cap=self.gamma_cap,
        )

        if evaluation_mask is not None:
            gamma[~evaluation_mask] = np.nan
            evaluated_mask = evaluation_mask
        else:
            evaluated_mask = ~np.isnan(gamma)

        evaluated_pixel_count = int(np.sum(evaluated_mask))
        passed_pixel_count = int(np.sum((gamma <= 1.0) & evaluated_mask))
        failed_pixel_count = max(0, evaluated_pixel_count - passed_pixel_count)

        self.gamma_map = gamma
        self.evaluation_mask = evaluated_mask
        self.evaluated_pixel_count = evaluated_pixel_count
        self.passed_pixel_count = passed_pixel_count
        self.failed_pixel_count = failed_pixel_count
        self.pass_rate_percent = (
            100.0 * passed_pixel_count / max(evaluated_pixel_count, 1)
        )

    def show(
        self,
        *,
        gamma_image_save_path: Path | None = None,
        failure_image_save_path: Path | None = None,
        plotting_config: PlottingConfig | None = None,
    ) -> None:
        """
        Display gamma map and failures (gamma > 1) on the reference grid.
        """
        if (
            self.gamma_map is None
            or self.evaluation_mask is None
            or self.pass_rate_percent is None
            or self.evaluated_pixel_count is None
            or self.passed_pixel_count is None
            or self.failed_pixel_count is None
        ):
            raise RuntimeError("compute() must be called before show().")

        plot_gamma_results(
            gamma_map=self.gamma_map,
            evaluation_mask=self.evaluation_mask,
            gamma_cap=self.gamma_cap,
            extent_mm=self._extent_mm(self.reference_sar_linear),
            gamma_image_save_path=gamma_image_save_path,
            failure_image_save_path=failure_image_save_path,
            plotting_config=plotting_config,
        )

    def _resample_measured_onto_reference(self) -> sitk.Image:
        """
        Resample measured_sar_linear onto the reference grid using measured->reference transform.
        Outside-domain default is 0.0 (handled by ROI masking if masks are provided).
        """
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(self.reference_sar_linear)
        resampler.SetTransform(self.measured_to_reference_transform.GetInverse())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0.0)
        return resampler.Execute(self.measured_sar_linear)

    def _build_evaluation_mask_on_reference(self) -> np.ndarray | None:
        """
        evaluation_mask = reference_mask ∩ resampled(measured_mask) on the reference grid.

        Returns None when neither mask is provided.
        """
        reference_roi = None
        if self.reference_mask_u8 is not None:
            reference_roi = sitk.GetArrayFromImage(self.reference_mask_u8).astype(bool)

        measured_roi_on_reference = None
        if self.measured_mask_u8 is not None:
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(self.reference_sar_linear)
            resampler.SetTransform(self.measured_to_reference_transform.GetInverse())
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetDefaultPixelValue(0)
            measured_mask_on_ref = resampler.Execute(self.measured_mask_u8)
            measured_roi_on_reference = sitk.GetArrayFromImage(
                measured_mask_on_ref
            ).astype(bool)

        if reference_roi is None and measured_roi_on_reference is None:
            return None
        if reference_roi is None:
            return measured_roi_on_reference
        if measured_roi_on_reference is None:
            return reference_roi
        return reference_roi & measured_roi_on_reference

    @staticmethod
    def _mm_to_pixels(distance_mm: float, img: sitk.Image) -> int:
        sx_m, sy_m = img.GetSpacing()
        min_spacing_m = max(min(float(sx_m), float(sy_m)), 1e-12)
        radius_m = float(distance_mm) / 1000.0
        return max(1, int(round(radius_m / min_spacing_m)))

    @staticmethod
    def _gamma_2d_peak_normalized(
        *,
        reference: np.ndarray,
        evaluation: np.ndarray,
        distance_to_agreement_pixels: int,
        dose_to_agreement_fraction: float,
        gamma_cap: float,
    ) -> np.ndarray:
        """
        Gamma for peak-normalized linear SAR (peak=1).

        Spatial term:
            (Δr / Δd_pixels)^2

        Dose term:
            (ΔSAR / ΔD_fraction)^2
        """
        if reference.shape != evaluation.shape or reference.ndim != 2:
            raise ValueError(
                "reference and evaluation must be 2D arrays with matching shape."
            )

        dta = int(distance_to_agreement_pixels)
        dose_tol = float(max(dose_to_agreement_fraction, 1e-12))
        search_radius = max(1, int(np.ceil(float(gamma_cap) * dta)))

        reference_padded = np.pad(reference, search_radius, mode="edge")

        offsets = np.arange(-search_radius, search_radius + 1, dtype=np.int32)
        rr_grid, cc_grid = np.meshgrid(offsets, offsets, indexing="ij")
        within_radius = (rr_grid**2 + cc_grid**2) <= search_radius**2
        rr = rr_grid[within_radius]
        cc = cc_grid[within_radius]
        dist_pix_sq = rr.astype(np.float32) ** 2 + cc.astype(np.float32) ** 2
        spatial_term_sq = dist_pix_sq / float(dta * dta)

        h, w = reference.shape
        gamma_cap_sq = float(gamma_cap * gamma_cap)
        gamma_sq = np.full((h, w), gamma_cap_sq, dtype=np.float32)
        finite_evaluation = np.isfinite(evaluation)

        for dr, dc, spatial_sq in zip(rr, cc, spatial_term_sq, strict=False):
            ref_shifted = reference_padded[
                search_radius + dr : search_radius + dr + h,
                search_radius + dc : search_radius + dc + w,
            ]
            with np.errstate(invalid="ignore"):
                dose_term_sq = ((evaluation - ref_shifted) / dose_tol) ** 2
            candidate_sq = spatial_sq + dose_term_sq
            np.minimum(gamma_sq, candidate_sq, out=gamma_sq, where=finite_evaluation)

        gamma = np.sqrt(gamma_sq).astype(np.float32, copy=False)
        gamma[~finite_evaluation] = np.nan
        gamma[gamma > gamma_cap] = float(gamma_cap)
        return gamma

    @staticmethod
    def _extent_mm(img: sitk.Image) -> tuple[float, float, float, float]:
        return image_extent_mm(img)
