"""
2D SAR Image Registration (SimpleITK, exhaustive search)
========================================================
1. Initialize transform via center-of-gravity (Moments).
2. Run coarse -> fine exhaustive search (rotation + translation) with Mattes MI.
3. Resample the measured (moving) image onto the reference (fixed) grid.
"""

from __future__ import annotations

import logging
from enum import Enum

import numpy as np
import SimpleITK as sitk

__all__ = ["Transform2D", "Rigid2DRegistration"]


class Transform2D(str, Enum):
    TRANSLATE = "translate"
    RIGID = "rigid"


class Rigid2DRegistration:
    def __init__(
        self,
        fixed_image: sitk.Image,
        moving_image: sitk.Image,
        *,
        transform_type: Transform2D = Transform2D.RIGID,
    ):
        self.fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
        self.moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
        self.transform_type = transform_type
        self._log = logging.getLogger("Rigid2DRegistration")

    def run(
        self,
        *,
        stages: list[dict],
        fixed_mask: sitk.Image | None = None,
        moving_mask: sitk.Image | None = None,
    ) -> tuple[sitk.Image, sitk.Transform]:
        fixed = self._sanitize_spacing(self.fixed_image)
        moving = self._sanitize_spacing(self.moving_image)

        mask_fixed = (
            self._sanitize_spacing(fixed_mask) if fixed_mask is not None else None
        )
        mask_moving = (
            self._sanitize_spacing(moving_mask) if moving_mask is not None else None
        )
        mask_fixed = self._empty_mask_to_none(mask_fixed)
        mask_moving = self._empty_mask_to_none(mask_moving)

        # Log image geometry for debugging
        self._log_image_geometry("Fixed", fixed)
        self._log_image_geometry("Moving", moving)
        if mask_fixed is not None:
            self._log_mask_stats("Fixed mask", mask_fixed)
        if mask_moving is not None:
            self._log_mask_stats("Moving mask", mask_moving)

        # Pre-compute effective stages (with translation steps clamped to the
        # physical extent of the smaller image) so we know the max search range
        # before building the expanded moving buffer.
        effective_stages = [
            self._clamp_stage_steps(s, fixed, moving, self._log) for s in stages
        ]
        self._log.debug(
            "Configured %d effective registration stages", len(effective_stages)
        )
        for idx, stage in enumerate(effective_stages, start=1):
            self._log.debug(
                "[stage %d config] t_step=%.6f m, tx=±%d, ty=±%d, rot_step=%s deg, rot_span=%s deg",
                idx,
                float(stage["translation_step"]),
                int(stage["tx_steps"]),
                int(stage["ty_steps"]),
                stage.get("rot_step_deg", "n/a"),
                stage.get("rot_span_deg", "n/a"),
            )

        # Expand moving (and its mask) to cover the union of both physical extents
        # plus a safety padding derived from rotation, translation search, and
        # initial center offsets. Off-center SAR peaks outside the reference FOV
        # are preserved while the moving-valid mask prevents background fill from
        # polluting Mattes MI.
        # The original `moving` is kept for the final high-fidelity resample output.
        moving_fill = float(np.nanmin(sitk.GetArrayViewFromImage(moving)))
        pad_x, pad_y = self._compute_union_padding(fixed, moving, effective_stages)
        self._log.debug(
            "Union expansion padding: pad_x=%.6f m, pad_y=%.6f m, moving_fill=%.6f",
            pad_x,
            pad_y,
            moving_fill,
        )
        moving_reg = self._expand_to_union(
            fixed, moving, moving_fill, sitk.sitkLinear, pad_x, pad_y
        )
        self._log_image_geometry("Moving (union+pad)", moving_reg)

        # Build a mask that marks which pixels of moving_reg came from the original
        # moving image (not from background fill).  This is always used as the
        # metric moving mask so that background zeros never pollute Mattes MI,
        # regardless of whether the caller supplies an explicit moving mask.
        ones_arr = np.ones((moving.GetSize()[1], moving.GetSize()[0]), dtype=np.uint8)
        ones_img = sitk.GetImageFromArray(ones_arr)
        ones_img.CopyInformation(moving)
        valid_reg_mask = self._expand_to_union(
            fixed, ones_img, 0, sitk.sitkNearestNeighbor, pad_x, pad_y
        )

        # Combine with the caller-supplied moving mask (intersection = AND).
        if mask_moving is not None:
            user_moving_reg = self._expand_to_union(
                fixed, mask_moving, 0, sitk.sitkNearestNeighbor, pad_x, pad_y
            )
            eff_moving_mask_reg = sitk.Cast(
                sitk.Cast(user_moving_reg, sitk.sitkFloat32)
                * sitk.Cast(valid_reg_mask, sitk.sitkFloat32),
                sitk.sitkUInt8,
            )
        else:
            eff_moving_mask_reg = valid_reg_mask
        eff_moving_mask_reg = self._empty_mask_to_none(eff_moving_mask_reg)

        self._log_mask_stats("Moving valid extent mask (union grid)", valid_reg_mask)
        if eff_moving_mask_reg is not None:
            self._log_mask_stats("Moving effective metric mask", eff_moving_mask_reg)
        else:
            self._log.debug("Moving effective metric mask: none")

        resample_transform: sitk.Transform | None = None

        for i, effective_stage in enumerate(effective_stages, 1):
            # Use original `moving` (not the union-expanded copy) for moments
            # initialisation so background fill does not shift the intensity centroid.
            init = (
                sitk.Transform(resample_transform)
                if resample_transform is not None
                else self._moments_init(
                    fixed,
                    moving,
                    fixed_mask=mask_fixed,
                    moving_mask=mask_moving,
                )
            )
            self._log_transform_init(f"Stage {i}/{len(effective_stages)}", init)

            attempts: list[tuple[sitk.Image | None, sitk.Image | None, str]] = []
            if mask_fixed is not None and eff_moving_mask_reg is not None:
                attempts.append((mask_fixed, eff_moving_mask_reg, "fixed+moving masks"))
            if mask_fixed is not None:
                attempts.append((mask_fixed, None, "fixed mask"))
            if eff_moving_mask_reg is not None:
                attempts.append((None, eff_moving_mask_reg, "moving extent mask"))
            attempts.append((None, None, "no masks"))

            last_exception: RuntimeError | None = None

            for mf, mm, label in attempts:
                attempt_init = self._clone_transform(init)
                method = self._build_method(attempt_init, effective_stage, mf, mm)
                self._log.debug(
                    "[stage %d] attempt=%s, init=%s, fixed_mask=%s, moving_mask=%s",
                    i,
                    label,
                    self._transform_to_debug_string(attempt_init),
                    self._mask_to_debug_string(mf),
                    self._mask_to_debug_string(mm),
                )

                self._log.info(
                    self._stage_str(
                        i,
                        len(effective_stages),
                        effective_stage,
                        mask_on=mf is not None,
                    )
                )

                try:
                    resample_transform = method.Execute(fixed, moving_reg)
                    self._log.info(
                        "  -> metric: %.6f (%s)", method.GetMetricValue(), label
                    )
                    self._log.debug(
                        "[stage %d] %s optimizer_position=%s stop_condition=%s",
                        i,
                        label,
                        method.GetOptimizerPosition(),
                        method.GetOptimizerStopConditionDescription(),
                    )
                    if resample_transform is not None:
                        self._log_transform_result(
                            f"Stage {i} success", resample_transform
                        )
                    last_exception = None
                    break
                except RuntimeError as ex:
                    self._log.debug(
                        "[stage %d] %s raised RuntimeError: %s",
                        i,
                        label,
                        str(ex),
                    )
                    last_exception = ex
                    if "All samples map outside moving image buffer" in str(ex):
                        self._log.info(
                            "  -> %s failed (insufficient overlap). Trying fallback...",
                            label,
                        )
                        continue
                    raise

            if last_exception is not None:
                if "All samples map outside moving image buffer" in str(last_exception):
                    self._log.info(
                        "  -> stage failed after mask fallbacks. Retrying with identity init + no masks."
                    )
                    if self.transform_type == Transform2D.TRANSLATE:
                        identity_init: sitk.Transform = sitk.TranslationTransform(2)
                    else:
                        identity_init = sitk.Euler2DTransform()

                    method = self._build_method(
                        identity_init, effective_stage, None, None
                    )
                    self._log.debug(
                        "[stage %d] identity fallback init=%s",
                        i,
                        self._transform_to_debug_string(identity_init),
                    )
                    try:
                        resample_transform = method.Execute(fixed, moving_reg)
                        self._log.info(
                            "  -> metric: %.6f (identity-init fallback)",
                            method.GetMetricValue(),
                        )
                        self._log.debug(
                            "[stage %d] identity fallback optimizer_position=%s stop_condition=%s",
                            i,
                            method.GetOptimizerPosition(),
                            method.GetOptimizerStopConditionDescription(),
                        )
                        if resample_transform is not None:
                            self._log_transform_result(
                                f"Stage {i} identity-fallback success",
                                resample_transform,
                            )
                        continue
                    except RuntimeError as ex:
                        last_exception = ex

                raise last_exception

        if resample_transform is None:
            raise RuntimeError("No transform produced (empty stages?).")

        aligned = self._resample(moving, fixed, resample_transform)
        return aligned, resample_transform.GetInverse()

    @staticmethod
    def build_adaptive_stages(
        *,
        fixed_image: sitk.Image,
        moving_image: sitk.Image,
        transform_type: Transform2D,
        fixed_mask: sitk.Image | None = None,
        moving_mask: sitk.Image | None = None,
        assume_axial_symmetry: bool = True,
        max_stages: int = 4,
        max_stage_evals: int = 50000,
    ) -> list[dict[str, float | int]]:
        """
        Build a coarse-to-fine stage schedule from image geometry and masks.

        The first stage is sized from support extents and centroid offset, then
        successive stages apply fixed reduction ratios until the schedule is local.
        """
        log = logging.getLogger("Rigid2DRegistration")
        if max_stages < 1:
            raise ValueError("max_stages must be >= 1")
        if max_stage_evals < 1:
            raise ValueError("max_stage_evals must be >= 1")

        fx, fy = Rigid2DRegistration._support_extent_m(fixed_image, fixed_mask)
        mx, my = Rigid2DRegistration._support_extent_m(moving_image, moving_mask)
        min_extent = max(min(fx, fy, mx, my), 1e-6)
        support_cap_x = max(1e-6, min(fx, mx))
        support_cap_y = max(1e-6, min(fy, my))
        log.debug(
            "[adaptive] fixed_support_extent=[%.4f, %.4f] m, moving_support_extent=[%.4f, %.4f] m, "
            "min_extent=%.4f m, support_cap=[%.4f, %.4f] m",
            fx,
            fy,
            mx,
            my,
            min_extent,
            support_cap_x,
            support_cap_y,
        )

        fixed_cx, fixed_cy = Rigid2DRegistration._support_centroid_m(
            fixed_image, fixed_mask
        )
        moving_cx, moving_cy = Rigid2DRegistration._support_centroid_m(
            moving_image, moving_mask
        )
        offset_x = abs(fixed_cx - moving_cx)
        offset_y = abs(fixed_cy - moving_cy)
        log.debug(
            "[adaptive] fixed_centroid=[%.4f, %.4f] m, moving_centroid=[%.4f, %.4f] m, "
            "centroid_offset=[%.4f, %.4f] m",
            fixed_cx,
            fixed_cy,
            moving_cx,
            moving_cy,
            offset_x,
            offset_y,
        )

        t_step_1 = float(np.clip(min_extent / 8.0, 0.003, 0.02))
        range_x = offset_x + 0.25 * max(fx, mx)
        range_y = offset_y + 0.25 * max(fy, my)
        tx_steps_1 = int(np.clip(np.ceil(range_x / t_step_1), 1, 12))
        ty_steps_1 = int(np.clip(np.ceil(range_y / t_step_1), 1, 12))

        fixed_ecc = Rigid2DRegistration._support_eccentricity(fixed_mask)
        moving_ecc = Rigid2DRegistration._support_eccentricity(moving_mask)
        fixed_symmetry_ok = Rigid2DRegistration._axial_symmetry_reliable(
            fixed_mask, fixed_ecc
        )
        moving_symmetry_ok = Rigid2DRegistration._axial_symmetry_reliable(
            moving_mask, moving_ecc
        )
        log.debug(
            "[adaptive] eccentricity: fixed=%.3f (ok=%s), moving=%.3f (ok=%s), assume_symmetry=%s",
            fixed_ecc,
            fixed_symmetry_ok,
            moving_ecc,
            moving_symmetry_ok,
            assume_axial_symmetry,
        )
        if assume_axial_symmetry and fixed_symmetry_ok and moving_symmetry_ok:
            rot_span_1 = 90.0
        else:
            rot_span_1 = 180.0
        rot_step_1 = 4.0
        log.debug(
            "[adaptive] stage-1 rotation span: %.1f° (reduced_by_symmetry=%s)",
            rot_span_1,
            rot_span_1 == 90.0,
        )

        stages: list[dict[str, float | int]] = []
        stage = dict(
            translation_step=t_step_1,
            rot_step_deg=rot_step_1,
            rot_span_deg=rot_span_1,
            tx_steps=tx_steps_1,
            ty_steps=ty_steps_1,
        )
        stage = Rigid2DRegistration._cap_steps_by_extent(
            stage=stage,
            cap_extent_x_m=support_cap_x,
            cap_extent_y_m=support_cap_y,
        )

        if transform_type == Transform2D.TRANSLATE:
            stage["rot_step_deg"] = 0.0
            stage["rot_span_deg"] = 0.0

        stage = Rigid2DRegistration._fit_stage_eval_budget(
            stage=stage,
            transform_type=transform_type,
            max_stage_evals=max_stage_evals,
        )
        log.debug(
            "[adaptive] stage-1 final: t_step=%.6f, steps=±(%d,%d), rot_step=%.1f°, rot_span=%.1f°",
            stage["translation_step"],
            stage["tx_steps"],
            stage["ty_steps"],
            stage["rot_step_deg"],
            stage["rot_span_deg"],
        )
        stages.append(stage)

        for _ in range(max_stages - 1):
            prev = stages[-1]
            t_step = max(0.001, float(prev["translation_step"]) / 3.0)
            tx_steps = max(1, int(np.ceil(int(prev["tx_steps"]) / 2.0)))
            ty_steps = max(1, int(np.ceil(int(prev["ty_steps"]) / 2.0)))

            if transform_type == Transform2D.RIGID:
                rot_step = max(0.5, float(prev["rot_step_deg"]) / 2.0)
                rot_span = max(rot_step, float(prev["rot_span_deg"]) / 3.0)
            else:
                rot_step = 0.0
                rot_span = 0.0

            next_stage = dict(
                translation_step=float(t_step),
                rot_step_deg=float(rot_step),
                rot_span_deg=float(rot_span),
                tx_steps=int(tx_steps),
                ty_steps=int(ty_steps),
            )
            next_stage = Rigid2DRegistration._cap_steps_by_extent(
                stage=next_stage,
                cap_extent_x_m=support_cap_x,
                cap_extent_y_m=support_cap_y,
            )

            next_stage = Rigid2DRegistration._fit_stage_eval_budget(
                stage=next_stage,
                transform_type=transform_type,
                max_stage_evals=max_stage_evals,
            )
            log.debug(
                "[adaptive] stage-%d final: t_step=%.6f, steps=±(%d,%d), rot_step=%.1f°, rot_span=%.1f°",
                len(stages) + 1,
                next_stage["translation_step"],
                next_stage["tx_steps"],
                next_stage["ty_steps"],
                next_stage["rot_step_deg"],
                next_stage["rot_span_deg"],
            )
            stages.append(next_stage)

            if (
                int(next_stage["tx_steps"]) == 1
                and int(next_stage["ty_steps"]) == 1
                and (
                    transform_type == Transform2D.TRANSLATE
                    or float(next_stage["rot_span_deg"])
                    <= float(next_stage["rot_step_deg"])
                )
            ):
                break

        return stages

    @staticmethod
    def _cap_steps_by_extent(
        *,
        stage: dict[str, float | int],
        cap_extent_x_m: float,
        cap_extent_y_m: float,
    ) -> dict[str, float | int]:
        """Cap tx/ty steps by physical support extents for the given translation step."""
        out = dict(stage)
        t_step = float(out["translation_step"])
        if t_step <= 0:
            return out

        max_steps_x = max(1, int(cap_extent_x_m / (2.0 * t_step)))
        max_steps_y = max(1, int(cap_extent_y_m / (2.0 * t_step)))
        out["tx_steps"] = min(int(out["tx_steps"]), max_steps_x)
        out["ty_steps"] = min(int(out["ty_steps"]), max_steps_y)
        return out

    @staticmethod
    def _support_extent_m(
        img: sitk.Image, mask: sitk.Image | None
    ) -> tuple[float, float]:
        """Return support extent (x, y) in meters from mask; fall back to full image."""
        spacing = np.array(img.GetSpacing(), dtype=float)
        size = np.array(img.GetSize(), dtype=float)
        full_extent = size * spacing

        if mask is None:
            return float(full_extent[0]), float(full_extent[1])

        arr = sitk.GetArrayFromImage(mask).astype(np.uint8)
        ys, xs = np.where(arr > 0)
        if xs.size == 0 or ys.size == 0:
            return float(full_extent[0]), float(full_extent[1])

        extent_x = max(1, int(xs.max() - xs.min() + 1)) * spacing[0]
        extent_y = max(1, int(ys.max() - ys.min() + 1)) * spacing[1]
        return float(extent_x), float(extent_y)

    @staticmethod
    def _support_centroid_m(
        img: sitk.Image, mask: sitk.Image | None
    ) -> tuple[float, float]:
        """Estimate support centroid in physical coordinates (meters)."""
        spacing = np.array(img.GetSpacing(), dtype=float)
        origin = np.array(img.GetOrigin(), dtype=float)
        arr = sitk.GetArrayFromImage(img).astype(np.float64)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr -= float(np.min(arr))

        if mask is not None:
            m = sitk.GetArrayFromImage(mask).astype(np.float64)
            arr *= (m > 0).astype(np.float64)

        total = float(np.sum(arr))
        if total <= 0:
            size = np.array(img.GetSize(), dtype=float)
            center = origin + 0.5 * size * spacing
            return float(center[0]), float(center[1])

        yy, xx = np.indices(arr.shape)
        cx_idx = float(np.sum(xx * arr) / total)
        cy_idx = float(np.sum(yy * arr) / total)

        cx = origin[0] + (cx_idx + 0.5) * spacing[0]
        cy = origin[1] + (cy_idx + 0.5) * spacing[1]
        return float(cx), float(cy)

    @staticmethod
    def _support_eccentricity(mask: sitk.Image | None) -> float:
        """Approximate support eccentricity in [0,1), 0=circular, high=elongated."""
        if mask is None:
            return 1.0

        arr = sitk.GetArrayFromImage(mask).astype(np.uint8)
        ys, xs = np.where(arr > 0)
        if xs.size < 8 or ys.size < 8:
            return 1.0

        pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
        cov = np.cov(pts, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)
        lam_min = float(max(eigvals[0], 0.0))
        lam_max = float(max(eigvals[-1], 1e-12))
        ratio = min(max(lam_min / lam_max, 0.0), 1.0)
        return float(np.sqrt(max(0.0, 1.0 - ratio)))

    @staticmethod
    def _masked_intensity_moments(
        img: sitk.Image,
        mask: sitk.Image | None,
    ) -> tuple[tuple[float, float], float] | None:
        """
        Return (centroid_xy_m, principal_axis_angle_rad) from positive image weights
        restricted by `mask`. The principal axis is defined modulo pi.
        """
        arr = sitk.GetArrayFromImage(img).astype(np.float64)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        if mask is not None:
            mask_arr = sitk.GetArrayFromImage(mask).astype(np.uint8)
            if not np.any(mask_arr > 0):
                return None
            mask_bool = mask_arr > 0
            masked_values = arr[mask_bool]
            arr = np.where(mask_bool, arr - float(np.min(masked_values)), 0.0)
        else:
            arr = arr - float(np.min(arr))

        weights = np.clip(arr, a_min=0.0, a_max=None)
        total = float(np.sum(weights))
        if total <= 0.0:
            return None

        yy, xx = np.indices(weights.shape, dtype=np.float64)
        spacing = np.array(img.GetSpacing(), dtype=np.float64)
        origin = np.array(img.GetOrigin(), dtype=np.float64)

        x = origin[0] + (xx + 0.5) * spacing[0]
        y = origin[1] + (yy + 0.5) * spacing[1]

        cx = float(np.sum(x * weights) / total)
        cy = float(np.sum(y * weights) / total)

        dx = x - cx
        dy = y - cy
        cov_xx = float(np.sum(weights * dx * dx) / total)
        cov_xy = float(np.sum(weights * dx * dy) / total)
        cov_yy = float(np.sum(weights * dy * dy) / total)
        cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=np.float64)

        eigvals, eigvecs = np.linalg.eigh(cov)
        if eigvals[-1] <= 0.0:
            return None

        principal = eigvecs[:, int(np.argmax(eigvals))]
        angle = float(np.arctan2(principal[1], principal[0]))
        return ((cx, cy), angle)

    @staticmethod
    def _normalize_axial_angle(angle_rad: float) -> float:
        """Normalize an axis angle difference into [-pi/2, pi/2)."""
        return float(((angle_rad + 0.5 * np.pi) % np.pi) - 0.5 * np.pi)

    @staticmethod
    def _axial_symmetry_reliable(mask: sitk.Image | None, ecc: float) -> bool:
        """Return True when support is sufficiently dense/stable for 90° span reduction."""
        if mask is None:
            return False
        if ecc >= 0.35:
            return False

        arr = sitk.GetArrayFromImage(mask).astype(np.uint8)
        ys, xs = np.where(arr > 0)
        n = int(xs.size)
        if n < 64:
            return False

        w = int(xs.max() - xs.min() + 1)
        h = int(ys.max() - ys.min() + 1)
        if w < 8 or h < 8:
            return False

        fill_ratio = float(n) / float(max(w * h, 1))
        return fill_ratio >= 0.15

    @staticmethod
    def _fit_stage_eval_budget(
        *,
        stage: dict[str, float | int],
        transform_type: Transform2D,
        max_stage_evals: int,
    ) -> dict[str, float | int]:
        """Shrink tx/ty steps if needed so one stage stays within eval budget."""

        def _stage_evals(s: dict[str, float | int]) -> int:
            tx = int(s["tx_steps"])
            ty = int(s["ty_steps"])
            if transform_type == Transform2D.TRANSLATE:
                return (2 * tx + 1) * (2 * ty + 1)
            rot_step = max(float(s["rot_step_deg"]), 1e-6)
            rot_span = max(float(s["rot_span_deg"]), rot_step)
            rot_n = int(max(1, round(rot_span / rot_step)))
            return (2 * rot_n + 1) * (2 * tx + 1) * (2 * ty + 1)

        out = dict(stage)
        while _stage_evals(out) > max_stage_evals:
            tx = int(out["tx_steps"])
            ty = int(out["ty_steps"])
            if tx > 1 or ty > 1:
                out["tx_steps"] = max(1, int(np.ceil(tx / 2.0)))
                out["ty_steps"] = max(1, int(np.ceil(ty / 2.0)))
                continue

            if transform_type == Transform2D.RIGID:
                denom = (2 * tx + 1) * (2 * ty + 1)
                max_rot_n = int((max_stage_evals // max(denom, 1) - 1) // 2)
                max_rot_n = max(1, max_rot_n)

                rot_span = max(float(out["rot_span_deg"]), 1e-6)
                # Coarsen angular sampling until the stage fits budget.
                out["rot_step_deg"] = max(
                    float(out["rot_step_deg"]), rot_span / max_rot_n
                )
            break
        return out

    @staticmethod
    def _log_image_geometry(name: str, img: sitk.Image) -> None:
        """Log image geometry for debugging."""
        log = logging.getLogger("Rigid2DRegistration")
        size = img.GetSize()
        spacing = img.GetSpacing()
        origin = img.GetOrigin()
        extent_x = size[0] * spacing[0]
        extent_y = size[1] * spacing[1]
        log.debug(
            "%s: size=%s, spacing=[%.6f, %.6f] mm/px, origin=[%.6f, %.6f], "
            "extent=[%.3f x %.3f] mm",
            name,
            size,
            spacing[0] * 1000,
            spacing[1] * 1000,
            origin[0],
            origin[1],
            extent_x,
            extent_y,
        )

    @staticmethod
    def _mask_to_debug_string(mask: sitk.Image | None) -> str:
        if mask is None:
            return "none"

        arr = sitk.GetArrayFromImage(mask).astype(np.uint8)
        active = int(np.count_nonzero(arr > 0))
        total = int(arr.size)
        frac = (active / total) if total > 0 else 0.0
        return f"size={mask.GetSize()}, active={active}/{total} ({frac:.4f})"

    @staticmethod
    def _empty_mask_to_none(mask: sitk.Image | None) -> sitk.Image | None:
        if mask is None:
            return None
        arr = sitk.GetArrayViewFromImage(mask)
        if not np.any(arr > 0):
            return None
        return mask

    def _log_mask_stats(self, name: str, mask: sitk.Image) -> None:
        arr = sitk.GetArrayFromImage(mask).astype(np.uint8)
        ys, xs = np.where(arr > 0)
        active = int(xs.size)
        total = int(arr.size)
        frac = (active / total) if total > 0 else 0.0

        if active == 0:
            self._log.debug(
                "%s: size=%s, active=0/%d (0.0000), bbox=empty",
                name,
                mask.GetSize(),
                total,
            )
            return

        xmin, xmax = int(xs.min()), int(xs.max())
        ymin, ymax = int(ys.min()), int(ys.max())
        self._log.debug(
            "%s: size=%s, active=%d/%d (%.4f), bbox_px=[x:%d..%d, y:%d..%d]",
            name,
            mask.GetSize(),
            active,
            total,
            frac,
            xmin,
            xmax,
            ymin,
            ymax,
        )

    @staticmethod
    def _transform_to_debug_string(transform: sitk.Transform) -> str:
        name = transform.GetName()
        params = tuple(transform.GetParameters())
        fixed = tuple(transform.GetFixedParameters())
        return f"{name}(params={params}, fixed={fixed})"

    @staticmethod
    def _clone_transform(transform: sitk.Transform) -> sitk.Transform:
        name = transform.GetName()
        if name == "Euler2DTransform":
            return sitk.Euler2DTransform(transform)
        if name == "TranslationTransform":
            return sitk.TranslationTransform(transform)
        return sitk.Transform(transform)

    def _log_transform_init(self, context: str, transform: sitk.Transform) -> None:
        self._log.debug(
            "%s init: %s", context, self._transform_to_debug_string(transform)
        )

    def _log_transform_result(self, context: str, transform: sitk.Transform) -> None:
        self._log.debug(
            "%s transform: %s", context, self._transform_to_debug_string(transform)
        )

    def _moments_init(
        self,
        fixed: sitk.Image,
        moving: sitk.Image,
        *,
        fixed_mask: sitk.Image | None = None,
        moving_mask: sitk.Image | None = None,
    ) -> sitk.Transform:
        """
        Initialize transform by aligning image centers.
        Tries multiple strategies: moments → geometry-based → identity fallback.
        """
        try:
            fixed_moments = (
                self._masked_intensity_moments(fixed, fixed_mask)
                if fixed_mask is not None and moving_mask is not None
                else None
            )
            moving_moments = (
                self._masked_intensity_moments(moving, moving_mask)
                if fixed_mask is not None and moving_mask is not None
                else None
            )
            if fixed_moments is not None and moving_moments is not None:
                fixed_center, fixed_angle = fixed_moments
                moving_center, moving_angle = moving_moments
                translation = (
                    float(moving_center[0] - fixed_center[0]),
                    float(moving_center[1] - fixed_center[1]),
                )

                if self.transform_type == Transform2D.TRANSLATE:
                    translate = sitk.TranslationTransform(2)
                    translate.SetOffset(translation)
                    self._log.debug(
                        "Using mask-aware moments initialization (TRANSLATE)"
                    )
                    return translate

                if self.transform_type == Transform2D.RIGID:
                    euler = sitk.Euler2DTransform()
                    euler.SetCenter((float(fixed_center[0]), float(fixed_center[1])))
                    euler.SetAngle(
                        self._normalize_axial_angle(moving_angle - fixed_angle)
                    )
                    euler.SetTranslation(translation)
                    self._log.debug("Using mask-aware moments initialization (RIGID)")
                    return euler

            # Try moments-based initialization first
            if self.transform_type == Transform2D.TRANSLATE:
                rigid_seed = sitk.CenteredTransformInitializer(
                    fixed,
                    moving,
                    sitk.Euler2DTransform(),
                    sitk.CenteredTransformInitializerFilter.MOMENTS,
                )
                translate = sitk.TranslationTransform(2)
                translate.SetOffset(rigid_seed.GetTranslation())
                self._log.debug("Using moments-based initialization (TRANSLATE)")
                return translate

            if self.transform_type == Transform2D.RIGID:
                result = sitk.CenteredTransformInitializer(
                    fixed,
                    moving,
                    sitk.Euler2DTransform(),
                    sitk.CenteredTransformInitializerFilter.MOMENTS,
                )
                self._log.debug("Using moments-based initialization (RIGID)")
                return result
        except Exception as e:
            self._log.debug(
                "Moments initialization failed (%s), trying geometry-based fallback",
                str(e),
            )

        # Fallback 1: align image centers based on geometry
        try:
            result = self._geometry_based_init(
                fixed,
                moving,
                fixed_mask=fixed_mask,
                moving_mask=moving_mask,
            )
            self._log.debug("Using geometry-based initialization")
            return result
        except Exception as e:
            self._log.debug(
                "Geometry initialization failed (%s), using identity transform",
                str(e),
            )

        # Fallback 2: identity transform - exhaustive search will find alignment
        if self.transform_type == Transform2D.TRANSLATE:
            return sitk.TranslationTransform(2)
        elif self.transform_type == Transform2D.RIGID:
            return sitk.Euler2DTransform()
        else:
            raise ValueError(f"Unsupported transform_type: {self.transform_type}")

    def _geometry_based_init(
        self,
        fixed: sitk.Image,
        moving: sitk.Image,
        *,
        fixed_mask: sitk.Image | None = None,
        moving_mask: sitk.Image | None = None,
    ) -> sitk.Transform:
        """
        Initialize transform by aligning image center-of-geometry.
        This is more robust than moments when data has anomalies.
        """
        if fixed_mask is not None and moving_mask is not None:
            fixed_center = np.array(
                self._support_centroid_m(fixed, fixed_mask), dtype=float
            )
        else:
            fixed_size = np.array(fixed.GetSize(), dtype=float)
            fixed_spacing = np.array(fixed.GetSpacing(), dtype=float)
            fixed_origin = np.array(fixed.GetOrigin(), dtype=float)
            fixed_center = fixed_origin + 0.5 * fixed_size * fixed_spacing

        if fixed_mask is not None and moving_mask is not None:
            moving_center = np.array(
                self._support_centroid_m(moving, moving_mask), dtype=float
            )
        else:
            moving_size = np.array(moving.GetSize(), dtype=float)
            moving_spacing = np.array(moving.GetSpacing(), dtype=float)
            moving_origin = np.array(moving.GetOrigin(), dtype=float)
            moving_center = moving_origin + 0.5 * moving_size * moving_spacing

        # Registration uses fixed-space sample points mapped into moving space.
        translation = moving_center - fixed_center

        if self.transform_type == Transform2D.TRANSLATE:
            tx = sitk.TranslationTransform(2)
            tx.SetOffset((float(translation[0]), float(translation[1])))
            return tx

        if self.transform_type == Transform2D.RIGID:
            # Create Euler2D with center-aligned translation, no rotation
            euler = sitk.Euler2DTransform()
            euler.SetTranslation((float(translation[0]), float(translation[1])))
            # Center of rotation follows the fixed support when masks are available.
            euler.SetCenter((float(fixed_center[0]), float(fixed_center[1])))
            return euler

        raise ValueError(f"Unsupported transform_type: {self.transform_type}")

    def _build_method(
        self,
        init_transform: sitk.Transform,
        stage: dict,
        fixed_mask: sitk.Image | None,
        moving_mask: sitk.Image | None,
    ) -> sitk.ImageRegistrationMethod:
        t_step = float(stage["translation_step"])
        tx_n = int(stage["tx_steps"])
        ty_n = int(stage["ty_steps"])
        rot_step = float(stage.get("rot_step_deg", 0.0))
        rot_span = float(stage.get("rot_span_deg", 0.0))

        method = sitk.ImageRegistrationMethod()
        method.SetMetricAsMattesMutualInformation()
        method.SetMetricSamplingStrategy(sitk.ImageRegistrationMethod.RANDOM)
        method.SetMetricSamplingPercentage(0.3, 42)

        if fixed_mask is not None:
            method.SetMetricFixedMask(fixed_mask)
        if moving_mask is not None:
            method.SetMetricMovingMask(moving_mask)

        shrink_factors, smoothing_sigmas = self._multiresolution_schedule(
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
        )
        method.SetShrinkFactorsPerLevel(shrink_factors)
        method.SetSmoothingSigmasPerLevel(smoothing_sigmas)
        method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()

        if self.transform_type == Transform2D.TRANSLATE:
            method.SetOptimizerAsExhaustive([tx_n, ty_n])
            method.SetOptimizerScales([t_step, t_step])
        elif self.transform_type == Transform2D.RIGID:
            if rot_step <= 0:
                raise ValueError("rot_step_deg must be > 0 for rigid.")
            rot_n = int(max(1, round(rot_span / rot_step)))
            method.SetOptimizerAsExhaustive([rot_n, tx_n, ty_n])
            method.SetOptimizerScales([np.deg2rad(max(rot_step, 1e-6)), t_step, t_step])
        else:
            raise ValueError(f"Unsupported transform_type: {self.transform_type}")

        method.SetInitialTransform(init_transform, inPlace=True)
        method.SetInterpolator(sitk.sitkLinear)
        return method

    @staticmethod
    def _expand_to_union(
        fixed: sitk.Image,
        moving: sitk.Image,
        default_value: float,
        interpolator: int,
        padding_x: float = 0.0,
        padding_y: float = 0.0,
    ) -> sitk.Image:
        """
        Resample `moving` onto a grid that covers the union of both images' physical
        extents using fixed's spacing, optionally padded on each axis. The result is
        always at least as large as `fixed`.
        """
        fixed_origin = np.array(fixed.GetOrigin(), dtype=float)
        moving_origin = np.array(moving.GetOrigin(), dtype=float)
        fixed_spacing = np.array(fixed.GetSpacing(), dtype=float)
        fixed_end = (
            fixed_origin + np.array(fixed.GetSize(), dtype=float) * fixed_spacing
        )
        moving_end = moving_origin + np.array(moving.GetSize(), dtype=float) * np.array(
            moving.GetSpacing(), dtype=float
        )

        pad = np.array([padding_x, padding_y], dtype=float)
        union_origin = np.minimum(fixed_origin, moving_origin) - pad
        union_end = np.maximum(fixed_end, moving_end) + pad
        union_size = np.ceil((union_end - union_origin) / fixed_spacing).astype(int)
        union_size = np.maximum(union_size, 1)

        union_ref = sitk.Image(int(union_size[0]), int(union_size[1]), sitk.sitkFloat32)
        union_ref.SetSpacing([float(fixed_spacing[0]), float(fixed_spacing[1])])
        union_ref.SetOrigin([float(union_origin[0]), float(union_origin[1])])

        identity = sitk.Transform(2, sitk.sitkIdentity)
        return sitk.Resample(
            moving,
            union_ref,
            identity,
            interpolator,
            default_value,
        )

    @staticmethod
    def _compute_union_padding(
        fixed: sitk.Image,
        moving: sitk.Image,
        effective_stages: list[dict],
    ) -> tuple[float, float]:
        """
        Compute per-axis padding (m) to add beyond the union extent so that
        the exhaustive optimizer never escapes the moving buffer.

        Three additive contributions per axis:
        1.  Circumradius excess:  sqrt(half_x² + half_y²) − half_fixed_axis
            A rotated fixed-corner sample reaches at most the circumradius from
            the fixed image centre; this covers the extra reach vs a pure translation.
        2.  Max translation search range: max(N × t_step) across all effective stages.
        3.  Image-centre distance |c_fixed − c_moving|: the geometry-based /
            moments init translates fixed samples by this offset before the
            exhaustive delta is added.
        Plus one pixel safety margin.
        """
        fixed_spacing = np.array(fixed.GetSpacing(), dtype=float)
        fixed_half = 0.5 * np.array(fixed.GetSize(), dtype=float) * fixed_spacing
        moving_half = (
            0.5
            * np.array(moving.GetSize(), dtype=float)
            * np.array(moving.GetSpacing(), dtype=float)
        )
        c_fixed = np.array(fixed.GetOrigin(), dtype=float) + fixed_half
        c_moving = np.array(moving.GetOrigin(), dtype=float) + moving_half
        center_dist = np.abs(c_fixed - c_moving)

        circumradius = float(np.sqrt(np.sum(fixed_half**2)))

        max_tx = max(
            (s["tx_steps"] * s["translation_step"] for s in effective_stages),
            default=0.0,
        )
        max_ty = max(
            (s["ty_steps"] * s["translation_step"] for s in effective_stages),
            default=0.0,
        )

        pad_x = (
            circumradius - fixed_half[0] + center_dist[0] + max_tx + fixed_spacing[0]
        )
        pad_y = (
            circumradius - fixed_half[1] + center_dist[1] + max_ty + fixed_spacing[1]
        )
        return float(max(pad_x, 0.0)), float(max(pad_y, 0.0))

    @staticmethod
    def _clamp_stage_steps(
        stage: dict,
        fixed: sitk.Image,
        moving: sitk.Image,
        log: logging.Logger,
    ) -> dict:
        """
        Cap tx_steps / ty_steps so the search range never exceeds the smaller of the
        two images' physical extents on each axis.  Searching beyond that explores
        zero-overlap configurations and wastes compute.  User presets are treated as
        ceilings; steps are never increased.
        """
        t_step = float(stage["translation_step"])
        if t_step <= 0:
            return stage

        fixed_spacing = np.array(fixed.GetSpacing(), dtype=float)
        moving_spacing = np.array(moving.GetSpacing(), dtype=float)
        fixed_extent = np.array(fixed.GetSize(), dtype=float) * fixed_spacing
        moving_extent = np.array(moving.GetSize(), dtype=float) * moving_spacing
        min_extent = np.minimum(fixed_extent, moving_extent)  # [x, y]

        max_steps_x = max(1, int(min_extent[0] / (2.0 * t_step)))
        max_steps_y = max(1, int(min_extent[1] / (2.0 * t_step)))

        orig_tx = int(stage["tx_steps"])
        orig_ty = int(stage["ty_steps"])
        new_tx = min(orig_tx, max_steps_x)
        new_ty = min(orig_ty, max_steps_y)

        if new_tx < orig_tx or new_ty < orig_ty:
            log.warning(
                "Clamping translation steps from (±%d, ±%d) to (±%d, ±%d) "
                "to stay within image extents (t_step=%.4f m, "
                "fixed=[%.1f x %.1f] mm, moving=[%.1f x %.1f] mm)",
                orig_tx,
                orig_ty,
                new_tx,
                new_ty,
                t_step,
                fixed_extent[0] * 1000,
                fixed_extent[1] * 1000,
                moving_extent[0] * 1000,
                moving_extent[1] * 1000,
            )

        return {**stage, "tx_steps": new_tx, "ty_steps": new_ty}

    @staticmethod
    def _sanitize_spacing(img: sitk.Image) -> sitk.Image:
        spacing = [
            max(float(s), 1e-6) if np.isfinite(s) else 1e-6 for s in img.GetSpacing()
        ]
        out = sitk.Image(img)
        out.CopyInformation(img)
        out.SetSpacing(spacing)
        return out

    @staticmethod
    def _resample(
        moving: sitk.Image, fixed: sitk.Image, transform: sitk.Transform
    ) -> sitk.Image:
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        resampler.SetTransform(transform)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(float(np.nanmin(sitk.GetArrayFromImage(fixed))))
        return resampler.Execute(moving)

    def _multiresolution_schedule(
        self,
        *,
        fixed_mask: sitk.Image | None,
        moving_mask: sitk.Image | None,
    ) -> tuple[list[int], list[float]]:
        min_dim = min(*self.fixed_image.GetSize(), *self.moving_image.GetSize())
        for mask in (fixed_mask, moving_mask):
            if mask is None:
                continue
            mask_min_dim = self._mask_min_dimension(mask)
            if mask_min_dim is not None:
                min_dim = min(min_dim, mask_min_dim)

        if min_dim >= 16:
            return [4, 2, 1], [2.0, 1.0, 0.0]
        if min_dim >= 8:
            return [2, 1], [1.0, 0.0]
        return [1], [0.0]

    @staticmethod
    def _mask_min_dimension(mask: sitk.Image) -> int | None:
        arr = sitk.GetArrayFromImage(mask).astype(np.uint8)
        ys, xs = np.where(arr > 0)
        if xs.size == 0 or ys.size == 0:
            return None
        return int(min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))

    def _stage_str(self, index: int, total: int, stage: dict, *, mask_on: bool) -> str:
        rot_step = stage.get("rot_step_deg")
        rot_span = stage.get("rot_span_deg")

        rot_part = ""
        if rot_step is not None and rot_span is not None and float(rot_step) > 0:
            rot_part = f", rot_step={rot_step}°, span={rot_span}°"

        mask_label = "mask ON" if mask_on else "mask OFF"

        return (
            f"[Stage {index}/{total}] "
            f"t_step={stage['translation_step']}, steps=±({stage['tx_steps']},{stage['ty_steps']})"
            f"{rot_part} ({mask_label})"
        )

    @staticmethod
    def db_to_linear(img_db: sitk.Image, floor_norm: float) -> sitk.Image:
        floor_db = float(floor_norm)
        arr = sitk.GetArrayFromImage(img_db).astype(np.float32)
        arr = np.nan_to_num(arr, nan=floor_db, posinf=floor_db, neginf=floor_db)
        arr[arr < floor_db] = floor_db
        lin = (10.0 ** (arr / 10.0)).astype(np.float32)
        out = sitk.GetImageFromArray(lin)
        out.CopyInformation(img_db)
        return out
