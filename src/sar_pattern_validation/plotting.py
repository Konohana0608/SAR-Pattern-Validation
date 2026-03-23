from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure

from sar_pattern_validation.utils import image_extent_mm as _shared_image_extent_mm
from sar_pattern_validation.workflow_config import PlottingConfig


def image_extent_mm(img: sitk.Image) -> tuple[float, float, float, float]:
    extent = _shared_image_extent_mm(img)
    return (extent[0], extent[1], extent[2], extent[3])


def extent_mm_from_axes(
    x_axis_m: np.ndarray, y_axis_m: np.ndarray
) -> tuple[float, float, float, float]:
    dx = float(np.median(np.diff(x_axis_m))) if len(x_axis_m) > 1 else 0.0
    dy = float(np.median(np.diff(y_axis_m))) if len(y_axis_m) > 1 else 0.0
    return (
        1000 * (x_axis_m.min() - 0.5 * dx),
        1000 * (x_axis_m.max() + 0.5 * dx),
        1000 * (y_axis_m.min() - 0.5 * dy),
        1000 * (y_axis_m.max() + 0.5 * dy),
    )


def show_registration_overlay(
    fixed_image: sitk.Image,
    aligned_moving_image: sitk.Image,
    *,
    title: str = "Rigid Registration Overlay",
    image_save_path: Path | None = None,
    plotting_config: PlottingConfig | None = None,
) -> None:
    config = plotting_config or PlottingConfig()
    fixed_u8 = sitk.GetArrayFromImage(sitk.RescaleIntensity(fixed_image)).astype(
        np.uint8
    )
    aligned_u8 = sitk.GetArrayFromImage(
        sitk.RescaleIntensity(aligned_moving_image)
    ).astype(np.uint8)
    if fixed_u8.shape != aligned_u8.shape:
        raise ValueError("Fixed and aligned images must have the same shape.")

    overlay = np.zeros((*fixed_u8.shape, 3), dtype=np.uint8)
    overlay[..., 0] = fixed_u8
    overlay[..., 2] = aligned_u8

    with _style_context(config):
        fig, ax = plt.subplots(figsize=config.single_figure_size)
        fig.set_facecolor(config.figure_facecolor)
        ax.set_facecolor(config.dark_axes_facecolor)
        ax.tick_params(axis="both", which="major")
        ax.imshow(
            overlay,
            extent=image_extent_mm(fixed_image),
            origin="lower",
            aspect="equal",
        )
        ax.set_title(_two_line_title(title))
        ax.set(
            xlim=(config.window_mm[0], config.window_mm[1]),
            ylim=(config.window_mm[2], config.window_mm[3]),
        )
        ax.set_xlabel("$x'_e$ (mm)")
        ax.set_ylabel("$y'_e$ (mm)")

        patches = [
            mpatches.Patch(color="red", label="Reference"),
            mpatches.Patch(color="blue", label="Measured"),
            mpatches.Patch(color="magenta", label="Overlap"),
        ]
        legend = ax.legend(handles=patches, loc="lower right", frameon=True)
        legend.get_frame().set_facecolor(config.dark_axes_facecolor)
        legend.get_frame().set_edgecolor("w")
        for text in legend.get_texts():
            text.set_color("w")

        fig.tight_layout()
        _save_or_show(fig, image_save_path, config)


def plot_loaded_images(
    *,
    measured_image: np.ndarray,
    reference_image: np.ndarray,
    measured_axes: tuple[np.ndarray, np.ndarray],
    reference_axes: tuple[np.ndarray, np.ndarray],
    image_save_path: Path | None = None,
    plotting_config: PlottingConfig | None = None,
) -> None:
    config = plotting_config or PlottingConfig()

    with _style_context(config):
        fig, (ax0, ax1) = plt.subplots(
            1,
            2,
            figsize=config.combined_figure_size,
            facecolor=config.figure_facecolor,
        )
        common = dict(origin="lower", aspect="equal", cmap="hot")

        sar_im = None
        for ax, data, axes, label in [
            (ax0, measured_image, measured_axes, "Measured"),
            (ax1, reference_image, reference_axes, "Reference"),
        ]:
            im = ax.imshow(data, extent=extent_mm_from_axes(*axes), **common)
            ax.set_title(_two_line_title(f"{label} SAR (linear, unit-peak)"))
            ax.set_facecolor(config.dark_axes_facecolor)
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)")
            if sar_im is None:
                sar_im = im

        if image_save_path is not None and sar_im is not None:
            colorbar_path = _derive_colorbar_path(image_save_path, "sar_colorbar")
            _save_colorbar_only(
                sar_im,
                colorbar_path,
                config,
                label="Normalized SAR",
                orientation="vertical",
            )

        fig.tight_layout()
        _save_or_show(fig, image_save_path, config)


def plot_sar_image(
    *,
    sar_image: np.ndarray,
    x_axis_m: np.ndarray,
    y_axis_m: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: Path | None = None,
    show_colorbar: bool,
    plotting_config: PlottingConfig | None = None,
) -> None:
    config = plotting_config or PlottingConfig()

    with _style_context(config):
        fig, ax = plt.subplots(figsize=config.single_figure_size)
        fig.set_facecolor(config.figure_facecolor)
        ax.set_facecolor(config.dark_axes_facecolor)
        im = ax.imshow(
            sar_image,
            cmap="hot",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            extent=extent_mm_from_axes(x_axis_m, y_axis_m),
            origin="lower",
            aspect="equal",
        )
        ax.set_title(_two_line_title(title))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(config.window_mm[0], config.window_mm[1])
        ax.set_ylim(config.window_mm[2], config.window_mm[3])

        if show_colorbar and save_path is not None:
            colorbar_path = _derive_colorbar_path(save_path, "colorbar")
            _save_colorbar_only(
                im,
                colorbar_path,
                config,
                label="Normalized SAR",
                orientation="vertical",
            )
        fig.tight_layout()

        _save_or_show(fig, save_path, config)


def plot_gamma_results(
    *,
    gamma_map: np.ndarray,
    evaluation_mask: np.ndarray,
    gamma_cap: float,
    extent_mm: tuple[float, float, float, float],
    gamma_image_save_path: Path | None = None,
    failure_image_save_path: Path | None = None,
    plotting_config: PlottingConfig | None = None,
) -> None:
    config = plotting_config or PlottingConfig()
    masked_gamma = np.ma.masked_invalid(gamma_map)
    failures = (gamma_map > 1.0) & evaluation_mask

    with _style_context(config):
        fig, ax = plt.subplots(figsize=config.single_figure_size)
        fig.set_facecolor(config.figure_facecolor)
        ax.set_facecolor(config.light_axes_facecolor)

        cmap = mpl.colormaps["plasma"].copy()
        cmap.set_bad(config.light_axes_facecolor)

        im = ax.imshow(
            masked_gamma,
            cmap=cmap,
            vmin=0,
            vmax=gamma_cap,
            interpolation="nearest",
            extent=extent_mm,
            origin="lower",
            aspect="equal",
        )
        ax.set_title(_two_line_title("Gamma Index"))
        ax.set_xlim(config.window_mm[0], config.window_mm[1])
        ax.set_ylim(config.window_mm[2], config.window_mm[3])
        ax.set_xlabel("$x'_e$ (mm)")
        ax.set_ylabel("$y'_e$ (mm)")
        fig.tight_layout()
        if gamma_image_save_path is not None:
            colorbar_path_v = _derive_colorbar_path(
                gamma_image_save_path, "colorbar_vertical"
            )
            colorbar_path_h = _derive_colorbar_path(
                gamma_image_save_path, "colorbar_horizontal"
            )
            _save_colorbar_only(
                im,
                colorbar_path_v,
                config,
                label="Gamma Index",
                orientation="vertical",
            )
            _save_colorbar_only(
                im,
                colorbar_path_h,
                config,
                label="Gamma Index",
                orientation="horizontal",
            )
        _save_or_show(fig, gamma_image_save_path, config)

    with _style_context(config):
        fig, ax = plt.subplots(figsize=config.single_figure_size)
        fig.set_facecolor(config.figure_facecolor)
        ax.set_facecolor(config.light_axes_facecolor)
        ax.imshow(
            evaluation_mask.astype(float),
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="nearest",
            extent=extent_mm,
            origin="lower",
            aspect="equal",
        )
        ax.imshow(
            failures.astype(float),
            cmap="Reds",
            interpolation="nearest",
            alpha=0.85,
            extent=extent_mm,
            origin="lower",
            aspect="equal",
        )
        ax.set_title(_two_line_title("Gamma Pass / Fail Map"))
        ax.set_xlim(config.window_mm[0], config.window_mm[1])
        ax.set_ylim(config.window_mm[2], config.window_mm[3])
        ax.set_xlabel("$x'_e$ (mm)")
        ax.set_ylabel("$y'_e$ (mm)")
        ax.legend(
            handles=[
                mpatches.Patch(
                    facecolor=mpl.colormaps["gray"](0.85), edgecolor="k", label="Pass"
                ),
                mpatches.Patch(
                    facecolor=mpl.colormaps["Reds"](0.8), edgecolor="k", label="Fail"
                ),
            ],
            loc="lower right",
        )
        fig.tight_layout()
        _save_or_show(fig, failure_image_save_path, config)


def _style_context(config: PlottingConfig):
    return mpl.rc_context({"font.size": config.font_size})


def _add_matched_colorbar(
    im,
    ax,
    config: PlottingConfig,
    *,
    label: str,
    width: float = 0.03,
    pad: float = 0.01,
):
    fig = ax.figure
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + pad, pos.y0, width, pos.height])
    return fig.colorbar(im, cax=cax, label=label)


def _save_or_show(fig: Figure, save_path: Path | None, config: PlottingConfig) -> None:
    if save_path is not None:
        fig.savefig(save_path, dpi=config.save_dpi)
        plt.close(fig)
        return
    plt.show()


def _derive_colorbar_path(save_path: Path, suffix: str) -> Path:
    return save_path.with_name(f"{save_path.stem}_{suffix}{save_path.suffix}")


def _save_colorbar_only(
    im,
    save_path: Path,
    config: PlottingConfig,
    *,
    label: str,
    orientation: str = "vertical",
) -> None:
    if orientation == "vertical":
        figsize = (0.8, 4.0)
        cax_bounds: tuple[float, float, float, float] = (0.15, 0.1, 0.1, 0.8)
    else:
        figsize = (4.0, 0.8)
        cax_bounds = (0.1, 0.15, 0.8, 0.1)

    fig = plt.figure(figsize=figsize)
    fig.set_facecolor(config.figure_facecolor)
    cax = fig.add_axes(cax_bounds)
    sm = ScalarMappable(norm=im.norm, cmap=im.cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation=orientation)
    cbar.set_label(label)
    fig.savefig(save_path, dpi=config.save_dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _two_line_title(title: str, max_chars_per_line: int = 32) -> str:
    if len(title) <= max_chars_per_line:
        return title

    wrapped = textwrap.wrap(title, width=max_chars_per_line)
    if len(wrapped) <= 2:
        return "\n".join(wrapped)

    return wrapped[0] + "\n" + " ".join(wrapped[1:])
