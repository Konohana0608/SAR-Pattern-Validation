from pathlib import Path

import pytest

from sar_pattern_validation.report import (
    DEFAULT_TEMPLATE_DIR,
    TEMPLATE_FIGURE_MAPPING,
    _set_latex_macro,
    generate_report,
)
from sar_pattern_validation.workflow_config import WorkflowConfig
from sar_pattern_validation.workflows import WorkflowResult


def _make_workflow_result(
    *,
    pass_rate_percent: float = 100.0,
    gamma_image: Path | None = None,
    failure_image: Path | None = None,
    overlay: Path | None = None,
    measured_image: Path | None = None,
    reference_image: Path | None = None,
) -> WorkflowResult:
    return WorkflowResult(
        pass_rate_percent=pass_rate_percent,
        evaluated_pixel_count=1000,
        passed_pixel_count=int(10 * pass_rate_percent),
        failed_pixel_count=1000 - int(10 * pass_rate_percent),
        gamma_image_path=gamma_image,
        failure_image_path=failure_image,
        registered_overlay_path=overlay,
        loaded_images_path=None,
        reference_image_path=reference_image,
        measured_image_path=measured_image,
        aligned_measured_path=None,
        measured_pssar=24.63,
        reference_pssar=24.71,
        scaling_error=-0.0035,  # -0.35 %
        dose_to_agreement=5.0,
        distance_to_agreement=2.0,
        gamma_map=None,
        evaluation_mask=None,
    )


def test_set_latex_macro_replaces_newcommand_body():
    text = r"\newcommand{\powerlevel}{16.5} % dBm"
    out = _set_latex_macro(text, "powerlevel", "10")
    assert r"\newcommand{\powerlevel}{10}" in out


def test_set_latex_macro_replaces_def_body():
    text = r"\def\passrate{100.0}"
    out = _set_latex_macro(text, "passrate", "97.5")
    assert r"\def\passrate{97.5}" in out


def test_set_latex_macro_no_match_keeps_text_unchanged():
    text = r"\newcommand{\foo}{bar}"
    out = _set_latex_macro(text, "missing", "x")
    assert out == text


def test_default_template_dir_resolves_to_repo_template():
    assert DEFAULT_TEMPLATE_DIR.name == "report_template"
    assert (DEFAULT_TEMPLATE_DIR / "main.tex").is_file()


def test_template_figure_mapping_keys_match_template():
    template_text = (DEFAULT_TEMPLATE_DIR / "main.tex").read_text(encoding="utf-8")
    for figure_name in TEMPLATE_FIGURE_MAPPING:
        assert f"figures/{figure_name}" in template_text, (
            f"Template no longer references {figure_name!r}; update the mapping."
        )


def test_generate_report_writes_filled_tex_with_all_substitutions(tmp_path: Path):
    config = WorkflowConfig(
        measured_file_path="data/measurements/D900_Flat_HSL_15mm_10dBm_10g_3.csv",
        reference_file_path="data/database/dipole_900MHz_Flat_15mm_10g.csv",
        power_level_dbm=10.0,
        noise_floor=0.001,
    )
    result = _make_workflow_result(pass_rate_percent=97.5)

    out = generate_report(
        workflow_result=result,
        workflow_config=config,
        output_dir=tmp_path,
        antenna_type="dipole",
        frequency_mhz=900,
        distance_mm=15,
        mass_g=10,
    )

    assert out == tmp_path / "main.tex"
    text = out.read_text(encoding="utf-8")
    assert r"\newcommand{\filemeas}{D900\_Flat\_HSL\_15mm\_10dBm\_10g\_3.csv}" in text
    assert r"\newcommand{\powerlevel}{10}" in text
    assert r"\newcommand{\noiselevel}{0.001}" in text
    assert r"\newcommand{\antennatype}{dipole}" in text
    assert r"\newcommand{\frequency}{900}" in text
    assert r"\newcommand{\distance}{15}" in text
    assert r"\newcommand{\mass}{10}" in text
    assert r"\newcommand{\pssarref}{24.71}" in text
    assert r"\newcommand{\pssarmeas}{24.63}" in text
    assert r"\newcommand{\errscale}{-0.35}" in text
    assert r"\newcommand{\deltadist}{2~mm\xspace}" in text
    assert r"\newcommand{\deltadose}{5~\%\xspace}" in text
    assert r"\def\passrate{97.5}" in text


def test_generate_report_copies_existing_figures_with_template_filenames(
    tmp_path: Path,
):
    plot_dir = tmp_path / "plots"
    plot_dir.mkdir()
    gamma = plot_dir / "gamma.png"
    failure = plot_dir / "failure.png"
    overlay = plot_dir / "overlay.png"
    measured_img = plot_dir / "measured.png"
    reference_img = plot_dir / "reference.png"
    for path in (gamma, failure, overlay, measured_img, reference_img):
        path.write_bytes(b"\x89PNG fake")

    config = WorkflowConfig(
        measured_file_path="data/example/measured_sSAR1g.csv",
        reference_file_path="data/example/reference_sSAR1g.csv",
    )
    result = _make_workflow_result(
        gamma_image=gamma,
        failure_image=failure,
        overlay=overlay,
        measured_image=measured_img,
        reference_image=reference_img,
    )

    out_dir = tmp_path / "report"
    generate_report(
        workflow_result=result,
        workflow_config=config,
        output_dir=out_dir,
    )

    figures_dir = out_dir / "figures"
    for figure_name in TEMPLATE_FIGURE_MAPPING:
        assert (figures_dir / figure_name).is_file(), (
            f"Missing template figure {figure_name!r} in report output"
        )


def test_generate_report_skips_missing_figures_without_failing(tmp_path: Path):
    config = WorkflowConfig(
        measured_file_path="data/example/measured_sSAR1g.csv",
        reference_file_path="data/example/reference_sSAR1g.csv",
    )
    result = _make_workflow_result()
    # All figure paths are None → none of the template figures get copied,
    # but the .tex still renders so the user can see the parameter table.
    out = generate_report(
        workflow_result=result,
        workflow_config=config,
        output_dir=tmp_path,
    )

    assert out.is_file()
    figures_dir = tmp_path / "figures"
    assert figures_dir.is_dir()
    assert list(figures_dir.iterdir()) == []


def test_generate_report_raises_when_template_missing(tmp_path: Path):
    missing_template = tmp_path / "no_template_here"
    missing_template.mkdir()

    config = WorkflowConfig(
        measured_file_path="data/example/measured_sSAR1g.csv",
        reference_file_path="data/example/reference_sSAR1g.csv",
    )
    result = _make_workflow_result()

    with pytest.raises(FileNotFoundError):
        generate_report(
            workflow_result=result,
            workflow_config=config,
            output_dir=tmp_path / "out",
            template_dir=missing_template,
        )
