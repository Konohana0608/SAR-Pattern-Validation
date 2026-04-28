"""
Generate SAR Pattern Validation reports from workflow results.

Per MGD 2026-04-24 feedback (slide 10): produce a report from each pipeline
run using MGD's LaTeX template (report_template/main.tex), auto-filled with
generated plots and data (gamma map, scaling error, pass/fail, parameters).

Output path is exposed for [[Task 6.10 - User Report Download Button]] (MEST).
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from sar_pattern_validation.workflow_config import WorkflowConfig
from sar_pattern_validation.workflows import WorkflowResult

DEFAULT_TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "report_template"

# Plot-file mapping: template figure name -> WorkflowResult attribute holding
# the source path. Order matters only for documentation; lookups are by key.
TEMPLATE_FIGURE_MAPPING: dict[str, str] = {
    "gamma_index_with_colorbar.png": "gamma_image_path",
    "gamma_failures.png": "failure_image_path",
    "registration_nocolorbar.png": "registered_overlay_path",
    "measured_with_colorbar.png": "measured_image_path",
    "reference_with_colorbar.png": "reference_image_path",
}


def _set_latex_macro(text: str, name: str, value: str) -> str:
    """
    Substitute the body of a LaTeX macro definition. Handles both
    `\\newcommand{\\NAME}{...}` and `\\def\\NAME{...}` (the template uses
    `\\def` for ``\\passrate`` because of the FPeval branching below it).
    """
    newcmd = re.compile(r"\\newcommand\{\\" + re.escape(name) + r"\}\{[^}]*\}")
    if newcmd.search(text):
        return newcmd.sub(lambda _m: f"\\newcommand{{\\{name}}}{{{value}}}", text)
    def_re = re.compile(r"\\def\\" + re.escape(name) + r"\{[^}]*\}")
    return def_re.sub(lambda _m: f"\\def\\{name}{{{value}}}", text)


def _latex_escape_filename(name: str) -> str:
    """Escape characters that LaTeX interprets specially in a typewritten filename."""
    return (
        name.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
        .replace("$", r"\$")
    )


def generate_report(
    *,
    workflow_result: WorkflowResult,
    workflow_config: WorkflowConfig,
    output_dir: str | Path,
    template_dir: str | Path = DEFAULT_TEMPLATE_DIR,
    antenna_type: str = "dipole",
    frequency_mhz: int = 0,
    distance_mm: int = 0,
    mass_g: int = 0,
) -> Path:
    """
    Render the SAR Pattern Validation LaTeX report into ``output_dir``.

    Reads ``template_dir/main.tex`` and substitutes the macros listed in
    ``TEMPLATE_FIGURE_MAPPING``'s sibling table below. Copies the generated
    plots into ``output_dir/figures/`` using the filenames the template
    expects.

    The .tex output can be compiled to PDF externally (``pdflatex main.tex``);
    PDF compilation is intentionally not handled here so the module has no
    LaTeX runtime dependency.

    Returns the path to the rendered ``main.tex``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    template_dir = Path(template_dir)
    main_tex_template = template_dir / "main.tex"
    if not main_tex_template.is_file():
        raise FileNotFoundError(f"Report template not found: {main_tex_template}")
    text = main_tex_template.read_text(encoding="utf-8")

    bib_src = template_dir / "sample.bib"
    if bib_src.is_file():
        shutil.copy2(bib_src, output_dir / bib_src.name)

    for target_name, attr_name in TEMPLATE_FIGURE_MAPPING.items():
        source_path = getattr(workflow_result, attr_name, None)
        if source_path is None or not Path(source_path).is_file():
            continue
        shutil.copy2(source_path, figures_dir / target_name)

    measured_filename = Path(workflow_config.measured_file_path).name
    substitutions: dict[str, str] = {
        "filemeas": _latex_escape_filename(measured_filename),
        "powerlevel": f"{workflow_config.power_level_dbm:g}",
        "noiselevel": f"{workflow_config.noise_floor:g}",
        "antennatype": antenna_type,
        "frequency": f"{frequency_mhz}",
        "distance": f"{distance_mm}",
        "mass": f"{mass_g}",
        "pssarref": f"{workflow_result.reference_pssar:.2f}",
        "pssarmeas": f"{workflow_result.measured_pssar:.2f}",
        "errscale": f"{100.0 * workflow_result.scaling_error:.2f}",
        "deltadist": rf"{workflow_result.distance_to_agreement:g}~mm\xspace",
        "deltadose": rf"{workflow_result.dose_to_agreement:g}~\%\xspace",
        "passrate": f"{workflow_result.pass_rate_percent:.1f}",
    }
    for name, value in substitutions.items():
        text = _set_latex_macro(text, name, value)

    out_path = output_dir / "main.tex"
    out_path.write_text(text, encoding="utf-8")
    return out_path
