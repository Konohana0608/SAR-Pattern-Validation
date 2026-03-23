# SAR Pattern Validation (2D)

Toolkit to load SAR maps from CSV, register a measured map to a reference one, and evaluate 2D gamma for SAR pattern validation on a fixed grid.

## License

This project is licensed under the MIT license terms specified in the [LICENSE](LICENSE) file.

## Run

Notebook assets live under `notebooks/`:
- `notebooks/tutorial_gamma_pattern_validation_notebook.ipynb` — tutorial walkthrough of the full validation pipeline, from CSV loading through registration and gamma evaluation.
- `notebooks/voila.ipynb` — Voila-integrated notebook UI for interactive use of the workflow.

Features:
- **Load & preprocess**: `image_loader.py` (CSV → dense grids → SimpleITK images; absolute masking, peak normalization, dB conversion, plotting).
- **2D registration**: `registration2d.py` (translate/rigid, exhaustive search, multi-resolution pyramid).
- **Gamma evaluation**: `gamma_eval.py` (resample measured → reference, compute in **linear** peak-normalized space, optional ROI masks, plots).

## Install

```bash
uv venv .venv
source .venv/bin/activate # Windows PowerShell: .venv\Scripts\Activate.ps1
uv pip install -e ".[dev]"
uv pip install ipykernel
python -m ipykernel install --user --name sar-pattern-validation --display-name "Python (sar-pattern-validation)"
```

If you need the repository data files under `data/database/` or `data/measurements/`, install Git LFS and pull the large CSV assets:

```bash
git lfs install
git lfs pull
```

## Data
**Folder structure**
- `data/example/`
  - Small example CSVs for quick local runs.
- `data/database/`
  - Reference SAR CSVs stored with Git LFS.
- `data/measurements/`
  - Measurement CSVs stored with Git LFS - used for validation & regression testing.

**File Structure**
Examples of accepted headers:
- `x, y, SAR` (defaults to meters)
- `x [m], y [m], sSAR1g`
- `x_mm, y_mm, sSAR1g`
- `x [mm], y [mm], pssar10g`

The loader auto-detects:
- coordinate columns that explicitly match forms like `x`, `y`, `x [m]`, `y_mm`
- SAR column containing `"sar"`

Coordinates must use consistent physical units between files.

Policy note:
- Bare / unspecified coordinate headers default to **meters**.
- Explicit unit-bearing headers are still preferred, especially when preparing new datasets.

---

### Workflow (high level)

1. **Load** measured & reference CSVs with `SARImageLoader`
   - Builds grids (native or resampled)
   - Applies absolute cutoff mask
     $$ \text{cutoff} = \min(0.1,\; 2 \times \text{noise\_floor\_wkg}) $$
   - Peak-normalizes linear SAR
   - Produces linear and dB images

2. **Register**
   - Fixed = reference
   - Moving = measured
   - Returns measured resampled onto the reference grid

3. **Gamma**
   - Inputs are **linear peak-normalized SAR**
   - Measured is resampled onto the reference grid
   - Gamma is evaluated on the reference grid over the default overlap ROI:
     reference mask ∩ transformed measured mask

### Validation

Broad test runs skip `slow` tests by default. Run them explicitly when needed:

```bash
make tests-fast
make tests-slow
```

Run the measurement validation suite in parallel:

```bash
make measurement-validation
```

The slow integration and validation tests that read `data/database/` or `data/measurements/` also require `git lfs pull`.

---

### Key Parameters

**`SARImageLoader`**
- `measured_path`, `reference_path`
- `noise_floor_wkg`
- `resample_resolution` (meters or `None`)
- `show_plot`
- `warn`

**`Rigid2DRegistration`**
- `transform_type`: `TRANSLATE | RIGID`
- Stage parameters:
  - `translation_step`
  - `tx_steps`, `ty_steps`
  - `rot_step_deg`
  - `rot_span_deg`
- Uses Mattes Mutual Information
- Multi-resolution pyramid: `[4, 2, 1]`

**`GammaMapEvaluator`**
- `reference_sar_linear`
- `measured_sar_linear`
- `measured_to_reference_transform`
- `dose_to_agreement_percent`
- `distance_to_agreement_mm`
- `gamma_cap`
- Optional: `reference_mask_u8`, `measured_mask_u8`

Gamma tolerance:

$$
\Delta SAR = \frac{\text{dose\_to\_agreement\_percent}}{100}
$$

Distance-to-agreement (mm) is internally converted to pixel radius using reference spacing.

## Contributors
- Matthew Morvan
- Javier Ordonez -- Maintainer
- Melanie Steiner
- Mark Douglas -- Project Owner

Contact: (last-name)@itis.swiss
