# Suggested Commands

## Environment
```bash
uv sync --all-groups          # install all deps including dev
uv run <cmd>                  # run any command in the venv
```

## Testing
```bash
make tests                                        # all tests (parallel with xdist)
make tests-fast                                   # exclude slow marker
make tests-slow                                   # slow tests only
make tests-cov                                    # with coverage report
make tests target=test_voila_frontend.py          # specific file
make tests target=test_voila_frontend.py::TestSarGammaComparisonUI::test_foo  # specific test
uv run pytest -v tests/test_voila_frontend.py     # direct pytest
uv run pytest -v -m "not slow" tests/             # fast tests
uv run pytest -v -m e2e tests/test_voila_e2e.py  # e2e only (needs voila server)
```

## Linting & Formatting
```bash
make lint       # ruff check
make format     # ruff format
make typecheck  # ty type checker
uv run ruff check --fix src/ tests/
uv run ruff format src/ tests/
```

## Voila
```bash
make voila-smoke                               # smoke test
make kill-voila                                # stop stale voila processes
uv run voila notebooks/voila.ipynb             # launch dashboard
uv run python run_voila_smoke.py               # smoke runner directly
```

## Measurement validation
```bash
make measurement-validation         # parallel xdist run
make measurement-dashboard          # generate HTML dashboard
make measurement-dashboard-open     # generate + open in browser
```

## Pre-commit
```bash
make setup-pre-commit      # install hooks
make run-pre-commit        # run on staged files
make run-pre-commit-all    # run on all files
```

## Git LFS (for data files)
```bash
git lfs install
git lfs pull
```
