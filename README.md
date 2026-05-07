# When Does Trimming Help Conformal Prediction? A Retained-Law Diagnostic under Calibration Contamination

This directory contains the experiment framework for `When Does Trimming Help Conformal Prediction? A Retained-Law Diagnostic under Calibration Contamination`.

## Main commands

Setup `UV`

```bash
uv sync
```

Run the main suite

```bash
uv run run_all_main.py
uv run run_clarkson_direct_comparison.py
uv run run_clarkson_regression_comparison.py
uv run run_baseline_comparison.py
uv run run_supplemental_experiments.py
```

Run the smoke tests

```bash
uv run python -m unittest experiments.tests.test_smoke
```
