# Migration verification record

As of 13 September 2026. The tests use the supplied inputs and the environments
pinned by `uv.lock`, with Python 3.12.14.

## Completed checks

| Check | Result |
|---|---|
| Windows: `pytest -q` | 15 tests passed in 31.55 seconds |
| Linux/WSL: scientific reference and adapter tests | 13 tests passed in 12.66 seconds |
| Windows: `ruff check src tests tools` | Passed |
| Windows: complete run with `configs/smoke.yaml` | All eight stages succeeded; HTML report and PNG/PDF files generated |
| Linux/WSL: Snakemake with `configs/smoke-linux.yaml` | All eight stages succeeded; resumed from saved results after an interruption before the report stage |
| Package build with `uv build --offline` | Wheel and source distribution built successfully |
| Installed Windows dependencies: `uv pip check` | No version conflicts |
| Lockfile: `uv lock --check --offline` | Current and consistent; 189 resolved packages including optional groups |
| SLURM GPU profile with ESM configuration, `--dry-run` | Workflow planned successfully; no jobs submitted |
| SLURM CPU profile with default configuration, `--dry-run` | Workflow planned successfully; no jobs submitted |

The smoke configuration uses six real laboratory variants, one classical
descriptor source and reduced optimisation. It is an installation check and
does not provide a reliable scientific model evaluation.

During the Linux test, the Pytest cache previously created on Windows was not
writable; all 13 tests passed nevertheless. When alternating between platforms
in the same working directory, Linux can use a separate cache:
`pytest -q -o cache_dir=.pytest_cache-linux`.

## What the tests cover

- Laboratory measurements: mean, sample standard deviation, SEM and the documented
  exclusion of the variant outside the modelled positions.
- Strict mutation mapping, correct WT residues and unchanged checksums of the
  original laboratory and candidate data.
- Epistatic GP kernel and prediction against an independent linear algebra calculation.
- Candidate means and observed standard deviations at the archived final
  hyperparameters against the complete historical candidate table, using
  tolerances `rtol=1e-7`, `atol=1e-6`.
- Training-only PCA and SEM imputation, and sigma calibration.
- DPP size, seed repeatability, invalid rank and the corrected mapping between
  sorted candidates and kernel rows.
- End-to-end execution, resuming without unnecessary recomputation, changing a DPP
  seed without refitting the GP, and detecting damaged artifacts.
- The original discrete model comparison and strict position holdout on small datasets.
- The ESM interface, revision pinning and feature storage with a controlled test
  backend; this test does not load real model weights.

## Not yet executed

- Full optimisation and nested validation with all 35 eligible laboratory variants,
  six descriptor sources and the regular number of restarts.
- Real ESM/METL inference with downloaded weights, and CUDA computation. METL also
  requires verified checkpoints supplied by the user.
- Submission and execution on the actual SLURM cluster. The account, partitions,
  drivers and resource limits must be checked there.

The historical DPP laboratory panel is not an equality reference for the new
selection because of the corrected index bug. The GP reference check is independent
of this correction. A permanently archived full run, including data rights and
citation information, is still required before scientific publication.
