# ML_NylC — Epistatic GP

A reproducible Python pipeline for NylC laboratory data, Epistatic GP modelling,
and experimental panel selection with calibrated uncertainty and k-DPP.

The implementation is based on the agreed notebooks **02, 03 and 04**.
MD analyses, the separate `Baslines` and `boltzgen_analysis` notebooks, and running
BoltzGen are outside the project scope. Candidate preparation from notebook 04
is included.

## Windows quick start

Prerequisite: uv 0.12.13. `.python-version` requests Python 3.12.14.
The following installation uses the checked-in `uv.lock` without changing it:

```powershell
uv sync --locked
uv run --no-sync nylc check --config configs/smoke.yaml
uv run --no-sync nylc run --config configs/smoke.yaml
uv run --no-sync pytest -q
```

The test report is written to `results/smoke/report/report.html`. The smoke
configuration uses six real variants, fewer optimisation steps and a small panel.
It checks the implementation and **does not provide a scientifically reliable
model evaluation**.

This working copy already contains a Windows environment in `.venv`.
You can also use it directly:

```powershell
.\.venv\Scripts\python.exe -m nylc run --config configs/smoke.yaml
```

## Linux and WSL: Snakemake

Windows and WSL need separate virtual environments when accessing the same
project. The provided setup installs all tools inside this project:

```bash
bash tools/setup_linux.sh
.venv-linux/bin/snakemake --snakefile workflow/Snakefile \
  --profile workflow/profiles/local \
  --config experiment_config=configs/smoke-linux.yaml
```

Alternatively, with uv already installed:

```bash
export UV_PROJECT_ENVIRONMENT=.venv-linux
uv sync --locked --group workflow
uv run --no-sync snakemake --snakefile workflow/Snakefile --cores 2 \
  --config experiment_config=configs/smoke-linux.yaml
```

The direct `nylc run` command also works on Linux. Snakemake and the Windows
entry point execute the same scientific functions.

## Full scientific run

```bash
uv run --no-sync nylc run --config configs/default.yaml
```

The default configuration reads all 36 laboratory entries; 35 fall within the
four modelled positions. F301L is recorded and excluded from GP training.
Six classical descriptor families are compared. Nested validation with multiple
optimisation restarts and the additional comparison analyses can take substantially
longer than the smoke test.

The full scientific run has not yet been executed during the migration.
Completed checks and remaining limitations are recorded in
[docs/verification.md](docs/verification.md).

Individual targets are also available:

```bash
uv run --no-sync nylc run features --config configs/default.yaml
uv run --no-sync nylc run evaluate --config configs/default.yaml
uv run --no-sync nylc run select --config configs/default.yaml
```

`run` computes required upstream stages and verifies existing results using their
checksums. `stage` executes exactly one stage and requires current upstream
results; Snakemake uses this command. `--force` forces recomputation.

## CPU, GPU and optional protein models

GP optimisation with SciPy runs on CPUs. A GPU accelerates the optional protein
models during feature computation. Changing hardware does not change model size
or scientific parameters. `--device cuda` fails with a clear error if CUDA is
unavailable; `auto` records the device actually selected.

ESM-2, including the three representations from notebook 03:

```bash
uv sync --locked --extra protein
uv run --no-sync python tools/fetch_esm2.py --config configs/esm2.yaml
uv run --no-sync nylc run --config configs/esm2.yaml --device cpu
# Run the same experiment on a GPU, if available:
uv run --no-sync nylc run --config configs/esm2.yaml --device cuda
```

`fetch_esm2.py` is the explicit download step. Computation runs offline by default.
Model revisions are pinned to immutable commits in the configuration; the
embedding cache accounts for revision and device. A second model size is not
activated automatically. The resolved revisions of both previously used sizes
are recorded in `references/model_revisions.json`.

For METL:

```bash
uv sync --locked --extra metl
```

Obtain the two 1D checkpoints from the [official METL model directory](https://github.com/gitter-lab/metl-pretrained#global-source-models)
and enter their actual SHA-256 checksums in a copy of `configs/metl.example.yaml`.
The example configuration intentionally fails while the checksums are missing.
The METL package is pinned to commit
`52358614c4b412e81e19e300485e8b85123bd903`. The classical CPU installation does
not download checkpoints or optional packages. METL-3D is not enabled in this
implementation; it was also disabled in the source notebook and requires a
suitable unsplit reference structure.

## SLURM

1. Copy the project and inputs to a filesystem shared by login and compute nodes.
   Recreate virtual environments on the cluster.
2. Run `bash tools/setup_linux.sh` on the login node. Add `--extra protein` for
   ESM-2 or `--extra metl` for METL.
3. Provide any required model weights in advance; compute nodes do not need
   internet access for the analysis itself.
4. Check the account and partitions in `workflow/profiles/slurm-cpu/config.yaml`
   and `slurm-gpu/config.yaml`. `thes2304`, `c23ms` and `c23g` were copied from the
   existing SLURM scripts; resource limits remain cluster-specific.
5. Inspect the execution plan first, then run it:

```bash
bash tools/run_slurm.sh --dry-run
bash tools/run_slurm.sh
```

GPU variant for an experiment with protein models:

```bash
NYLC_PROFILE=slurm-gpu NYLC_DEVICE=cuda NYLC_CONFIG=configs/esm2.yaml \
  bash tools/run_slurm.sh --dry-run
```

Only the feature stage requests a GPU in the GPU profile. Statistical stages
remain CPU jobs. Parallel execution currently takes place between independent
stages; outer cross-validation folds run within a single job. The validation
stage may therefore require an adjusted walltime limit. Installation and tests
alone do not submit cluster jobs.

## Project structure

| Location | Responsibility |
|---|---|
| `src/nylc/data` | Data validation, mutation/sequence mapping, candidate import |
| `src/nylc/features` | Classical and optional protein features, stored feature matrices |
| `src/nylc/models` | Epistatic GP, kernels, optimisation, calibration |
| `src/nylc/validation` | Original model comparison and baseline models from notebook 03 |
| `src/nylc/selection` | Activity threshold, k-DPP, control variants, sensitivity analyses |
| `src/nylc/reporting` | Figures and HTML report |
| `workflow` | Snakemake dependencies, local execution and SLURM profiles |
| `configs` | Scientific settings and test configurations |
| `inputs` | Copies of raw data with checksums |
| `references` | Historical results and provenance records, not pipeline inputs |

Each experiment has eight stages:
`prepare → features → diagnostics/evaluate/fit → predict → select → report`.
`diagnostics` and `evaluate` are independent of the final fit; the report collects
their results. By default, `fit` uses the six classical sources from notebook 04,
even when additional embeddings are enabled for notebook 03 comparisons.
Changes to this protocol are made explicitly through `selection.sources`.

## Reproducibility and methodological limitations

- Each stage records the fully resolved configuration, relevant parameters, code
  checksums, Git state, Python/package versions and output checksums in
  `manifest.json`. Changed or damaged artifacts are detected.
- Outputs are first created in a separate working directory. A successful run
  publishes the results and then its manifest. An exclusive stage lock prevents
  concurrent writers.
- An abrupt process termination may leave `.running` behind. Before removing it
  manually, check that the recorded process is no longer running. Normal Python
  exceptions remove the lock automatically. Failure details remain in `.work/`.
- PCA, distance scaling, hyperparameter selection and uncertainty calibration
  take place within the respective training split. Missing SEM values are
  imputed from that training set. The fixed scaling of the 20 classical amino
  acids does not use experimental target values.
- Calibration of the final model uses the setting selected on the full dataset.
  Its own goodness of fit is not independent evidence of uncertainty quality;
  nested outer validation provides that assessment.
- DPP selection uses the predicted mean and the prior kernel. Sigma calibration
  changes reported uncertainty, not the DPP draw.
- Candidate/kernel ordering was corrected relative to notebook 04. The corrected
  panel may therefore differ from the historical selection. See `docs/migration.md`.
- Identical software and seeds support controlled repetitions. Different CPUs,
  BLAS libraries or GPUs may introduce numerical differences; bitwise equality
  across platforms is not guaranteed.

Before publication, add **the exact unit and provenance of the activity
measurements, usage rights for the data and model weights, and a permanent
archive with a DOI**. These details cannot be reliably inferred from the available
tables. Code licensing and data rights are separate matters. The MIT licence
from the source project has been retained.

The CI template in `.github/workflows` is prepared for future use of `ML_NylC`
as a standalone repository. A nested workflow file is not automatically executed
by the parent thesis repository.

Further reading: [Snakemake](https://snakemake.readthedocs.io/en/stable/snakefiles/deployment.html),
[uv lockfiles](https://docs.astral.sh/uv/concepts/projects/sync/),
[PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).
