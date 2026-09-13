# Migration and changes

Source repository commit: `208ff10b36edee35991c75909d3cc333edb1f46b`.
Checksums of the notebook files actually read are also recorded in
`references/migration_sources.json`. This preserves provenance even when files
differ from the commit. The source project was not modified.

## Notebook 02

Includes measurement aggregation, traceable exclusions, strict mutation mapping,
FASTA export and sequence features. The previously exported and validated WT
FASTA is used as an explicit input. Interactive PDB inspection, identifying the
His-tag offset and manually transferring BoltzGen positions are not repeated
for every analysis run. The established mapping is versioned in
`data/candidates.py`. Other constructs or structures require a new scientific
review of this mapping.

## Notebook 03

Classical descriptors, frozen ESM/METL representations, Epistatic GP kernels,
additive comparison, the original discrete hyperparameter comparison, regularised
marginal likelihood optimisation, nested source/model selection, training-only
PCA, position-family holdout and the four baseline models have been moved into
separate modules. Feature sensitivity and kernel distributions are exported as
tables. The methodologically weaker holdout using parameters previously selected
on all data has been replaced by the strict holdout already present in the
notebook. Centrally generated figures and a report replace the previous plot
layouts; the pipeline does not blindly execute every notebook cell.

ESM and METL are explicit optional configurations. The classical default compares
six descriptor families, matching the source scope of notebook 04. The separate
`Baslines.ipynb` was subsequently added at the user's request; see below.

## Excluded boltzgen_analysis notebook

The separate notebook loads the BoltzGen candidate table, maps the four target
positions from the chain sequences, checks agreement with `designed_sequence`,
removes duplicate variants and marks mutation combinations already tested in the
laboratory. It also displays status/mutation frequencies and the `design_iiptm`
value. It does not implement its own GP fit or DPP selection. Candidate preparation
required by notebook 04 remains part of ML_NylC.

## Notebook 04

Includes nested sigma calibration, full deployment model selection/fitting,
candidate mapping and deduplication, prediction, activity threshold, shortlist
sensitivity, a fixed k-DPP draw, seed sensitivity, panel overlap, control variants
and result export. Prediction and selection are separate so that changing DPP
parameters does not trigger another GP training run.

### Kernel index correction

The source notebook assigned `candidate_matrix_index` before sorting by activity,
but computed the full kernel on the already sorted candidates. Access through
the old index could therefore use another candidate's kernel entries. ML_NylC
creates the kernel in the same original order as the explicitly stored
`candidate_ids`. This order is checked on loading. Subsequent sorting carries
valid matrix indices along. A test verifies the exact mapping after deliberately
reordering candidates.

This bug fix may affect the laboratory panel; it does not change GP means.
The historical panel is archived in `references`.

### Additional safeguards

- Missing SEM values are imputed within the training split. The supplied dataset
  has no missing SEM values, so this does not change its reference predictions.
- Invalid mutation strings, incorrect WT residues, duplicate positions and unknown
  amino acids are rejected instead of merely producing a warning.
- Model revisions and METL checkpoint checksums contribute to cache identity.
- Failed optimisations returning the numerical penalty value 1e30 are not selected
  as valid finite solutions. Non-convergence and boundary hits remain recorded
  in the optimisation metadata.
- Parameters and profiles do not modify global notebook variables. Separate
  EpistaticGP instances have independent settings and feature caches.

## Reference verification

An automated test loads the archived final hyperparameters and checks candidate
predictions against the historical table, joined by `candidate_id`. Means and
observed standard deviations must satisfy `rtol=1e-7`, `atol=1e-6`. This verifies
the numerical core; it does not replace a complete optimisation rerun with all
sources and seeds.

## Baselines notebook extension (14 September 2026)

The added `baselines` stage includes the notebook's classical models, optional
masked-marginal ESM-2 scores, Ridge with the zero-shot score, optional TabICL, and
the physical Epistatic GP with a linear zero-shot prior mean. Source notebook and
support-script checksums are recorded in `references/migration_sources.json`.

Scientific details preserved from the source:

- All mutated positions are masked simultaneously for ESM-2 scoring; log-odds are
  summed relative to WT residues, and WT receives score zero.
- Ridge and kNN hyperparameters are selected within inner LOOCV. StandardScaler
  is part of each estimator pipeline and is fitted anew inside each training fold.
- Bayesian Ridge and the 500-tree random forest are fixed comparators.
- The zero-shot prior GP refits its linear mean and covariance parameters in each
  outer fold. Its helper retains the original population target standard deviation
  (`ddof=0`), whereas the existing notebook-04 GP uses `ddof=1`. This distinction
  has not been silently harmonised. Its uncertainty is the original plug-in
  calculation and does not propagate uncertainty in the fitted mean coefficients.

Explicit adaptations:

- Hardware no longer changes ESM model size or silently enables/disables TabICL.
  Models are configured explicitly; weights are pinned and downloaded separately.
- The notebook read an older GP prediction file and required every fold to have
  selected physical/epistatic. The pipeline recomputes a clearly labelled fixed
  physical Epistatic GP comparator and includes current nested source-selection
  predictions separately. These are distinct evaluation protocols.
- Comparators must contain exactly the same variants and observed targets. A
  mismatch raises an error rather than a warning. Duplicate predictions are not
  silently discarded.
- kNN settings larger than an inner training set are excluded, allowing small
  smoke datasets. The full 35-variant analysis retains k = 1, 3, 5, 7.
- Missing SEM is imputed within training folds. Optimiser penalty failures are
  rejected. Runtime thread limits replace the notebook's unrestricted CPU use.
- The original MAE/Spearman comparison figure is now exported as PDF and PNG,
  with English labels. Raw zero-shot scores are omitted from the MAE panel.
- Paired model errors and the descriptive best-model gain are saved as CSV.

Reference fixtures in `references/baselines` were generated from the unmodified
original classical functions on nine variants and from the original zero-shot GP
support script on deterministic synthetic data. Tests compare migrated predictions
to these fixtures. They do not replace validation with real ESM/TabICL weights.
