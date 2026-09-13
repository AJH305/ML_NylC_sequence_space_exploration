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
`Baslines.ipynb` was not migrated following the scope restriction.

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
