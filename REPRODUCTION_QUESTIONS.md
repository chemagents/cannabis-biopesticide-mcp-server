# Reproduction questions

Ask these questions through CoScientist after registering the server. Return computed values, cohort definitions, methods, limitations and all generated figures/tables with download URLs and SHA-256 hashes. Download full tables rather than relying on previews.

The tool names below describe expected execution; they are not evidence that a run succeeded. [cases/cases.json](cases/cases.json) contains separate scientific checks; [cases/prompts.jsonl](cases/prompts.jsonl) contains prompts without reference answers. Keep expected numerical values out of blind test prompts.

**C01**

> Which compounds occur in the source, standardized and docking-prepared Cannabis sativa datasets (IJMS 2026, 27, 8000)? Reconstruct the selection flow and explain how well the prepared set represents the full metabolome.

Tools: `cannabis2026_dataset_audit`.

**C02**

> Do pesticide and AChE-inactive control docking profiles differ in the Cannabis study? Recompute all target comparisons, median differences and multiple-testing correction, retaining opposite effect directions.

Tools: `cannabis2026_docking_comparisons`.

**C03**

> Which residues are informative in the Cannabis study? For every target, reproduce Appendix A selection on saved outer split zero: RMT spectrum, inner-validation residue count, energy filtering and application to test. Verify that test data are not used for selection.

Tools: `cannabis2026_select_residues`.

**C04**

> Do RTE descriptors distinguish pesticides from AChE-inactive compounds? Reproduce CatBoost and LinearSVM for each target and the combined descriptors, selecting features within each outer training split.

Tools: `cannabis2026_validate_rte`.

**C05**

> How does the Cannabis structural DMPNN generalize to new scaffolds? Reproduce paired random and scaffold validation of one configuration with ROC-AUC and PR-AUC, separating it from the paper’s main random-CV result.

Tools: `cannabis2026_validate_dmpnn`.

**C06**

> Does the product of Cannabis structural and RTE scores reduce false positives? On identical held-out compounds compare both components and their product using ROC-AUC, PR-AUC, FPR, recall, confusion matrices and coverage at the paper’s evaluation threshold.

Tools: `cannabis2026_evaluate_fusion`.

**C07**

> Which Cannabis metabolites form the final top ten, and why? Reproduce structural screening, primary-metabolite exclusion and product-score ranking. Return both component scores, structures, provenance and changed ranks, exposing unavailable annotations.

Tools: `cannabis2026_prioritize_candidates`.

**C08**

> Are predicted Cannabis toxicology and ecotoxicology profiles more favorable than those of pesticides? Recompute endpoint comparisons and candidate profiles, retaining units, applicability and model provenance.

Tools: `cannabis2026_open_toxicity_profile`.

**C09**

> How reliable is a safety interpretation for prioritized Cannabis biopesticides? Check each toxicology prediction’s applicability, model errors and limitations of AChE-inactive controls, and identify what still needs experimental testing.

Tools: `cannabis2026_open_toxicity_profile`, `cannabis2026_dataset_audit`.

**C10**

> Reproduce the open toxicology benchmarks in Cannabis Supplementary Table S3: Ames, reproductive toxicity, Daphnia and fathead minnow under the reported splits. For each number give the actual cohort and size, base model or aggregation, measured metric and paper comparison. Assess whether random/scaffold differences isolate transfer and whether meta-model validation is independent. Keep these benchmarks separate from the main Syntelly profiles.

Tools: `cannabis2026_open_toxicity_validation`.

Record adapter/core revisions, parameters and input hashes with each run. Compare computed results with the article independently of whether the final narrative sounds plausible. See the [README](README.md#reproduction-fidelity) for known discrepancies.
