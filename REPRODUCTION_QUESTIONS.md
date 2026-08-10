# Research questions for reproducing the Cannabis biopesticide article

The server is meant to answer scientific questions, not to receive one instruction to
"confirm the paper." Use the four-question sequence below. Each question maps to a small set of
tools whose results are chained through `metadata.next_tools`.

Append this explicit output request to **every** question:

> "Answer the scientific question directly. Return every non-null figure artifact produced by the
> tools (URL or path, kind and SHA-256) alongside the supporting numbers. Distinguish recomputed
> results from paper lookups. Do not return an internal task list or merely say that the paper's
> conclusion was confirmed."

## Question 1 — What is being screened, and is its chemical space pesticide-like?

Ask:

> "What compounds are used to train and screen the Cannabis biopesticide model, and does the
> Cannabis metabolome occupy the same chemical space as known pesticides?"

Use: `canpest_dataset_overview` + `chemical_space`.

Report the labelled active/inactive set, the unlabelled metabolite set, the six pest targets, the
measured nearest-neighbour Tanimoto overlap, and the chemical-space figure artifact. Do not infer
overlap from t-SNE geometry alone.

## Question 2 — Is the docking signal mechanistically and statistically useful?

Ask:

> "Do Cannabis metabolites and known pesticides engage the six pest targets differently, does
> random-matrix denoising isolate useful signal rather than noise, and does a physical-consistency
> docking veto reduce false positives?"

Use: `docking_analysis` + `rmt_feature_selection` + `docking_veto`.

Report the five expected docking trends and the OR28 exception, the recomputed Marchenko–Pastur
edge / signal count / selected-feature count, the false-positive-rate effect of the docking veto,
and the docking figure artifact. Preserve the documented numerical divergences from the paper.

## Question 3 — How reliable is the activity model, and which metabolites are candidates?

Ask:

> "How reliable is the pesticidal-activity QSAR model, and which Cannabis metabolites are credible
> biopesticide candidates within its applicability domain?"

Use: `qsar_model_quality` + `model_stack` + `predict_biopesticides`.

Separate metrics recomputed by this server from the authors' bundled OOF-CV lookup. Compare the
candidate **fraction**, not the absolute count, because the screening denominators differ. Return
the probability cutoff, backend and applicability-domain basis.

## Question 4 — What safety conclusion is actually supported?

Ask:

> "What can we defensibly conclude about the mammalian and aquatic safety of Cannabis metabolites
> versus synthetic pesticides from the evidence bundled with this article reproduction?"

Use: `tox_ecotox_reference`.

This is a published Syntelly lookup, not a live toxicity calculation. Report only endpoints that
carry an actual metabolite-versus-pesticide comparison. Aquatic entries in Table S1 are model
quality metrics, not toxicity outcomes; no DILI outcome is present. For live open-model LD50/DILI
predictions, route the user to `heracleum-tox` and label the result as a separate analysis.

## Bulk reproduction is an audit fallback, not the recommended user flow

`canpest_reproduce_all` and `canpest_reproduce_claims` remain available for regression testing and
numeric audit. They should not replace the four questions above in a validation report: a bulk dump
retrieves poorly and invites a generic confirmation instead of an interpretable answer.

## Answer contract

For each question:

1. lead with the direct scientific answer;
2. give the computed evidence and uncertainty/coverage limits;
3. attach every returned figure artifact with its kind and SHA-256;
4. distinguish recomputed values from bundled or published lookups;
5. state non-reproduced quantities explicitly; and
6. never expose planner/task-tracker logs as the answer.
