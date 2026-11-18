# Каталог инструментов

Каталог получен через MCP `tools/list` из кода, побайтно сверенного с закреплённой ревизией backend. Полные ограничения параметров и схема результата находятся в [api/tools.json](../api/tools.json).

Область каждого вызова — закреплённый набор статьи. Аргументы не являются произвольной загрузкой CSV. Числовые значения по умолчанию нужно интерпретировать вместе с выбранным протоколом.

## `cannabis2026_dataset_audit`

Study: Cannabis2026; paper DOI: 10.3390/ijms27188000; dataset: cannabis_2026.
Which Cannabis source cohorts and six residue-energy matrices are available, and do they
match the final article? Audit IDs/structures, 3171 labels, 2749 unlabeled source rows and missing
final 2465/5027 DS1 cohort mapping. AChE-inactive references are not universal non-pesticides.
Return the separately reported raw/standardized/docking-prepared flow for all three paper cohorts.
Also return the bundled original paper PDF as a source-reference artifact for CoScientist's existing paper reader.

| Аргумент | Обязательный | По умолчанию | Допустимые значения |
|---|---|---|---|
| Нет параметров | — | — | — |

[Как читать результат](INTEGRATION.md#результат-инструмента).

## `cannabis2026_docking_comparisons`

Study: Cannabis2026; paper DOI: 10.3390/ijms27188000; dataset: cannabis_2026.
Do registered pesticides and AChE-inactive references have different docking profiles?
Recompute six two-sided Mann–Whitney tests, BH FDR, median differences and distributions.
Does not rerun Vina; tests source scores. Final DS1 subset remains unmapped.

| Аргумент | Обязательный | По умолчанию | Допустимые значения |
|---|---|---|---|
| Нет параметров | — | — | — |

[Как читать результат](INTEGRATION.md#результат-инструмента).

## `cannabis2026_select_residues`

Study: Cannabis2026; paper DOI: 10.3390/ijms27188000; dataset: cannabis_2026.
Which target residues survive Appendix A's training-only RMT and 98% energy-contribution gate?
Fit on an explicit source outer split; choose top-m by inner-validation ROC-AUC on cumulative raw
energies, retain original columns, export split registry and train/test features. Returns spectrum,
full m scan and contribution audit. This is not the older global RMT95/point-biserial protocol.
Unreported numerical choices and unverified final-paper split equivalence are explicitly disclosed.

| Аргумент | Обязательный | По умолчанию | Допустимые значения |
|---|---|---|---|
| `target` | Нет | `"GLC1"` | ["GSTE2", "ACHE", "GLC1", "UGT202A2", "GSTM12", "OR28"] |
| `split_index` | Нет | `0` | См. JSON Schema |
| `inner_splits` | Нет | `5` | См. JSON Schema |
| `inner_test_size` | Нет | `0.2` | См. JSON Schema |
| `seed` | Нет | `42` | См. JSON Schema |

[Как читать результат](INTEGRATION.md#результат-инструмента).

## `cannabis2026_validate_rte`

Study: Cannabis2026; paper DOI: 10.3390/ijms27188000; dataset: cannabis_2026.
Do RTE features discriminate pesticide and AChE-inactive reference compounds? Actually fit
CatBoost and linear SVM for all six proteins and ALL6, with Appendix A residue selection refitted
on each requested source outer split; default is all 30 source splits (420 fits).
A configured execution cache is verified against archived algorithm/software/source data and
metrics are recomputed from held-out scores; cached fits are explicitly distinguished from new training.
Return ROC-AUC, both PR area definitions, FPR, recall,
confusion counts and downloaded held-out predictions. Optionally export fitted models. Source
folds and explicit unspecified hyperparameter choices are not asserted to match final-paper runs.

| Аргумент | Обязательный | По умолчанию | Допустимые значения |
|---|---|---|---|
| `split_indices` | Нет | `null` | См. JSON Schema |
| `iterations` | Нет | `500` | См. JSON Schema |
| `include_models` | Нет | `false` | См. JSON Schema |

[Как читать результат](INTEGRATION.md#результат-инструмента).

## `cannabis2026_open_toxicity_validation`

Study: Cannabis2026; paper DOI: 10.3390/ijms27188000; dataset: cannabis_open_tox_2026.
Reproduce Supplement S3's open toxicity benchmark: five base learners, mean blend and source
meta-stack on pinned TOXRIC data. Uses a separately pinned worker environment and actual saved
OOF predictions; cached runs are verified. Source meta CV is not strictly nested or scaffold-disjoint.
These open benchmarks do not replace main-text Syntelly outputs. Reproductive has no scaffold reference.
The two aquatic endpoints expose separate TOXRIC-only and augmented TOXRIC+ECOTOX cohorts;
sample counts and estimator choices must remain explicit in random/scaffold comparisons.

| Аргумент | Обязательный | По умолчанию | Допустимые значения |
|---|---|---|---|
| `endpoint` | Нет | `"ames"` | ["ames", "reproductive", "daphnia", "fathead"] |
| `scheme` | Нет | `"random"` | ["random", "source_scaffold"] |
| `cohort` | Нет | `"toxric"` | ["toxric", "toxric_ecotox"] |

[Как читать результат](INTEGRATION.md#результат-инструмента).

## `cannabis2026_open_toxicity_profile`

Study: Cannabis2026; paper DOI: 10.3390/ijms27188000; dataset: cannabis_2026.
How do the available Cannabis compounds compare with pesticide references under open
toxicity models, and how applicable are the predictions? Fit the existing ChemAgents open
two-booster approximation motivated by Syntelly 2024, validate identity-random/scaffold 5-fold,
and predict every source compound. Return per-compound values, cohort comparisons, kNN domain
scores and training overlap. Uses verified caches after the first CPU run. These are new open
predictions; final-paper Syntelly weights/cohorts and absolute aquatic units are unavailable.
Includes named predictions and endpoint-specific domain/overlap matrices for all ten reviewed
Figure4 connectivities, with full candidate-level CSV evidence. These are source-model estimates.
hERG is channel blockade; a separate broad hepatotoxicity output is not invented from DILI.
recipe=nested_paper2024_v2 replays completed nested-tuning runs with applied 2024 preparation,
inner-fold feature selection and verified saved models. Request explicit endpoints; unavailable
completed runs are reported as unavailable, without falling back to fixed-v1 predictions.

| Аргумент | Обязательный | По умолчанию | Допустимые значения |
|---|---|---|---|
| `endpoints` | Нет | `null` | См. JSON Schema |
| `recipe` | Нет | `"fixed_public_v1"` | ["fixed_public_v1", "nested_paper2024_v2"] |

[Как читать результат](INTEGRATION.md#результат-инструмента).

## `cannabis2026_validate_dmpnn`

Study: Cannabis2026; paper DOI: 10.3390/ijms27188000; dataset: cannabis_2026.
Does the structure-based Cannabis DMPNN transfer to unseen scaffolds? Actually train Chemprop
directed-bond message passing plus train-normalized RDKit2D descriptors under the published
depth3/hidden300/dropout0.1/mean-sum-norm configuration. paired_cv evaluates both random and
scaffold five-fold CV, with separate validation subsets, best checkpoints and complete held-out
predictions. source_random uses the existing RTE outer split registry for matched fusion studies;
it defaults to all30source splits, or the explicitly requested indices. Original final3403labels
and author fold/checkpoint details are absent: the available3171labels are used and all deviations
declared. PR trapezoidal area and average precision are separate; class scores are not measured
pesticide efficacy. Long CPU training is cached; incomplete runs resume only completed verified folds.

| Аргумент | Обязательный | По умолчанию | Допустимые значения |
|---|---|---|---|
| `protocol` | Нет | `"paired_cv"` | ["paired_cv", "source_random"] |
| `source_split_indices` | Нет | `null` | См. JSON Schema |
| `max_epochs` | Нет | `60` | См. JSON Schema |
| `include_models` | Нет | `false` | См. JSON Schema |

[Как читать результат](INTEGRATION.md#результат-инструмента).

## `cannabis2026_evaluate_fusion`

Study: Cannabis2026; paper DOI: 10.3390/ijms27188000; dataset: cannabis_2026.
Does multiplying Cannabis DMPNN and RTE scores improve held-out classification?
Train an ALL6 CatBoost branch with AppendixA train-only residue selection on exactly the
structural outer partitions. Compare both branches and product at threshold0.5: ROC-AUC,
both PR-area definitions, FPR, recall, confusion counts and positive retention. Return all
matched held-out/query predictions and optional model exports. Reuse verified DMPNN fits.
Product is not a calibrated joint posterior; lower FPR may accompany lower recall. Available
source cohort and reconstructed folds are not claimed to reproduce author final checkpoints.

| Аргумент | Обязательный | По умолчанию | Допустимые значения |
|---|---|---|---|
| `protocol` | Нет | `"paired_cv"` | ["paired_cv", "source_random"] |
| `source_split_indices` | Нет | `null` | См. JSON Schema |
| `max_epochs` | Нет | `60` | См. JSON Schema |
| `iterations` | Нет | `500` | См. JSON Schema |
| `include_models` | Нет | `false` | См. JSON Schema |

[Как читать результат](INTEGRATION.md#результат-инструмента).

## `cannabis2026_prioritize_candidates`

Study: Cannabis2026; paper DOI: 10.3390/ijms27188000; dataset: cannabis_2026.
Which Cannabis candidates rank highest, and can Figure4 be reproduced?
Return the published top10 names/rounded product scores separately from new source-cohort
probabilities and ranks. Computed mode uses cached paired DMPNN/RTE ensembles and the0.70
structural screen. Secondary-metabolite/occurrence annotations require an explicit source and
matching SMILES; missing annotation stays unknown and is never silently excluded as primary.
No0.5final-ranking cutoff or toxicity selection is applied. reference_only skips model fits.
Reviewed Figure4 graphs are linked to exact source records: return their identity evidence and,
in computed mode, newly reconstructed component probabilities separately from reported scores.
Without author final CCD cohort/annotations, original component scores and exact final selection
remain unavailable; no names, biological occurrence or separate probabilities are invented.

| Аргумент | Обязательный | По умолчанию | Допустимые значения |
|---|---|---|---|
| `annotations` | Нет | `null` | См. JSON Schema |
| `reference_only` | Нет | `false` | См. JSON Schema |
| `scheme` | Нет | `"random"` | ["random", "scaffold"] |
| `max_epochs` | Нет | `60` | См. JSON Schema |
| `iterations` | Нет | `500` | См. JSON Schema |

[Как читать результат](INTEGRATION.md#результат-инструмента).
