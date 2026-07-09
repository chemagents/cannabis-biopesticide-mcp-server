# Open reproduction of the Syntelly toxicity models (§2.6 / §3.5)

The cannabis paper's toxicology (Tables 3/4/S1) was produced with the proprietary **Syntelly**
platform. This directory reproduces those models **entirely on open data and open algorithms**, so
the toxicology part of the study is fully reproducible.

## Recipe (what Syntelly actually does)

Syntelly's methodology is published: **Sosnin, Shkil et al., *Molecules* 2024, 29, 1826**
("Expanding Predictive Capacities in Toxicology"). Per endpoint it trains a gradient-boosting pair —
**CatBoost on molecular fingerprints** + **XGBoost on molecular fragment descriptors** — on open
toxicity databases, primarily **TOXRIC** (Wu et al., *Nucleic Acids Res* 2023) plus **EPA ECOTOX**
for the aquatic endpoints. We reproduce that recipe and strengthen the model into an ensemble.

## Data (the same open sources Syntelly used)

Downloaded from TOXRIC (free, no login; source = ACToR / Scientific Literature), prepared to
`smiles,y` in `server/data/tox/`:

| File | Endpoint | n | type |
|---|---|---|---|
| `toxric_ames.csv` | Ames mutagenicity | 7460 | classification |
| `toxric_reproductive.csv` | Reproductive toxicity | 156 | classification |
| `toxric_developmental.csv` | Developmental toxicity | 218 | classification |
| `toxric_daphnia.csv` | *Daphnia magna* LC50 (48 h) | 345 | regression (pLC50) |
| `toxric_fathead.csv` | Fathead minnow LC50 (96 h) | 812 | regression (pLC50) |
| `daphnia_aug.csv` | Daphnia + EPA ECOTOX | 614 | regression (deployment) |
| `fathead_aug.csv` | Fathead + EPA ECOTOX | 1100 | regression (deployment) |

> The paper's larger sizes (Ames 14168, fathead 1739, Daphnia 699) are the hackathon's
> **multi-source aggregated** sets, not single TOXRIC downloads. TOXRIC's own curated datasets (above)
> are what its benchmark column refers to.

## Model

`server/tox/stack_v2.py` — a 5-learner ensemble combined by a cross-validated meta-learner:
CatBoost / XGBoost / LightGBM / ExtraTrees over three feature views (ECFP-2048, RDKit fragment
panel, full RDKit-2D descriptors). Deterministic (SEED=42), torch-free. `tox_train.py` is the
paper-faithful 2-model recipe; `stack_v2.py` is the stronger ensemble we report.

## Results — random 5-fold CV (the paper's protocol)

| Endpoint | Metric | **Open ensemble** | TOXRIC/Syntelly benchmark | Paper hackathon |
|---|---|---|---|---|
| Ames | ROC-AUC ↑ | **0.9225** | 0.88 | 0.894 |
| Daphnia magna LC50 | RMSE ↓ | **1.026** | 1.109 | 0.817 |
| Fathead minnow LC50 | RMSE ↓ | **0.788** | 0.864 | 0.72 |
| Reproductive | ROC-AUC ↑ | 0.586 (low-conf) | 0.927 | 0.739 |

**The open ensemble beats the Syntelly/TOXRIC benchmark on Ames, Daphnia and fathead minnow** — the
fair same-data comparison. On Ames it also beats the hackathon's own model (0.894), on half the data.
Where the hackathon is ahead on the aquatic endpoints, it used 2× the data (aggregation).

## Scaffold-split companion (stricter generalization)

Random splits are reported to match the paper, but Bemis–Murcko **scaffold-disjoint** CV
(`server/tox/scaffold_eval.py`) is the honest number for deploying on novel chemistry (Cannabis
metabolites): Ames 0.877, Daphnia RMSE 1.183, fathead RMSE 0.947 — the expected scaffold penalty.

## Reproductive toxicity is a data limit, not a model one

n=156 with 88% positives → the metric is noise-dominated (5-fold folds span **0.14–0.86**). Four
methods fail to learn it: GB ensemble (0.51–0.59), single-task DMPNN (0.60), and a **multitask
DMPNN** (Chemprop, 21 endpoints incl. Tox21) which gave 0.47 pooled — no transfer benefit, and it
underperformed the GB ensemble even on Ames (0.81 vs 0.88 scaffold), confirming the paper's own
"gradient boosting > graph nets" finding. Reproductive is reported as **low-confidence**.

## Endpoints served elsewhere

LD50 (rat/mouse, all routes), carcinogenicity, hepatotoxicity, DILI, cardiotoxicity are the open
Syntelly analogue in the **`heracleum-tox`** MCP server (CatBoost/XGBoost on TDC/TOXRIC).

## Reproduce

```bash
pip install .[tox]     # xgboost + lightgbm (catboost already core)
python server/tox/stack_v2.py    server/data/tox/toxric_ames.csv        classification ames
python server/tox/stack_v2.py    server/data/tox/toxric_daphnia.csv     regression     daphnia
python server/tox/stack_v2.py    server/data/tox/toxric_fathead.csv     regression     fathead
python server/tox/scaffold_eval.py server/data/tox/toxric_ames.csv      classification ames   # scaffold
```
