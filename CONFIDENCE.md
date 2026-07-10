# Does docking / RMT-RTE make the QSAR model *more confident*? (Section 3.3)

An adversarial, honest test of the brief **"show that adding RMT — or other forms of docking scores —
improves the model's confidence."** Confidence here means the *quality of the probabilities themselves*
(calibration, sharpness, trustworthiness of a high-confidence call), which is orthogonal to the
paper's headline discrimination metric (ROC-AUC).

## Design

Same base learner and feature ladder as `qsar_model_quality` (`models.qsar_ablation`) — **only the
features change across rungs**, so any change in confidence is attributable to the docking / RMT-RTE
information, not the model. Base learner: `HistGradientBoostingClassifier` (the repo's torch-free
DMPNN-SD analogue). Protocol: the **10 random 80/20 splits** (`split_registry.csv`) + the **1
Bemis–Murcko scaffold split** (novel-chemistry / deployment regime). Trainer: `server/confidence.py`
(deterministic, SEED=42); `python -m server.confidence` regenerates the bundled result and figure.

| rung | docking channel added to the 217 RDKit-2D descriptors |
|---|---|
| `structure` | none (structure-only baseline) |
| `+dock6` | the 6 global docking scores |
| `+raw_rte` | the 390 raw residue-term energies (full per-residue docking, no denoising) |
| `+rmt_rte_sel` | RMT-selected top-*m* RTE columns |
| `+rmt_rte_rec` | RMT signal-subspace reconstruction (the paper's headline RMT-RTE) |

Metrics per split test set (mean ± std over the splits): **Brier ↓**, **log-loss ↓**, **ECE ↓**
(10-bin), Murphy **reliability ↓** / **resolution ↑**, **sharpness** (std of p) ↑, ROC-AUC ↑
(reference), **precision@0.7 ↑** (tied to the paper's >0.7 candidate rule) and **@0.9**, plus the
docking-consistency **veto** `p_final = p_QSAR × p_RMT-RTE` and a symmetric **mean-fusion** control.

## Result 1 — docking / RMT-RTE as *features*: no confidence gain

Random 5-fold-style CV (mean over 10 splits):

| rung | Brier ↓ | ECE ↓ | resolution ↑ | ROC-AUC ↑ | precision@0.7 ↑ |
|---|---|---|---|---|---|
| `structure` | **0.1130** | 0.0400 | **0.1387** | **0.9179** | 0.9037 |
| `+dock6` | 0.1126 | **0.0332** | 0.1385 | 0.9165 | 0.9103 |
| `+raw_rte` | 0.1146 | 0.0361 | 0.1361 | 0.9133 | 0.9072 |
| `+rmt_rte_sel` | 0.1136 | 0.0381 | 0.1377 | 0.9143 | **0.9113** |
| `+rmt_rte_rec` | 0.1148 | 0.0350 | 0.1358 | 0.9134 | 0.9039 |

Paired vs `structure` (Wilcoxon signed-rank, 10 splits): the richest rung `+rmt_rte_rec` **worsens**
Brier (+0.0017, *p*=0.037) and resolution (−0.0029, *p*=0.014); ECE improvements are not significant;
only a small precision@0.7 bump for `+dock6` (*p*=0.010) and `+rmt_rte_sel` (*p*=0.037) survives. The
**scaffold** split says the same — `structure` ROC-AUC (0.8232) is the best rung, every RTE variant is
below it (raw 0.8076, rec 0.8134):

| rung | Brier ↓ | ECE ↓ | ROC-AUC ↑ | precision@0.7 ↑ |
|---|---|---|---|---|
| `structure` | **0.2696** | **0.2721** | 0.8232 | 0.6099 |
| `+dock6` | 0.2733 | 0.2778 | **0.8240** | 0.6089 |
| `+raw_rte` | 0.2883 | 0.2846 | 0.8076 | 0.6119 |
| `+rmt_rte_sel` | 0.2874 | 0.2953 | 0.8121 | 0.6028 |
| `+rmt_rte_rec` | 0.2806 | 0.2823 | 0.8134 | 0.6066 |

The 217 RDKit-2D descriptors already saturate; RMT-RTE is a weak (~0.80 AUC) standalone signal that
only dilutes them. This is **consistent with the repo's own bundled ablation**, whose HGB rows show
RMT-RTE-rec *lowering* HGB ROC-AUC (0.9300 → 0.9203).

## Result 2 — the docking veto: a thresholding artifact, not a confidence gain

The `p_QSAR × p_RMT-RTE` veto looks like a big win at a fixed 0.7 cut — but it is an operating-point
shift, not better probabilities:

| | random | scaffold |
|---|---|---|
| FPR | 0.154 → **0.067** | 0.588 → **0.442** |
| recall | 0.846 → 0.630 | 0.922 → 0.802 |
| precision@0.7 | 0.904 → **0.952** | 0.610 → **0.697** |
| coverage@0.7 | 0.453 → 0.179 | 0.666 → 0.400 |
| **ROC-AUC** | 0.918 → **0.901** | 0.823 → **0.751** |
| **precision at matched coverage** | struct **0.987** vs veto 0.952 | struct **0.772** vs veto 0.697 |

The precision@0.7 / FPR gains come entirely from the product deflating probabilities so **coverage
collapses** (0.45 → 0.18). Judged fairly — same number of most-confident calls (matched coverage), or
by ROC-AUC/PR-AUC — **the structure model alone is more precise and ranks better**. You get a better
precision/recall trade by simply raising the structure model's own threshold. Symmetric mean-fusion
`½(p_QSAR + p_RMT-RTE)` is no better (random AUC 0.900).

## The DMPNN and the DMPNN+HGB stack — RMT-RTE doesn't help them either

The obvious rebuttal is "you used HGB; the paper's model is the DMPNN stack, where RMT is complementary."
We checked directly. The team's own **paired k-fold CV** (`Activity/dmpnn/eval_rmt_rte.py` — same
train/val/test partition, same RMT fit, same DMPNN seeds per fold; only the extra feature block differs;
bundled as `server/data/reference/dmpnn_stack_rmt_cv.json`) adds `rmt_rte_rec` to the `dock_eng+rdkit`
baseline and measures DMPNN, HGB and the blend. Adding RMT-RTE **lowers** ROC-AUC and PR-AUC everywhere:

| model | random baseline → +rmt | scaffold baseline → +rmt |
|---|---|---|
| DMPNN | 0.9260 → **0.9175** | 0.9084 → **0.9020** |
| HGB | 0.9159 → **0.9064** | 0.8989 → **0.8951** |
| **blend** | 0.9313 → **0.9253** | 0.9149 → **0.9108** |

Same direction for PR-AUC, and the same across all three base-feature configs (`rdkit_only`,
`integration`, `global_dock`). A residue-feature breakdown shows why — RMT-**reconstructed** RTE is the
*weakest* of the RTE variants, below even raw RTE:

| features (HGB, random) | RDKit-2D | raw RTE | RMT-selected RTE | RMT-reconstructed RTE |
|---|---|---|---|---|
| ROC-AUC | **0.9207** | 0.7932 | 0.7837 | 0.7600 |

RMT-RTE is a weak (~0.80) and partly redundant signal; adding it dilutes rather than sharpens, on the
graph net exactly as on the tree. **My earlier claim that RMT lifts the blend (0.9343 → 0.9415) was an
artifact of the single `split_00` ablation; the honest k-fold CV refutes it for every model.**

## Verdict

**Adding docking / RMT-RTE does not improve model confidence — on any model here.** On the HGB analogue
it is flat-to-worse as features and a thresholding trade as a veto (dominated by structure at matched
coverage). On the DMPNN and the DMPNN+HGB stack, the team's paired k-fold CV shows it *lowers*
discrimination on both random and scaffold splits. The one number that suggested otherwise was a
single-split artifact. What remains unmeasured is calibration (Brier/ECE) *on the stack specifically* —
`eval_rmt_rte.py` saved only summary AUC/PR-AUC, not per-fold OOF probabilities; re-running it with
probability persistence would settle it, but since RMT-RTE uniformly *lowers* the stack's discrimination,
a calibration reversal is unlikely.

## Reproduce

```bash
python -m server.confidence     # -> server/data/reference/confidence_ablation.json + confidence_reliability.png
```
