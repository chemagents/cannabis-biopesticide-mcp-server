"""Published reference values from the Cannabis biopesticide paper (for reproduce_all/claims)."""
from __future__ import annotations

import json
from pathlib import Path

from .config import get_settings

PAPER = "Biopesticidal Potential of the Cannabis sativa L. Metabolome (Denoised, Docking-Informed QSAR)"

# Section 3.1 datasets
DATASETS = {"DS1_metabolites": 5211, "DS2_pesticides": 1709, "DS3_ache": 1698,
            "labelled_docked": 3171, "active": 1680, "inactive": 1491}

# Section 3.2 docking (Mann-Whitney median-diff active vs inactive)
DOCKING = {"expected_delta_range": [-1.01, -0.70], "or28_delta": 1.20,
           "metabolite_median_range": [-7.2, -5.2], "anomalous_protein": "OR28"}

# Section 3.3 RMT + QSAR
RMT = {"lambda_plus": 1.938, "q": 0.1538, "n_signal": 19,
       "m_opt_random": 161, "m_opt_scaffold": 64, "inner_auc_random": 0.747}
QSAR = {"cb_sd_range": [0.679, 0.775], "svm_sd_range": [0.642, 0.754], "cb_sd_all6": 0.802,
        "dmpnn_sd_roc": 0.9283, "fpr_before": 0.1220, "fpr_after_veto": 0.0492,
        "roc_after_veto": 0.9176}

# DMPNN-SD is a weighted stack (soft-voting ensemble) of a D-MPNN graph net and an HGB model on
# RDKit-2D + engineered docking features. Numbers below are the authors' matched 5-fold OOF CV on
# the 3171 labelled compounds (feature_set dock_eng+rdkit) — the blend beats either component on
# every metric. Per-feature-set CV baselines reach blend ROC-AUC ~0.930 (≈ the paper's 0.9283).
QSAR_STACK = {
    "blend_w_dmpnn": 0.62, "threshold": 0.45,
    "dmpnn": {"roc_auc": 0.9087, "pr_auc": 0.9268, "f1": 0.8531, "bal_acc": 0.8436},
    "hgb":   {"roc_auc": 0.8995, "pr_auc": 0.9190, "f1": 0.8396, "bal_acc": 0.8281},
    "blend": {"roc_auc": 0.9142, "pr_auc": 0.9326, "f1": 0.8578, "bal_acc": 0.8509},
    "ranking": "blend > dmpnn > hgb",
    "per_featureset_blend_roc": {"rmt_rte_rdkit_only": 0.9313, "rmt_rte_integration": 0.9300,
                                 "rmt_rte_global_dock": 0.9317},
}

# Confidence / calibration ablation (Section 3.3) — does docking / RMT-RTE make the model MORE SURE?
# An adversarial test of the TZ "adding RMT / docking-scores improves confidence", orthogonal to
# ROC-AUC. HGB base learner + qsar_ablation feature ladder, calibration/sharpness/precision metrics
# over the 10 random splits + the scaffold split. Bundled: server/data/reference/confidence_ablation.json
# (+ confidence_reliability.png). Regenerate: python -m server.confidence. See CONFIDENCE.md.
# Honest verdict: for the open torch-free HGB analogue it does NOT — structure descriptors saturate,
# and the p_QSAR x p_RMT-RTE veto's precision@0.7/FPR gains are a thresholding artifact (at matched
# coverage the structure model alone is more precise; veto AUC is lower). RMT helps only the DMPNN/blend.
CONFIDENCE_ABLATION = {
    "protocol": "10 random 80/20 splits + 1 Bemis-Murcko scaffold split; HGB base learner; features vary",
    "random_ladder_auc": {"structure": 0.9179, "+dock6": 0.9165, "+raw_rte": 0.9133,
                          "+rmt_rte_sel": 0.9143, "+rmt_rte_rec": 0.9134},   # structure is best; RMT does not help
    "random_ladder_brier": {"structure": 0.1130, "+rmt_rte_rec": 0.1148},   # +rmt_rte_rec Brier WORSE (p=0.037)
    "veto_random": {"fpr": [0.1542, 0.0672], "precision@0.7": [0.9037, 0.9523],
                    "coverage@0.7": [0.4534, 0.1789], "auc": [0.9179, 0.9007],
                    "matched_coverage_precision": [0.9871, 0.9523]},         # veto worse at matched coverage
    "veto_scaffold": {"auc": [0.8232, 0.7509], "matched_coverage_precision": [0.7717, 0.6969]},
    "dmpnn_blend_rmt_helps": {"blend_roc_auc": [0.9343, 0.9415], "blend_pr_auc": [0.9485, 0.9543]},  # split_00
    "verdict": "docking/RMT-RTE does not improve the HGB analogue's confidence in either regime; the veto "
               "is a precision/operating-point tradeoff, not a calibration gain. RMT's real benefit is on "
               "the graph-net stack (DMPNN/blend), where it lifts ROC-AUC 0.9343->0.9415.",
}

# Section 3.4 candidates
CANDIDATES = {"n_high_conf": 1010, "fraction": 0.4097, "outside_ad_fraction": 0.5213}

# Supplementary Table S1 — Syntelly toxicity-model quality (same platform as the Heracleum paper)
TOX_METRICS = {
    "mouse_oral_ld50": ("RMSE", 0.45), "rat_oral_ld50": ("RMSE", 0.47),
    "carcinogenicity": ("ROC-AUC", 0.79), "ames": ("ROC-AUC", 0.89),
    "reproductive": ("ROC-AUC", 0.74), "hepatotoxicity": ("ROC-AUC", 0.81),
    "dili": ("ROC-AUC", 0.90), "daphnia_lc50": ("RMSE", 0.81),
    "fathead_minnow_lc50": ("RMSE", 0.72), "acute_aquatic": ("ROC-AUC", 0.82),
}
# Section 3.5 headline tox/ecotox comparison (metabolites vs synthetic pesticides)
TOX_FINDINGS = {"metabolite_ld50_rat": 1480, "metabolite_ld50_mouse": 1290,
                "pesticide_ld50_rat": 1250, "pesticide_ld50_mouse": 990,
                "hepatotoxic_metabolites_pct": 15, "hepatotoxic_pesticides_pct": 81}

# --- Open reproduction of the Syntelly toxicity models (Section 2.6 / 3.5) ------------------------
# Syntelly's recipe is published in Sosnin/Shkil et al., Molecules 2024, 29, 1826: per endpoint a
# gradient-boosting pair (CatBoost on fingerprints + XGBoost on fragment descriptors) trained on the
# open databases TOXRIC + EPA ECOTOX. We reproduce it on that SAME open data with a stronger ensemble
# (CatBoost/XGBoost/LightGBM/ExtraTrees over ECFP + fragment + full-RDKit-descriptor views, combined
# by a cross-validated meta-learner). Metrics below are 5-fold RANDOM CV (the paper's own protocol);
# see TOX_REPRODUCTION.md for the scaffold-split companion and the multitask-DMPNN negative result.
# Trainer: server/tox/stack_v2.py on server/data/tox/*.csv (deterministic, SEED=42, torch-free).
TOX_OPEN_MODELS = {
    "recipe": "ensemble (CatBoost/XGBoost/LightGBM/ExtraTrees over ECFP + fragments + RDKit-2D) + meta-stack",
    "protocol": "5-fold random CV (matches the paper); scaffold companion in TOX_REPRODUCTION.md",
    "data_sources": "TOXRIC (Wu et al., Nucleic Acids Res 2023) + EPA ECOTOX (aquatic augmentation)",
    # endpoint -> metric, our open ensemble, TOXRIC/Syntelly benchmark, paper hackathon, n
    "endpoints": {
        "ames":                {"metric": "ROC-AUC", "ours": 0.9225, "benchmark": 0.88,  "hackathon": 0.894, "n": 7460},
        "daphnia_magna_lc50":  {"metric": "RMSE",    "ours": 1.026,  "benchmark": 1.109, "hackathon": 0.817, "n": 345},
        "fathead_minnow_lc50": {"metric": "RMSE",    "ours": 0.788,  "benchmark": 0.864, "hackathon": 0.72,  "n": 812},
        "reproductive":        {"metric": "ROC-AUC", "ours": 0.586,  "benchmark": 0.927, "hackathon": 0.739, "n": 156,
                                "note": "low-confidence: n=156, 88% positive -> metric is noise-dominated "
                                        "(5-fold folds span 0.14-0.86). Not learnable at this open-data scale "
                                        "under GB, single-task DMPNN, or multitask DMPNN — a data limit, not a model one."},
    },
    "beats_benchmark": ["ames", "daphnia_magna_lc50", "fathead_minnow_lc50"],
    "covered_by_heracleum_tox": ["mouse_oral_ld50", "rat_oral_ld50", "carcinogenicity",
                                 "hepatotoxicity", "dili", "cardiotoxicity"],
}

# §3.5 reproduced with the open models on the FULL representative sets (2749 metabolites vs 1680
# labelled pesticides). The paper's safety conclusion reproduces directionally — magnitudes are
# compressed vs Syntelly (the open DILI model is milder/less-separating), but the direction and
# ~2x separation hold. Tables 3/4 for the top-10 candidates + 10 pesticides are bundled as
# server/data/tox/table{3,4}_*.csv; full stats in section35_open.json.
TOX_OPEN_FINDINGS = {
    # Full sets, stereo-free deduped + decontaminated (metabolites sharing a stereo-free structure
    # with the labelled set removed): 2283 metabolites vs 1680 pesticides. The raw 2749 gives
    # near-identical numbers (see "cleaning") — the metabolite/pesticide gap is a model-calibration
    # effect, not a data artifact.
    "n_metabolites": 2283, "n_pesticides": 1680,
    "median_ld50_mgkg": {"metabolites": 1782, "pesticides": 911},           # > paper's 1480; metabolites safer
    "pct_toxic_full_set": {
        "hepatotoxicity": {"metabolites": 31.9, "pesticides": 64.0},        # paper 15 vs 81; direction + ~2x hold
        "dili":           {"metabolites": 31.9, "pesticides": 64.0},
        "carcinogenicity":{"metabolites": 16.2, "pesticides": 12.9},        # ~tied
        "ames":           {"metabolites": 11.6, "pesticides": 14.5},        # ~tied (metabolites marginally safer)
    },
    "cleaning": "raw 2749 -> 2667 stereo-free-unique (dedup neutral: hepatotox 33.7->33.9%) -> 2283 "
                "after removing 384 metabolites structurally identical to labelled pesticides "
                "(hepatotox -> 31.9%, LD50 -> 1782). Duplicates/contamination move it only ~2 pts; the "
                "residual gap to the paper's 15% is the open TDC-DILI model being more liberal than Syntelly.",
    "verdict": "reproduces robustly (holds at every cleaning level): metabolites ~2x less hepatotoxic "
               "(32% vs 64%) and higher LD50 (1782 vs 911, exceeding the paper's 1480); "
               "Ames/carcinogenicity ~tied. Absolute magnitudes compressed vs Syntelly (model "
               "calibration, not data).",
    "caveats": [
        "Top-10 DMPNN candidates are pesticide-like scaffolds -> look hepatotoxic (70%); not "
        "representative of the metabolome (the full-set comparison is the fair test).",
        "Reproductive endpoint is degenerate (n=156, 88% positive) -> non-discriminative, excluded.",
        "Rat LD50 == Mouse LD50 (single open TDC LD50_Zhu acute set).",
        "Aquatic LC50 extrapolates for large glycosides (MW 870-970) -> soft applicability domain.",
    ],
}


def load_bundled_reference() -> dict:
    """Load the authors' bundled ablation/RMT reference metrics if present."""
    ref_dir = Path(get_settings().reference_dir)
    out = {}
    for p in sorted(ref_dir.glob("*.json")):
        try:
            out[p.stem] = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            pass
    return out
