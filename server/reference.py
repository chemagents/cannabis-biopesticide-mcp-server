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
