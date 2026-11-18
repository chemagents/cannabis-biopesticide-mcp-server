"""Reproduction tests for the Cannabis biopesticide QSAR paper.

    uv run pytest tests -v                  # all
    uv run pytest tests -v -m "not slow"    # fast (docking + RMT math) only
"""
import numpy as np
import pytest

from server import docking, models
from server.dataset import DOCK_COLS, load_dataset, load_rte

ds = load_dataset()


def test_dataset_shape():
    assert ds.n == 5920
    assert int(ds.active_mask.sum()) == 1680
    assert int(ds.inactive_mask.sum()) == 1491
    assert int(ds.metabolite_mask.sum()) == 2749
    assert len(DOCK_COLS) == 6


def test_docking_five_negative_or28_positive():
    res = docking.active_vs_inactive(ds)
    neg = [r for r in res["per_protein"] if r["expected_trend"]]
    assert len(neg) == 5                                  # 5 targets: actives bind stronger
    assert res["anomalous_protein"] == "OR28"             # OR28 opposite
    assert all(r["significant"] for r in res["per_protein"])   # all BH p < 0.001


def test_metabolite_docking_range():
    res = docking.active_vs_inactive(ds)
    lo, hi = res["metabolite_median_range"]
    assert -7.5 <= lo and hi <= -5.0                      # paper: -7.2 .. -5.2


@pytest.mark.slow
def test_rmt_lambda_plus_and_m_opt():
    r = models.rmt_selection(scaffold=False)
    assert abs(r["lambda_plus"] - 1.938) < 0.01           # Marchenko-Pastur threshold, exact
    assert 145 <= r["m_opt"] <= 175                       # paper m_opt = 161
    assert abs(r["inner_auc"] - 0.747) < 0.02


@pytest.mark.slow
def test_qsar_high_quality():
    ab = models.qsar_ablation(scaffold=False)              # authors' exact 217 descriptors
    best = max(m["roc_auc"] for m in ab["results"].values())
    assert best >= 0.92                                   # paper DMPNN-SD 0.9283
    cb = models.cb_sd_rte()["all6_rte"]["roc_auc"]
    assert 0.75 <= cb <= 0.83                             # paper CB-SD ALL6 0.802


@pytest.mark.slow
def test_docking_veto_reduces_fpr():
    v = models.docking_veto()
    assert v["fpr_after_veto"] < 0.08                     # paper 4.92%
    assert v["fpr_reduction_pct"] >= 40                   # paper ~60%


def test_confidence_ablation_bundled_and_honest():
    """The bundled confidence ablation ships and records the honest (adversarial) conclusion."""
    from server import confidence
    res = confidence.load_confidence_ablation()
    assert res is not None                                        # confidence_ablation.json is bundled
    mm = res["metrics_mean"]
    # structure descriptors saturate: adding RMT-RTE as features does not beat structure on ROC-AUC
    assert mm["+rmt_rte_rec"]["auc"] <= mm["structure"]["auc"] + 1e-6
    # the veto is a thresholding trade, not a ranking/confidence gain
    v = res["veto_mean"]
    assert v["auc_veto"] <= v["auc_qsar"]                         # veto ranks no better than structure
    assert v["matched_prec_qsar"] >= v["matched_prec_veto"]       # structure purer at matched coverage
    # RMT-RTE does not help the DMPNN/stack either — team's paired k-fold CV: rmt_rte_rec LOWERS blend ROC
    kf = res["dmpnn_stack_rmt_kfold"]["paired_kfold"]["rmt_rte_rdkit_only"]["by_split"]
    assert kf["random"]["blend"]["rmt_rte_rec_roc"] < kf["random"]["blend"]["baseline_roc"]
    assert kf["scaffold"]["blend"]["rmt_rte_rec_roc"] < kf["scaffold"]["blend"]["baseline_roc"]
    # and RELIABILITY (OOF re-run): +rmt is no better at matched coverage — precision@0.7 down, Brier up
    rel = confidence._load_stack_reliability()["by_split"]
    for sp in ("random", "scaffold"):
        bl = rel[sp]["blend"]
        assert bl["precision@0.7"][1] <= bl["precision@0.7"][0] + 1e-9   # +rmt precision no higher
        assert bl["brier"][1] >= bl["brier"][0] - 1e-9                   # +rmt Brier no lower (worse/equal)


@pytest.mark.slow
def test_candidate_fraction():
    res = models.predict_biopesticides()
    # headline_fraction is the paper-comparable number for whichever backend is active:
    # DMPNN-SD -> exact 40.97% (raw prob>0.7); HGB fallback -> ~34% (prob>0.7 within Tanimoto AD).
    assert 0.30 <= res["headline_fraction"] <= 0.50       # paper 40.97%
    assert res["backend"].startswith(("dmpnn", "hgb"))
