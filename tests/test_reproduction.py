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


@pytest.mark.slow
def test_candidate_fraction():
    res = models.predict_biopesticides()
    assert 0.30 <= res["candidate_fraction"] <= 0.50      # paper 40.97%
