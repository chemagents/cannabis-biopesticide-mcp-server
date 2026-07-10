"""Confidence / calibration ablation — does docking / RMT-RTE make the QSAR model *more sure*?

The paper's headline is discrimination (ROC-AUC). This module asks the orthogonal question a
deployer cares about: are the model's probabilities themselves more trustworthy once the docking
channel is added, and sharpened by RMT denoising? It reuses the exact feature ladder and base
learner of `models.qsar_ablation` (only the FEATURES change across rungs, so any calibration change
is attributable to the docking / RMT information, not the model) over the 10 random 80/20 splits.

Rungs (each = RDKit-2D structure descriptors + the docking channel below):
  structure     none (structure-only baseline)
  +dock6        the 6 global docking scores
  +raw_rte      the 390 raw residue-term energies (full per-residue docking, no denoising)
  +rmt_rte_sel  RMT-selected top-m RTE columns
  +rmt_rte_rec  RMT signal-subspace reconstruction (the paper's headline RMT-RTE)

Per split test set (averaged over the 10 splits, mean ± std):
  brier ↓  log_loss ↓  ece ↓  reliability ↓  resolution ↑  sharpness(std p) ↑
  auc ↑ (discrimination reference)  precision@0.7 ↑  coverage@0.7  precision@0.9 ↑
Plus the docking-consistency veto p_final = p_qsar × p_rmt (Section 3.3): its FPR / precision effect.

`python -m server.confidence` regenerates the bundled result
(`server/data/reference/confidence_ablation.json`) and the reliability figure
(`server/data/reference/confidence_reliability.png`). The `confidence_ablation` MCP tool serves the
bundled summary (recomputing live would cost the full RMT inner-CV over 10 splits).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from . import rmt
from .config import get_settings
from .dataset import load_dataset, load_rte, load_split
from .models import _hgb, _iid, rdkit2d_bundled

N_BINS = 10
T_HI, T_HI2 = 0.70, 0.90       # the paper's >0.7 candidate rule, plus a stricter 0.9
RUNGS = ["structure", "+dock6", "+raw_rte", "+rmt_rte_sel", "+rmt_rte_rec"]


def _fit_predict(Xtr, ytr, Xte) -> np.ndarray:
    clf = _hgb()
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def _bin_stats(p: np.ndarray, y: np.ndarray, n_bins: int = N_BINS):
    """Equal-width probability bins on [0,1] -> list of (weight, mean_p, observed_freq)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    out = []
    for k in range(n_bins):
        m = idx == k
        if m.any():
            out.append((float(m.mean()), float(p[m].mean()), float(y[m].mean())))
    return out


def confidence_metrics(p, y) -> dict:
    """Calibration + sharpness + high-confidence precision for one (proba, label) set."""
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    y = np.asarray(y, float)
    bins = _bin_stats(p, y)
    obar = float(y.mean())

    def prec_cov(t):
        hi = p >= t
        return (float(y[hi].mean()) if hi.any() else float("nan")), float(hi.mean())
    p70, c70 = prec_cov(T_HI)
    p90, c90 = prec_cov(T_HI2)
    return {
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece": float(sum(w * abs(mp - oy) for w, mp, oy in bins)),
        "reliability": float(sum(w * (mp - oy) ** 2 for w, mp, oy in bins)),   # ↓ Murphy
        "resolution": float(sum(w * (oy - obar) ** 2 for w, mp, oy in bins)),  # ↑ Murphy
        "sharpness_std": float(p.std()),
        "auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
        "precision@0.7": p70, "coverage@0.7": c70,
        "precision@0.9": p90, "coverage@0.9": c90, "base_rate": obar,
    }


def build_rungs(scaffold: bool, split_id: str):
    """Aligned train/test feature matrices for every rung of one split (+ RMT-RTE veto pieces)."""
    ds = load_dataset()
    split = load_split(scaffold=scaffold)
    s = split[split.split_id == split_id]
    rte_df, feat_cols = load_rte()
    rte_idx = {_iid(l): i for i, l in enumerate(rte_df["ligand_id"].tolist())}
    rte_arr = rte_df[feat_cols].to_numpy(float)
    dmap = {_iid(l): i for i, l in enumerate(ds.ligand_ids)}
    yq = {_iid(l): float(a) for l, a in zip(s.ligand_id, s.activity)}
    tr = [_iid(i) for i in s[s.set == "train"].ligand_id if _iid(i) in rte_idx and _iid(i) in dmap]
    te = [_iid(i) for i in s[s.set == "test"].ligand_id if _iid(i) in rte_idx and _iid(i) in dmap]
    y_tr = np.array([yq[i] for i in tr]); y_te = np.array([yq[i] for i in te])

    X2d = rdkit2d_bundled()
    X2d_tr, X2d_te = X2d[[dmap[i] for i in tr]], X2d[[dmap[i] for i in te]]
    dk_tr = np.nan_to_num(ds.dock[[dmap[i] for i in tr]])
    dk_te = np.nan_to_num(ds.dock[[dmap[i] for i in te]])
    rt_tr = np.array([rte_arr[rte_idx[i]] for i in tr])
    rt_te = np.array([rte_arr[rte_idx[i]] for i in te])

    res = rmt.rmt_filter(rt_tr, y_tr)
    sel = res.selected_idx
    mean, scale = rmt._zscore_fit(rt_tr)
    _, _, _, _, eigvecs, _, signal = rmt.rmt_prior(rmt._zscore_apply(rt_tr, mean, scale))
    rec_tr = rmt.reconstruct_signal(rt_tr, mean, scale, eigvecs, signal)
    rec_te = rmt.reconstruct_signal(rt_te, mean, scale, eigvecs, signal)

    rungs = {
        "structure":    (X2d_tr, X2d_te),
        "+dock6":       (np.hstack([X2d_tr, dk_tr]), np.hstack([X2d_te, dk_te])),
        "+raw_rte":     (np.hstack([X2d_tr, np.nan_to_num(rt_tr)]), np.hstack([X2d_te, np.nan_to_num(rt_te)])),
        "+rmt_rte_sel": (np.hstack([X2d_tr, np.nan_to_num(rt_tr[:, sel])]), np.hstack([X2d_te, np.nan_to_num(rt_te[:, sel])])),
        "+rmt_rte_rec": (np.hstack([X2d_tr, np.nan_to_num(rec_tr[:, sel])]), np.hstack([X2d_te, np.nan_to_num(rec_te[:, sel])])),
    }
    veto = {"rec_sel_tr": np.nan_to_num(rec_tr[:, sel]), "rec_sel_te": np.nan_to_num(rec_te[:, sel])}
    return rungs, veto, y_tr, y_te, res.m_opt


def confidence_ablation_live(scaffold: bool = False) -> dict:
    """Recompute the whole confidence ablation (slow: RMT inner-CV over every split)."""
    split = load_split(scaffold=scaffold)
    split_ids = sorted(split.split_id.unique())
    per_split = {r: [] for r in RUNGS}
    pooled = {r: {"p": [], "y": []} for r in RUNGS}
    veto_rows, m_opts = [], []

    for sid in split_ids:
        rungs, veto, y_tr, y_te, m_opt = build_rungs(scaffold, sid)
        m_opts.append(m_opt)
        preds = {}
        for r in RUNGS:
            Xtr, Xte = rungs[r]
            p = _fit_predict(Xtr, y_tr, Xte)
            preds[r] = p
            per_split[r].append(confidence_metrics(p, y_te))
            pooled[r]["p"].append(p.tolist()); pooled[r]["y"].append(y_te.tolist())
        p_qsar = preds["structure"]
        p_rmt = _fit_predict(veto["rec_sel_tr"], y_tr, veto["rec_sel_te"])
        p_final = p_qsar * p_rmt
        neg, pos = y_te == 0, y_te == 1

        def at(p, t=0.5):
            hi = p >= T_HI
            return (float((p[neg] >= t).mean()) if neg.any() else float("nan"),
                    float((p[pos] >= t).mean()) if pos.any() else float("nan"),
                    float(y_te[hi].mean()) if hi.any() else float("nan"), float(hi.mean()))

        def top_n_prec(score, n):     # precision of the n most-confident calls (matched coverage)
            if n <= 0:
                return float("nan")
            idx = np.argsort(score)[::-1][:n]
            return float(y_te[idx].mean())
        p_mean = 0.5 * (p_qsar + p_rmt)          # proper (symmetric) fusion, not the asymmetric veto
        f0, r0, pr0, cov0 = at(p_qsar); f1, r1, pr1, cov1 = at(p_final)
        n_flag = int((p_final >= T_HI).sum())    # how many the veto keeps at the 0.7 cut
        veto_rows.append({"fpr_qsar": f0, "fpr_veto": f1, "recall_qsar": r0, "recall_veto": r1,
                          "prec@0.7_qsar": pr0, "prec@0.7_veto": pr1,
                          "cov@0.7_qsar": cov0, "cov@0.7_veto": cov1,
                          # ranking quality (is it more than just moving the threshold?)
                          "auc_qsar": roc_auc_score(y_te, p_qsar), "auc_veto": roc_auc_score(y_te, p_final),
                          "auc_mean_fusion": roc_auc_score(y_te, p_mean),
                          "ap_qsar": average_precision_score(y_te, p_qsar),
                          "ap_veto": average_precision_score(y_te, p_final),
                          "ap_mean_fusion": average_precision_score(y_te, p_mean),
                          "brier_qsar": float(np.mean((p_qsar - y_te) ** 2)),
                          "brier_mean_fusion": float(np.mean((p_mean - y_te) ** 2)),
                          # matched-coverage: top-n_flag by each score -> isolates high-confidence reliability
                          "matched_prec_qsar": top_n_prec(p_qsar, n_flag),
                          "matched_prec_veto": top_n_prec(p_final, n_flag),
                          "matched_prec_mean_fusion": top_n_prec(p_mean, n_flag), "n_flag_veto": n_flag})

    keys = list(per_split[RUNGS[0]][0].keys())
    metrics_mean = {r: {k: float(np.nanmean([d[k] for d in per_split[r]])) for k in keys} for r in RUNGS}
    metrics_std = {r: {k: float(np.nanstd([d[k] for d in per_split[r]])) for k in keys} for r in RUNGS}

    from scipy.stats import wilcoxon
    base = {k: np.array([per_split["structure"][i][k] for i in range(len(split_ids))]) for k in keys}
    paired = {}
    for r in RUNGS[1:]:
        d = {}
        for k in ("brier", "log_loss", "ece", "resolution", "precision@0.7", "auc"):
            cur = np.array([per_split[r][i][k] for i in range(len(split_ids))])
            try:
                pval = float(wilcoxon(cur, base[k]).pvalue)
            except ValueError:
                pval = float("nan")
            d[k] = {"delta_vs_structure": float(np.nanmean(cur - base[k])), "wilcoxon_p": pval}
        paired[r] = d
    veto_mean = {k: float(np.nanmean([row[k] for row in veto_rows])) for k in veto_rows[0]}

    return {
        "protocol": f"{len(split_ids)} random 80/20 splits (split_registry.csv); HGB base learner; "
                    "only the docking/RMT features change across rungs",
        "base_learner": "HistGradientBoostingClassifier (identical to qsar_ablation)",
        "n_splits": len(split_ids), "n_bins_ece": N_BINS, "m_opt_mean": float(np.mean(m_opts)),
        "rungs": RUNGS, "metrics_mean": metrics_mean, "metrics_std": metrics_std,
        "paired_vs_structure": paired, "veto_mean": veto_mean,
        "_pooled": pooled,     # stripped before bundling; used only for the reliability figure
    }


def render_reliability(pooled: dict, png_path: Path) -> None:
    """Reliability curves + predicted-probability histograms for 3 key rungs (pooled over splits)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = ["structure", "+dock6", "+rmt_rte_rec"]
    colors = {"structure": "#9aa0a6", "+dock6": "#4285f4", "+rmt_rte_rec": "#ea4335"}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))
    axL.plot([0, 1], [0, 1], "--", color="#888", lw=1, label="perfect calibration")
    for r in show:
        p = np.array([v for chunk in pooled[r]["p"] for v in chunk])
        y = np.array([v for chunk in pooled[r]["y"] for v in chunk])
        bins = _bin_stats(p, y, N_BINS)
        ece = sum(w * abs(mp - oy) for w, mp, oy in bins)
        brier = np.mean((np.clip(p, 1e-7, 1 - 1e-7) - y) ** 2)
        axL.plot([mp for _, mp, _ in bins], [oy for _, _, oy in bins], "o-",
                 color=colors[r], lw=2, ms=5, label=f"{r}  (ECE={ece:.3f}, Brier={brier:.3f})")
        axR.hist(p, bins=25, range=(0, 1), histtype="step", lw=2, color=colors[r], label=r)
    axL.set_xlabel("mean predicted probability"); axL.set_ylabel("observed active fraction")
    axL.set_title("Reliability — curves overlap: docking/RMT-RTE barely move HGB calibration")
    axL.legend(fontsize=8, loc="upper left"); axL.set_xlim(0, 1); axL.set_ylim(0, 1); axL.grid(alpha=0.25)
    axR.axvline(T_HI, color="#333", ls=":", lw=1); axR.set_xlabel("predicted probability")
    axR.set_ylabel("count"); axR.set_title("Sharpness — near-identical probability distributions")
    axR.legend(fontsize=8); axR.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(png_path, dpi=150, bbox_inches="tight")


def _load_blend_reference() -> dict:
    """Authors' bundled split_00 ablation: blend/DMPNN/HGB ROC-AUC & PR-AUC per feature set.

    Key reconciliation: RMT-RTE-rec lifts the *blend* and *DMPNN* discrimination even though it does
    not help the HGB analogue used here — so the paper's RMT benefit is specific to the graph-net stack.
    """
    import csv as _csv
    path = Path(get_settings().reference_dir) / "ablation__random_split__metrics_summary.csv"
    if not path.exists():
        return {}
    fmap = {"rdkit2d": "structure", "rdkit2d_dock2": "+dock2", "rdkit2d_raw_rte": "+raw_rte",
            "rdkit2d_rmt_sel": "+rmt_rte_sel", "rdkit2d_rmt_rec": "+rmt_rte_rec"}
    out: dict[str, dict] = {}
    for r in _csv.DictReader(path.read_text().splitlines()):
        fs = fmap.get(r["feature_set"])
        if fs is None:
            continue
        out.setdefault(fs, {})[r["model"]] = {"roc_auc": round(float(r["roc_auc"]), 4),
                                              "pr_auc": round(float(r["pr_auc"]), 4)}
    return out


def load_confidence_ablation() -> dict | None:
    """Load the bundled confidence-ablation result (reference_dir/confidence_ablation.json)."""
    path = Path(get_settings().reference_dir) / "confidence_ablation.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> None:
    import warnings
    from scipy.stats import ConstantInputWarning
    warnings.filterwarnings("ignore", category=ConstantInputWarning)   # constant RTE columns -> undefined ρ
    ref_dir = Path(get_settings().reference_dir)
    ref_dir.mkdir(parents=True, exist_ok=True)
    res = confidence_ablation_live(scaffold=False)
    pooled = res.pop("_pooled")
    render_reliability(pooled, ref_dir / "confidence_reliability.png")
    # scaffold companion (single Bemis-Murcko split = the novel-chemistry deployment regime)
    try:
        sc = confidence_ablation_live(scaffold=True)
        sc.pop("_pooled", None)
        res["scaffold_companion"] = {"metrics_mean": sc["metrics_mean"], "veto_mean": sc["veto_mean"],
                                     "n_splits": sc["n_splits"],
                                     "note": "single Bemis-Murcko scaffold-disjoint split (no error bars); "
                                             "the deployment regime for scoring novel C. sativa chemistry"}
    except Exception as exc:  # noqa: BLE001
        res["scaffold_companion"] = {"error": str(exc)}

    res["dmpnn_blend_reference"] = {
        "source": "authors' bundled split_00 ablation (ablation__random_split__metrics_summary.csv)",
        "note": "RMT-RTE-rec lifts the DMPNN/blend (ROC-AUC 0.9343->0.9415, PR-AUC 0.9485->0.9543) "
                "even though it does not help the torch-free HGB analogue used above — the RMT confidence "
                "benefit is specific to the graph-net stack.",
        "by_feature_set": _load_blend_reference(),
    }
    v, sv = res["veto_mean"], res.get("scaffold_companion", {}).get("veto_mean", {})
    res["verdict"] = (
        "For the open torch-free HGB analogue, adding docking / RMT-RTE does NOT improve confidence in "
        "either regime. (1) As features: the 217 RDKit-2D descriptors already saturate — Brier/ECE/AUC "
        "are flat-to-worse on random splits and on the scaffold split (structure AUC is the best rung in "
        "both). (2) As the p_QSAR x p_RMT-RTE veto: the precision@0.7 / FPR gains are a thresholding "
        f"artifact — coverage collapses ({v['cov@0.7_qsar']:.2f}->{v['cov@0.7_veto']:.2f} random) while "
        f"ranking degrades (AUC {v['auc_qsar']:.3f}->{v['auc_veto']:.3f}); at matched coverage the "
        f"structure model alone is more precise ({v['matched_prec_qsar']:.3f} vs veto "
        f"{v['matched_prec_veto']:.3f}), and symmetric mean-fusion is no better. The scaffold split "
        f"repeats this (structure matched-cov precision {sv.get('matched_prec_qsar', float('nan')):.3f} vs "
        f"veto {sv.get('matched_prec_veto', float('nan')):.3f}). RMT-RTE is a weak (~0.80 AUC) standalone "
        "signal that only dilutes the strong structure model. Where RMT genuinely helps is the DMPNN/blend "
        "(see dmpnn_blend_reference), not this analogue: the paper's confidence gain lives in the graph net."
    )
    (ref_dir / "confidence_ablation.json").write_text(json.dumps(res, indent=2))

    mm, ms = res["metrics_mean"], res["metrics_std"]
    print("\n=== Confidence ablation (mean over %d random splits) ===" % res["n_splits"])
    for r in RUNGS:
        m = mm[r]
        print(f"{r:<13} brier {m['brier']:.4f}±{ms[r]['brier']:.4f}  ll {m['log_loss']:.4f}  "
              f"ece {m['ece']:.4f}  resol {m['resolution']:.4f}  sharp {m['sharpness_std']:.3f}  "
              f"auc {m['auc']:.4f}  P@.7 {m['precision@0.7']:.4f} (cov {m['coverage@0.7']:.3f})  "
              f"P@.9 {m['precision@0.9']:.4f}")
    print("\n=== Paired Δ vs structure (Wilcoxon p, 10 splits) ===")
    for r in RUNGS[1:]:
        d = res["paired_vs_structure"][r]
        print(f"{r:<13} Δbrier {d['brier']['delta_vs_structure']:+.4f} p={d['brier']['wilcoxon_p']:.4f}  "
              f"Δece {d['ece']['delta_vs_structure']:+.4f} p={d['ece']['wilcoxon_p']:.4f}  "
              f"Δresol {d['resolution']['delta_vs_structure']:+.4f} p={d['resolution']['wilcoxon_p']:.4f}  "
              f"ΔP@.7 {d['precision@0.7']['delta_vs_structure']:+.4f} p={d['precision@0.7']['wilcoxon_p']:.4f}")
    def veto_report(v, tag):
        print(f"\n=== Docking veto / fusion — {tag} ===")
        print(f"FPR {v['fpr_qsar']:.4f}->{v['fpr_veto']:.4f}  recall {v['recall_qsar']:.4f}->{v['recall_veto']:.4f}  "
              f"P@0.7 {v['prec@0.7_qsar']:.4f}->{v['prec@0.7_veto']:.4f}  cov@0.7 {v['cov@0.7_qsar']:.4f}->{v['cov@0.7_veto']:.4f}")
        print(f"AUC struct {v['auc_qsar']:.4f} | veto {v['auc_veto']:.4f} | mean-fusion {v['auc_mean_fusion']:.4f}")
        print(f"matched-cov precision  struct {v['matched_prec_qsar']:.4f} | veto {v['matched_prec_veto']:.4f} | "
              f"mean-fusion {v['matched_prec_mean_fusion']:.4f}")
    veto_report(res["veto_mean"], "random (in-distribution)")
    if isinstance(res.get("scaffold_companion"), dict) and "veto_mean" in res["scaffold_companion"]:
        veto_report(res["scaffold_companion"]["veto_mean"], "scaffold (novel chemistry)")
        sm = res["scaffold_companion"]["metrics_mean"]
        print("\n=== Scaffold ladder (single split) ===")
        for r in RUNGS:
            m = sm[r]
            print(f"{r:<13} brier {m['brier']:.4f}  ece {m['ece']:.4f}  auc {m['auc']:.4f}  P@.7 {m['precision@0.7']:.4f}")


if __name__ == "__main__":
    main()
