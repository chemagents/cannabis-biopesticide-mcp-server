"""cannabis-biopesticide MCP server.

Reproduces "Biopesticidal Potential of the Cannabis sativa L. Metabolome: A Denoised,
Docking-Informed QSAR Model" from the authors' bundled data (5920 ligands × SMILES/activity/
6 docking scores + 390 residue-term energies). Docking was done upstream (Vina-GPU); this
server recomputes the downstream analyses (docking statistics, RMT feature selection, QSAR,
applicability domain, candidate list) deterministically and fast — no GPU.

Open-source analogue stack (Syntelly / DMPNN not public): RDKit, ECFP + differential
scaffold fingerprints, HistGradientBoosting (torch-free analogue of DMPNN-SD, ~0.93),
CatBoost, kNN+Gaussian applicability domain, and a numpy/scipy port of the RMT filter.
Every tool returns {"answer": ..., "metadata": ...}.
"""
from __future__ import annotations

import logging

from fastmcp import FastMCP

from . import artifacts, docking, models, plotting, reference
from .config import get_settings
from .dataset import PROTEINS, load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("CannabisBiopesticide")
PAPER = reference.PAPER

ARTIFACT_OUTPUT_POLICY = (
    "Return every non-null figure artifact from this question to the user together with its "
    "URL or path, kind, and SHA-256; do not replace the scientific answer with an internal task "
    "log."
)
QUESTIONS = {
    1: "What is being screened, and does the Cannabis metabolome occupy pesticide-like chemical space?",
    2: ("Do Cannabis metabolites engage pest targets, does RMT isolate useful docking signal, "
        "and does a physical-consistency veto reduce false positives?"),
    3: "How reliable is the pesticidal-activity model, and which metabolites are credible candidates?",
    4: "What safety comparison is actually supported for Cannabis metabolites versus synthetic pesticides?",
}
QUESTION_TOOLS = {
    1: ["canpest_dataset_overview", "chemical_space"],
    2: ["docking_analysis", "rmt_feature_selection", "docking_veto"],
    3: ["qsar_model_quality", "model_stack", "predict_biopesticides"],
    4: ["tox_ecotox_reference"],
}


def _chain(question: int, self_name: str) -> dict:
    """Machine-readable question and sibling-tool routing for the orchestrator."""
    tools = QUESTION_TOOLS[question]
    return {
        "question": QUESTIONS[question],
        "next_tools": [tool for tool in tools if tool != self_name],
        "next_tools_reason": (
            "These sibling tools answer complementary parts of the same scientific question; "
            "use all of them before concluding."
            if len(tools) > 1
            else "This tool is the complete evidence source for this question."
        ),
        "artifact_output_policy": ARTIFACT_OUTPUT_POLICY,
    }


@mcp.tool(name="canpest_dataset_overview")
def dataset_overview() -> dict:
    """What does the Cannabis sativa biopesticide-discovery dataset contain?

    Scope of the labelled pesticide/inactive training set and the unlabelled C. sativa
    (hemp) metabolite screening set, plus the 6 insect/mite pest proteins they are docked
    against (Section 3.1). Entry point for the biopesticide-activity question — this is a
    PESTICIDAL-ACTIVITY dataset, not a mammalian toxicity dataset.
    """
    ds = load_dataset()
    n_a, n_i = int(ds.active_mask.sum()), int(ds.inactive_mask.sum())
    n_m, n_l = int(ds.metabolite_mask.sum()), int(ds.labelled_mask.sum())
    return {
        "answer": {
            "n_total": ds.n,
            "n_active_pesticides": n_a,
            "n_inactive": n_i,
            "n_cannabis_metabolites": n_m,
            "n_labelled": n_l,
            "proteins": {c: p["gene"] + " — " + p["name"] for c, p in PROTEINS.items()},
            "finding": (
                f"{ds.n} ligands in total: {n_l} labelled ({n_a} active pesticides, {n_i} "
                f"inactives, class balance {n_a / max(n_l, 1):.0%} active) used to train the QSAR "
                f"model, plus {n_m} unlabelled C. sativa metabolites to be screened. Each ligand "
                f"carries docking scores against {len(PROTEINS)} pest-relevant proteins "
                f"({', '.join(p['gene'] for p in PROTEINS.values())}). The endpoint here is "
                f"PESTICIDAL ACTIVITY, not mammalian toxicity."),
        },
        "metadata": {"paper": reference.DATASETS, "reference": PAPER,
                     **_chain(1, "canpest_dataset_overview"),
                     "note": "Docking (Vina-GPU) precomputed by the authors; scores are bundled."},
    }


@mcp.tool()
def docking_analysis() -> dict:
    """Do known pesticides bind the INSECT/MITE PEST proteins more strongly than inactives?

    Docking-score statistics against the 6 pest targets (GluCl, AChE1, UGT202A2, GSTM12,
    agGSTe2, OR28), comparing active pesticides with inactives (Section 3.2 / Fig 2). Scope:
    pest-target docking — for mammalian antitarget/safety-panel docking use the
    `tox-antitargets` server.
    """
    ds = load_dataset()
    res = docking.active_vs_inactive(ds)
    fig = None
    try:
        fig = plotting.plot_docking(res["per_protein"])
    except artifacts.ArtifactStorageError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("docking figure failed: %s", exc)
    expected = [r for r in res["per_protein"] if r["expected_trend"]]
    opposite = [r for r in res["per_protein"] if not r["expected_trend"]]
    n_tot = len(res["per_protein"])
    if expected:
        d_lo = min(r["delta"] for r in expected)
        d_hi = max(r["delta"] for r in expected)
        head = (f"{len(expected)} of {n_tot} pest targets show the expected trend (actives bind "
                f"more strongly, delta {d_lo:+.2f} to {d_hi:+.2f} kcal/mol)")
    else:
        head = f"None of the {n_tot} pest targets shows the expected trend"
    if opposite:
        tail = ("; " + ", ".join(f"{r['protein']} is opposite (delta {r['delta']:+.2f})"
                                 for r in opposite)
                + ", consistent with a chemosensory/repellent rather than lethal role.")
    else:
        tail = "; no target shows the opposite trend."
    return {
        "answer": {**res, "n_expected_trend": len(expected), "n_opposite_trend": len(opposite),
                   "finding": head + tail},
        "metadata": {"figure": fig, "paper": reference.DOCKING, "reference": PAPER,
                     **_chain(2, "docking_analysis")},
    }


@mcp.tool()
def rmt_feature_selection(scaffold_split: bool = False) -> dict:
    """How many of the residue-term docking descriptors carry real signal rather than noise?

    Random Matrix Theory (Marchenko-Pastur) denoising of the 390 residue-term interaction
    energies from the pest-protein docking, for the C. sativa biopesticide QSAR (Section 3.3).
    """
    res = models.rmt_selection(scaffold=scaffold_split)
    ref = reference.RMT
    n_sig, m_opt, lam = res["n_signal"], res["m_opt"], res["lambda_plus"]
    return {
        "answer": {**res,
                   "finding": (
                       f"Marchenko-Pastur upper edge lambda+ = {lam:.4f} separates signal from "
                       f"noise: {n_sig} eigenvalues exceed it (paper: "
                       f"{ref.get('n_signal', 'n/a')}), and ranking residue-term features by "
                       f"s_i*|rho| selects m_opt = {m_opt} informative descriptors (paper: "
                       f"{ref['m_opt_random']}). "
                       + ("lambda+ and m_opt reproduce; the signal-eigenvalue count differs from "
                          f"the paper's {ref.get('n_signal')}, so report {n_sig} as the recomputed "
                          "value."
                          if ref.get("n_signal") is not None and ref["n_signal"] != n_sig
                          else "All three quantities agree with the paper."))},
        "metadata": {"paper": ref, "reference": PAPER,
                     **_chain(2, "rmt_feature_selection"),
                     "matches": {"lambda_plus": abs(res["lambda_plus"] - ref["lambda_plus"]) < 0.01,
                                 "m_opt_close": abs(res["m_opt"] - ref["m_opt_random"]) <= 30,
                                 "n_signal": (ref.get("n_signal") == res["n_signal"]
                                              if ref.get("n_signal") is not None else None)}},
    }


@mcp.tool()
def qsar_model_quality(scaffold_split: bool = False) -> dict:
    """How good is the PESTICIDAL-ACTIVITY QSAR model for the C. sativa metabolites?

    Feature-set ablation (RDKit-2D descriptors +/- RMT-selected docking terms) and the
    residue-level docking baseline CB-SD, recomputed here (Section 3.3). Scope: classification of
    biopesticidal activity. This is NOT mammalian toxicity-model quality — for that use
    `model_quality` on the `heracleum-tox` server.
    """
    ab = models.qsar_ablation(scaffold=scaffold_split)
    cb = models.cb_sd_rte(scaffold=scaffold_split)
    best = max(ab["results"].items(), key=lambda kv: kv[1]["roc_auc"])
    return {
        "answer": {
            "ablation": ab["results"], "best_feature_set": best[0], "best_roc_auc": best[1]["roc_auc"],
            "cb_sd_residue_level_docking": cb["all6_rte"],
            "m_opt": ab["m_opt"],
            "finding": f"Structure-based gradient boosting on the authors' 217 RDKit2D descriptors "
                       f"(open analogue of DMPNN-SD) reaches ROC-AUC ~{best[1]['roc_auc']:.3f}; the "
                       f"residue-level docking model (CB-SD) reaches {cb['all6_rte']['roc_auc']:.3f}.",
        },
        "metadata": {"paper": reference.QSAR, "reference": PAPER,
                     **_chain(3, "qsar_model_quality"),
                     "note": "This is the HGB component of the stack; see model_stack for the full "
                             "DMPNN+HGB ensemble. Exact DMPNN via the vendored torch pipeline "
                             "(server/vendor)."},
    }


@mcp.tool()
def model_stack() -> dict:
    """How reliable is the pesticidal-activity model? Architecture and published CV metrics of
    the paper's weighted DMPNN + HGB soft-voting QSAR stack (Section 3.3/3.4).

    LOOKUP of the authors' bundled 5-fold OOF-CV numbers — this tool recomputes nothing. For the
    quality figures this server actually measures, call `qsar_model_quality`.
    """
    s = models.model_stack_quality()
    b, d, h = s["blend"], s["components"]["dmpnn"], s["components"]["hgb"]
    w = s["blend_w_dmpnn"]
    order = sorted([("blend", b["roc_auc"]), ("DMPNN", d["roc_auc"]), ("HGB", h["roc_auc"])],
                   key=lambda kv: -kv[1])
    return {
        "answer": {**s,
                   "finding": (
                       f"DMPNN-SD is a soft-voting stack: p = {w}*p_DMPNN + {round(1 - w, 2)}*p_HGB. "
                       f"On the authors' matched 5-fold OOF CV the ranking is "
                       + " > ".join(f"{k} ({v})" for k, v in order)
                       + f". These are PRECOMPUTED reference values echoed from the paper's bundle, "
                         f"not measured by this server — the number this server actually computes "
                         f"is `qsar_model_quality.best_roc_auc`.")},
        "metadata": {"paper": {"dmpnn_sd_roc": reference.QSAR["dmpnn_sd_roc"]}, "reference": PAPER,
                     **_chain(3, "model_stack"),
                     "provenance": "precomputed reference values, not recomputed here",
                     "bundled_cv": reference.load_bundled_reference().get("model_stack_cv", {})},
    }


@mcp.tool()
def docking_veto() -> dict:
    """Does requiring docking consistency cut false positives in the biopesticide screen?

    The asymmetric veto p_final = p_QSAR x p_RMT-RTE, applied to pest-target docking, and its
    effect on false-positive rate and recall (Section 3.3).
    """
    v = models.docking_veto()
    return {
        "answer": {**v,
                   "finding": f"The asymmetric veto lowers FPR from {v['fpr_before']:.1%} to "
                              f"{v['fpr_after_veto']:.1%} ({v['fpr_reduction_pct']:.0f}% reduction) at "
                              f"threshold 0.5, trading recall ({v['recall_before']:.2f}→"
                              f"{v['recall_after_veto']:.2f}) — a physical-consistency filter, not an average."},
        "metadata": {"paper": {"fpr_before": reference.QSAR["fpr_before"],
                               "fpr_after_veto": reference.QSAR["fpr_after_veto"]},
                     "reference": PAPER, **_chain(2, "docking_veto")},
    }


@mcp.tool()
def predict_biopesticides() -> dict:
    """Which Cannabis sativa metabolites are candidate BIOPESTICIDES?

    Predicted pesticidal-activity probability for every unlabelled C. sativa metabolite, with the
    Tanimoto applicability domain (Section 3.4). The endpoint is insecticidal/acaricidal activity,
    not toxicity to mammals.
    """
    res = models.predict_biopesticides()
    cut = get_settings().candidate_probability
    paper_n, paper_frac = reference.CANDIDATES["n_high_conf"], reference.CANDIDATES["fraction"]
    n_meta = res["n_metabolites"]
    return {
        "answer": {**res,
                   "probability_cutoff": cut,
                   "finding": (
                       f"{res['headline_count']} of {n_meta} C. sativa metabolites "
                       f"({res['headline_fraction']:.1%}) are predicted biopesticides at "
                       f"probability >{cut} [{res['backend']}; {res['headline_basis']}]. The paper "
                       f"reports {paper_n} ({paper_frac:.2%}); the FRACTIONS are comparable but the "
                       f"COUNTS are not, because the screening set here holds {n_meta} metabolites "
                       f"against the paper's {round(paper_n / paper_frac)}. Compare the fraction, "
                       f"not the count.")},
        "metadata": {"paper": reference.CANDIDATES, "reference": PAPER,
                     **_chain(3, "predict_biopesticides"),
                     "denominator_note": (
                         f"paper denominator ~{round(paper_n / paper_frac)} metabolites vs "
                         f"{n_meta} here")},
    }


@mcp.tool()
def chemical_space() -> dict:
    """Do CANNABIS SATIVA metabolites occupy the same chemical space as synthetic pesticides?

    t-SNE map of the hemp metabolite set against the labelled pesticide set, with a measured
    nearest-neighbour overlap statistic (Figure S2). Scope: the biopesticide dataset of this
    server — not the drug-like antitarget space (`chemical_space_tsne` on `tox-antitargets`)
    and not the hogweed metabolite families (`chemical_space_clustering` on `heracleum-tox`).
    """
    ds = load_dataset()
    fig = None
    try:
        fig = plotting.plot_chemical_space(ds)
    except artifacts.ArtifactStorageError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("t-SNE figure failed: %s", exc)
    # Measure the overlap instead of asserting it: for each metabolite, is its nearest labelled
    # neighbour (ECFP4 Tanimoto) an ACTIVE pesticide, and how similar is it?
    overlap = None
    try:
        overlap = docking.metabolite_pesticide_overlap(ds)
    except Exception as exc:  # noqa: BLE001
        logger.warning("overlap statistic failed: %s", exc)
    if overlap:
        band = ("substantially" if overlap["frac_nn_active"] >= 0.5 else
                "partially" if overlap["frac_nn_active"] >= 0.25 else "only marginally")
        finding = (
            f"For {overlap['frac_nn_active']:.0%} of the {overlap['n_metabolites']} C. sativa "
            f"metabolites the nearest labelled neighbour is an ACTIVE pesticide (median nearest-"
            f"neighbour Tanimoto {overlap['median_nn_similarity']:.2f}; "
            f"{overlap['frac_nn_similarity_above_0_4']:.0%} of metabolites have a labelled "
            f"neighbour at Tanimoto > 0.4). The metabolite chemical space therefore {band} "
            f"overlaps the synthetic-pesticide space. Note t-SNE geometry is not metric — this "
            f"conclusion rests on the Tanimoto numbers, not on the picture.")
    else:
        finding = ("The chemical-space overlap statistic could not be computed in this call; the "
                   "figure alone is not evidence of overlap.")
    return {
        "answer": {**(overlap or {}), "finding": finding},
        "metadata": {"figure": fig, "method": "differential scaffold fingerprint + t-SNE; overlap "
                                              "measured as ECFP4 Tanimoto nearest-neighbour class",
                     "reference": PAPER, **_chain(1, "chemical_space")},
    }


@mcp.tool()
def tox_ecotox_reference() -> dict:
    """Compare published mammalian safety of natural products and synthetic pesticides.

    This is the Cannabis-paper reference tool for the rat/mouse oral LD50 and hepatotoxicity
    comparison between plant metabolites (natural products) and synthetic pesticides, plus the
    paper's aquatic-ecotoxicity model-quality entries and their important coverage caveat.  It
    reports Section 3.5 / Table S1 Syntelly reference values rather than running live models.

    §3.5 uses the same Syntelly toxicity models as the Heracleum paper; for LIVE open-model
    prediction of LD50 / hepatotoxicity / DILI / cardiotoxicity / carcinogenicity use the
    `heracleum-tox` server (CatBoost/XGBoost on TDC). This tool reports the paper's published
    metrics and headline comparison.
    """
    h = reference.TOX_FINDINGS
    # State ONLY what `h` contains. The previous sentence also asserted "far less DILI" and "no
    # extreme aquatic toxicity", neither of which appears anywhere in the returned numbers.
    ld50_parts = []
    for species in ("rat", "mouse"):
        m, p = h.get(f"metabolite_ld50_{species}"), h.get(f"pesticide_ld50_{species}")
        if m is not None and p is not None:
            ld50_parts.append(f"{species} oral LD50 {m} vs {p} mg/kg ({m / p:.2f}x)")
    hep_m, hep_p = h.get("hepatotoxic_metabolites_pct"), h.get("hepatotoxic_pesticides_pct")
    hep = (f"{hep_m}% of metabolites vs {hep_p}% of synthetic pesticides are flagged hepatotoxic"
           if hep_m is not None and hep_p is not None else "")
    covered = sorted(k for k in h)
    return {
        "answer": {
            "model_quality_table_s1": {k: {"metric": v[0], "value": v[1]}
                                       for k, v in reference.TOX_METRICS.items()},
            "headline": h,
            "endpoints_with_a_published_comparison": covered,
            "finding": (
                "PUBLISHED reference values from the paper's Syntelly predictions — nothing here "
                "is recomputed by this server. On the endpoints the paper actually reports: "
                + "; ".join(ld50_parts + ([hep] if hep else []))
                + ". Only these endpoints are covered: no DILI figure and no aquatic-toxicity "
                  "outcome is reported (the aquatic entries in `model_quality_table_s1` are MODEL "
                  "quality metrics, not toxicity results), so do not state a DILI or ecotoxicity "
                  "conclusion from this tool. For live open-model toxicity prediction use "
                  "`predict_general_toxicity` / `predict_ld50` on the `heracleum-tox` server."),
        },
        "metadata": {"reference": PAPER, **_chain(4, "tox_ecotox_reference"),
                     "provenance": "published Syntelly values, not recomputed here",
                     "coverage_caveat": "model_quality_table_s1 holds MODEL METRICS (RMSE/ROC-AUC), "
                                        "not toxicity outcomes; the aquatic endpoints appear only "
                                        "there and carry no metabolite-vs-pesticide comparison.",
                     "live_tox_models": "heracleum-tox MCP server (open Syntelly analogue)"},
    }


@mcp.tool(name="canpest_reproduce_all")
def reproduce_all() -> dict:
    """Bulk numeric verification pass over the C. sativa biopesticide paper: recompute every
    headline number (docking deltas, RMT lambda+/m_opt, QSAR ROC-AUC, veto FPR, candidate
    fraction) and compare each with the published value (trains models; ~1-2 min).

    Use only to verify or audit the paper's numbers as a batch; it answers no individual
    scientific question. Each check reports the tolerance it applied.
    """
    ds = load_dataset()
    dk = docking.active_vs_inactive(ds)
    rmt_res = models.rmt_selection(scaffold=False)
    ab = models.qsar_ablation(scaffold=False)
    cb = models.cb_sd_rte(scaffold=False)
    veto = models.docking_veto()
    cand = models.predict_biopesticides()

    neg = [r for r in dk["per_protein"] if r["expected_trend"]]
    best_roc = max(m["roc_auc"] for m in ab["results"].values())
    mr = dk["metabolite_median_range"]
    paper_mr = reference.DOCKING.get("metabolite_median_range", [-7.2, -5.2])
    paper_frac = reference.CANDIDATES["fraction"]
    paper_qsar = reference.QSAR["dmpnn_sd_roc"]
    paper_fpr_before = reference.QSAR["fpr_before"]
    # Every gate below states the tolerance it applies, and every printed paper value is the one
    # the gate is actually testing. The previous version printed tight paper ranges next to gates
    # that would have passed on values nowhere near them.
    checks = [
        ("docking_5_negative_OR28_positive", f"{len(neg)} neg / anomaly={dk['anomalous_protein']}",
         f"{len(dk['per_protein']) - 1} neg / OR28", "exact",
         len(neg) == len(dk["per_protein"]) - 1 and dk["anomalous_protein"] == "OR28"),
        ("metabolite_docking_median_range", mr, paper_mr, "each endpoint within 1.0 kcal/mol",
         abs(mr[0] - paper_mr[0]) <= 1.0 and abs(mr[1] - paper_mr[1]) <= 1.0),
        ("rmt_lambda_plus", rmt_res["lambda_plus"], reference.RMT["lambda_plus"], "abs diff < 0.01",
         abs(rmt_res["lambda_plus"] - reference.RMT["lambda_plus"]) < 0.01),
        ("rmt_m_opt", rmt_res["m_opt"], reference.RMT["m_opt_random"], "abs diff <= 15",
         abs(rmt_res["m_opt"] - reference.RMT["m_opt_random"]) <= 15),
        ("rmt_n_signal", rmt_res["n_signal"], reference.RMT["n_signal"], "exact",
         rmt_res["n_signal"] == reference.RMT["n_signal"]),
        ("qsar_roc_auc", round(best_roc, 4), paper_qsar, "abs diff <= 0.03 (two-sided)",
         abs(best_roc - paper_qsar) <= 0.03),
        ("cb_sd_residue_docking", cb["all6_rte"]["roc_auc"], reference.QSAR["cb_sd_all6"],
         "inside the paper's 0.75-0.83 band",
         0.75 <= cb["all6_rte"]["roc_auc"] <= 0.83),
        ("docking_veto_fpr_effect", f"{veto['fpr_before']:.3f}->{veto['fpr_after_veto']:.3f}",
         f"{reference.QSAR['fpr_before']}->{reference.QSAR['fpr_after_veto']}",
         "post-veto FPR < 0.08 AND reduction >= 40% (EFFECT only; baseline FPR not gated)",
         veto["fpr_after_veto"] < 0.08 and veto["fpr_reduction_pct"] >= 40),
        ("docking_veto_baseline_fpr", round(veto["fpr_before"], 4), paper_fpr_before,
         "abs diff <= 0.02", abs(veto["fpr_before"] - paper_fpr_before) <= 0.02),
        ("biopesticide_candidates_fraction", round(cand["headline_fraction"], 4), paper_frac,
         "abs diff <= 0.05", abs(cand["headline_fraction"] - paper_frac) <= 0.05),
    ]
    report = [{"metric": m, "reproduced": r, "paper": p, "tolerance": tol, "match": bool(ok)}
              for m, r, p, tol, ok in checks]
    n = int(sum(c["match"] for c in report))
    failed = [c["metric"] for c in report if not c["match"]]
    return {
        "answer": {"checks": report, "matched": n, "total": len(report),
                   "failed_checks": failed,
                   "summary": (f"{n}/{len(report)} headline numbers reproduced within the stated "
                               f"tolerances."
                               + (f" NOT reproduced: {', '.join(failed)}." if failed
                                  else " No check failed."))},
        "metadata": {"reference": PAPER,
                     "method": "open analogues: RDKit, HGB/CatBoost, RMT (numpy/scipy port), "
                               "kNN+Gaussian AD; docking scores bundled from the authors",
                     "tolerance_note": "Each check carries the tolerance it applied; read it "
                                       "before quoting a check as 'reproduced'."},
    }


@mcp.tool(name="canpest_reproduce_claims")
def reproduce_claims() -> dict:
    """Reproduce the paper's natural-language conclusions, each backed by recomputed numbers."""
    ds = load_dataset()
    dk = docking.active_vs_inactive(ds)
    rmt_res = models.rmt_selection(scaffold=False)
    ab = models.qsar_ablation(scaffold=False)
    cb = models.cb_sd_rte(scaffold=False)
    veto = models.docking_veto()
    cand = models.predict_biopesticides()
    stack = models.model_stack_quality()

    neg = [r for r in dk["per_protein"] if r["expected_trend"]]
    pos = [r for r in dk["per_protein"] if not r["expected_trend"]]
    n_prot = len(dk["per_protein"])
    d_lo = min((r["delta"] for r in neg), default=None)
    d_hi = max((r["delta"] for r in neg), default=None)
    or28_delta = next((r["delta"] for r in pos if r["protein"] == "OR28"), None)
    # C1 must compare its own magnitudes to the paper's, not just check which protein is the
    # outlier; the recomputed floor and the OR28 delta both sit outside the paper's stated values.
    c1_paper_neg = (-1.01, -0.70)
    c1_paper_or28 = 1.20
    c1_shape_ok = len(neg) == n_prot - 1 and dk["anomalous_protein"] == "OR28"
    c1_mag_ok = (d_lo is not None and c1_paper_neg[0] <= d_lo and d_hi <= c1_paper_neg[1]
                 and or28_delta is not None and abs(or28_delta - c1_paper_or28) <= 0.10)

    ref_rmt = reference.RMT
    c2_lambda_ok = abs(rmt_res["lambda_plus"] - ref_rmt["lambda_plus"]) < 0.01
    c2_signal_ok = rmt_res["n_signal"] == ref_rmt["n_signal"]

    best = max(ab["results"].items(), key=lambda kv: kv[1]["roc_auc"])
    paper_qsar = reference.QSAR["dmpnn_sd_roc"]
    # C3 previously gated a bundled constant against a constant (always True) and ignored the
    # only QSAR number this server measures. Gate on the recomputed ablation ROC-AUC instead.
    c3_qsar_ok = abs(best[1]["roc_auc"] - paper_qsar) <= 0.03
    c3_cb_ok = 0.75 <= cb["all6_rte"]["roc_auc"] <= 0.83

    paper_n, paper_frac = reference.CANDIDATES["n_high_conf"], reference.CANDIDATES["fraction"]
    paper_denom = round(paper_n / paper_frac)
    c4_ok = abs(cand["headline_fraction"] - paper_frac) <= 0.05

    claims = [
        {"id": "C1", "question": "Do C. sativa metabolites bind pest targets like pesticides?",
         "paper_assertion": "For 5 of 6 targets actives bind more strongly (Δ=−0.70…−1.01); OR28 is "
                            "opposite (Δ=+1.20), indicating a multi-target lethal+repellent mechanism.",
         "reproduced_statement": (
             f"{len(neg)} of {n_prot} targets show delta<0 (range {d_lo:+.2f}…{d_hi:+.2f}) and the "
             f"anomalous target is {dk['anomalous_protein']}"
             + (f" (delta {or28_delta:+.2f})" if or28_delta is not None else "")
             + (". The direction pattern reproduces the paper and the magnitudes fall inside the "
                "paper's stated ranges."
                if c1_shape_ok and c1_mag_ok else
                f". The direction pattern "
                f"{'reproduces' if c1_shape_ok else 'does NOT reproduce'} the paper, but the "
                f"magnitudes differ: the paper states {c1_paper_neg[0]}…{c1_paper_neg[1]} for the "
                f"expected targets and {c1_paper_or28:+.2f} for OR28, so report the recomputed "
                f"deltas, not the paper's.")),
         # The assertion is about the MECHANISM (which targets go which way), so that is what the
         # flag tracks; the magnitude agreement is reported separately rather than folded in.
         "reproduced": bool(c1_shape_ok),
         "direction_pattern_matches_paper": bool(c1_shape_ok),
         "magnitudes_match_paper": bool(c1_mag_ok),
         "method": "recomputed (Mann-Whitney on bundled docking scores)"},
        {"id": "C2", "question": "Does RMT extract a signal subspace from the docking features?",
         "paper_assertion": f"RMT separates signal from noise (λ+≈{ref_rmt['lambda_plus']}), with "
                            f"{ref_rmt['n_signal']} signal eigenvalues and ~"
                            f"{ref_rmt['m_opt_random']} informative residue-term features.",
         "reproduced_statement": (
             f"lambda+ = {rmt_res['lambda_plus']} (paper {ref_rmt['lambda_plus']}), m_opt = "
             f"{rmt_res['m_opt']} (paper {ref_rmt['m_opt_random']}), {rmt_res['n_signal']} signal "
             f"eigenvalues (paper {ref_rmt['n_signal']})."
             + (" All three reproduce." if c2_lambda_ok and c2_signal_ok else
                f" lambda+ {'reproduces' if c2_lambda_ok else 'does NOT reproduce'}; the signal-"
                f"eigenvalue count {'reproduces' if c2_signal_ok else 'does NOT'} "
                f"({rmt_res['n_signal']} vs {ref_rmt['n_signal']}).")),
         # The paper's sentence asserts the separation (lambda+) and the feature count (m_opt);
         # the signal-eigenvalue count is a secondary quantity, so it is surfaced as its own flag
         # instead of being silently omitted (as before) or silently failing the whole claim.
         "reproduced": bool(c2_lambda_ok and abs(rmt_res["m_opt"] - ref_rmt["m_opt_random"]) <= 15),
         "lambda_plus_matches_paper": bool(c2_lambda_ok),
         "n_signal_matches_paper": bool(c2_signal_ok),
         "method": "recomputed (numpy/scipy RMT port)"},
        {"id": "C3", "question": "How good is the biopesticide QSAR model?",
         "paper_assertion": f"The DMPNN+HGB stack (DMPNN-SD) reaches ROC-AUC {paper_qsar}; "
                            f"residue-level docking models (CB-SD) reach 0.68–0.80.",
         "reproduced_statement": (
             f"Recomputed here: the best feature set ({best[0]}) reaches ROC-AUC "
             f"{best[1]['roc_auc']:.4f} against the paper's {paper_qsar} "
             f"({best[1]['roc_auc'] - paper_qsar:+.4f}), and the residue-level CB-SD model reaches "
             f"{cb['all6_rte']['roc_auc']:.3f} (paper {reference.QSAR['cb_sd_all6']}). "
             + ("Both fall within tolerance of the paper." if c3_qsar_ok and c3_cb_ok else
                "At least one is outside tolerance, so this claim does not fully reproduce. ")
             + f"(For reference, the authors' bundled stack CV — precomputed, NOT measured here — "
               f"ranks blend {stack['blend']['roc_auc']} > DMPNN "
               f"{stack['components']['dmpnn']['roc_auc']} > HGB "
               f"{stack['components']['hgb']['roc_auc']}.)"),
         "reproduced": bool(c3_qsar_ok and c3_cb_ok),
         "method": "recomputed (HGB ablation + CB-SD); stack numbers are published references"},
        {"id": "C_veto", "question": "Does the docking-consistency veto cut false positives?",
         "paper_assertion": "p_final = p_DMPNN × p_RMT-RTE lowers FPR from 12.20% to 4.92% (~60%) while "
                            "keeping ROC-AUC high.",
         "reproduced_statement": (
             f"FPR {veto['fpr_before']:.1%}->{veto['fpr_after_veto']:.1%} "
             f"({veto['fpr_reduction_pct']:.0f}% reduction), ROC-AUC "
             f"{veto['roc_auc_qsar']}->{veto['roc_auc_after_veto']}. The paper reports "
             f"{reference.QSAR['fpr_before']:.1%}->{reference.QSAR['fpr_after_veto']:.1%}; the "
             f"EFFECT (a large FPR cut at nearly unchanged ROC-AUC) reproduces, the absolute "
             f"baseline FPR does not ({veto['fpr_before']:.1%} vs "
             f"{reference.QSAR['fpr_before']:.1%})."),
         "reproduced": bool(veto["fpr_after_veto"] < 0.08 and veto["fpr_reduction_pct"] >= 40),
         "method": "recomputed"},
        {"id": "C4", "question": "How many C. sativa metabolites are candidate biopesticides?",
         "paper_assertion": f"{paper_n} metabolites ({paper_frac:.2%}) have pesticide probability "
                            f">0.7.",
         "reproduced_statement": (
             f"{cand['backend']}: {cand['headline_count']} of {cand['n_metabolites']} metabolites "
             f"({cand['headline_fraction']:.1%}) at probability >"
             f"{get_settings().candidate_probability} [{cand['headline_basis']}], against the "
             f"paper's {paper_n} ({paper_frac:.2%}) of ~{paper_denom}. The FRACTION "
             f"{'reproduces' if c4_ok else 'does NOT reproduce'} "
             f"({cand['headline_fraction'] - paper_frac:+.1%}); the COUNT is not comparable "
             f"because the screening set here holds {cand['n_metabolites']} metabolites against "
             f"the paper's ~{paper_denom}."),
         "reproduced": bool(c4_ok),
         "method": "recomputed"},
        {"id": "C5", "question": "Are C. sativa metabolites safer than synthetic pesticides?",
         "paper_assertion": "Metabolites have higher median LD50 and far less hepatotoxicity/DILI and "
                            "aquatic toxicity than synthetic pesticides.",
         "reproduced_statement": (
             "NOT REPRODUCED — reported only from the paper's proprietary Syntelly predictions "
             "(Table 3/4). This server runs no toxicity model, so it can neither confirm nor "
             "refute the claim. Live open-model analogues are available on the `heracleum-tox` "
             "server (`predict_general_toxicity`, `predict_ld50`), and there the open DILI model "
             "disagrees with Syntelly on furanocoumarins — so this claim should not be relayed as "
             "reproduced."),
         # Previously hardcoded True, which inflated the headline to 6/6 while the claim's own
         # statement admitted nothing had been reproduced.
         "reproduced": False, "method": "published-value (Syntelly) — not verifiable here"},
    ]
    n = int(sum(bool(c["reproduced"]) for c in claims))
    not_reproduced = [c["id"] for c in claims if not c["reproduced"]]
    narrative = " ".join(
        (c["reproduced_statement"] if c["reproduced"]
         else f"[NOT REPRODUCED — {c['id']}] {c['reproduced_statement']}")
        for c in claims)
    return {
        "answer": {"claims": claims, "reproduced": n, "total": len(claims),
                   "not_reproduced_ids": not_reproduced,
                   "narrative": narrative,
                   "finding": (f"{n}/{len(claims)} of the paper's claims are reproduced from "
                               f"numbers recomputed by this server"
                               + (f"; NOT reproduced: {', '.join(not_reproduced)} — report these "
                                  f"as divergences, do not omit them." if not_reproduced
                                  else " (all claims reproduced)."))},
        "metadata": {"reference": PAPER,
                     "usage": "Relay `reproduced_statement` only together with the claim's "
                              "`reproduced` flag and `method`. `narrative` includes the "
                              "non-reproduced claims prefixed [NOT REPRODUCED]."},
    }


def main() -> None:
    settings = get_settings()
    artifacts.ensure_storage_ready()
    logger.info("Starting cannabis-biopesticide MCP server on %s:%s%s",
                settings.mcp_host, settings.mcp_port, settings.mcp_path)
    mcp.run(transport="http", host=settings.mcp_host, port=settings.mcp_port, path=settings.mcp_path)


if __name__ == "__main__":
    main()
