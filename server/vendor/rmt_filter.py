#!/usr/bin/env python3
"""Project-agnostic RMT feature filter.

This script takes a feature matrix and an external split registry and produces
two feature spaces:

  rmt_rte_sel: selected original feature columns, raw values preserved
  rmt_rte_rec: the same selected columns after signal-subspace reconstruction

The script is intentionally not tied to canpest paths, proteins, or Vina names.
It only assumes:

  features CSV:  <id_col> + numeric feature columns
  split CSV:     split_id + set(train/test) + <id_col> + target column

Example:

  python scripts/rmt_filter.py \
    --features-csv features/protein-related/rte.csv \
    --split-csv data/split_registry.csv \
    --out-dir results/rmt_filter/random_split \
    --target-col activity \
    --split-id split_00
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


MODES = ("rmt_rte_sel", "rmt_rte_rec")


@dataclass
class TargetInfo:
    kind: str
    y: np.ndarray
    raw_values: pd.Series
    class_mapping: dict[str, int] | None = None


@dataclass
class RMTFit:
    feature_cols: list[str]
    mean: np.ndarray
    scale: np.ndarray
    eigvals: np.ndarray
    eigvecs: np.ndarray
    raw_signal_mask: np.ndarray
    used_signal_mask: np.ndarray
    lambda_plus: float
    q: float
    n_signal_raw: int
    n_signal_used: int
    s_i: np.ndarray


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build RMT-selected and RMT-reconstructed feature matrices from CSV inputs."
    )
    p.add_argument("--features-csv", required=True, help="CSV with id column + numeric features")
    p.add_argument("--split-csv", required=True, help="CSV with split_id, set, id column, and target")
    p.add_argument("--out-dir", required=True, help="Output directory")

    p.add_argument("--id-col", default="ligand_id", help="ID column name shared by both CSV files")
    p.add_argument("--target-col", default="activity", help="Target column in split CSV")
    p.add_argument("--split-col", default="split_id", help="Split id column in split CSV")
    p.add_argument("--set-col", default="set", help="Train/test set column in split CSV")
    p.add_argument("--train-label", default="train", help="Value in set column denoting train rows")
    p.add_argument("--test-label", default="test", help="Value in set column denoting test rows")
    p.add_argument("--split-id", default="all", help="Split id to process, comma-list, or 'all'")
    p.add_argument(
        "--id-normalization",
        choices=["string", "int-like", "none"],
        default="string",
        help="How to normalize IDs before merging. 'int-like' accepts ligand_00001_tau_1 style IDs.",
    )

    p.add_argument(
        "--feature-cols",
        default=None,
        help="Optional comma-separated feature columns. Default: all numeric columns except metadata.",
    )
    p.add_argument(
        "--exclude-cols",
        default=None,
        help="Optional comma-separated columns to exclude from inferred features.",
    )
    p.add_argument(
        "--modes",
        default=",".join(MODES),
        help="Comma-separated output modes. Available: rmt_rte_sel,rmt_rte_rec",
    )

    p.add_argument("--target-type", choices=["auto", "binary", "continuous"], default="auto")
    p.add_argument(
        "--rank-by",
        choices=["hybrid", "rmt", "target"],
        default="hybrid",
        help="Feature ranking score: s_i*target_score, s_i only, or target_score only.",
    )
    p.add_argument(
        "--target-score",
        choices=["spearman", "pearson"],
        default="spearman",
        help="Univariate target relevance used in hybrid ranking.",
    )

    p.add_argument("--n-inner-splits", type=int, default=4, help="Inner CV split count")
    p.add_argument("--inner-train-frac", type=float, default=0.7, help="Inner train fraction")
    p.add_argument("--max-m", type=int, default=0, help="Maximum top-m features to scan; 0 = all")
    p.add_argument("--m-step", type=int, default=1, help="Step for m scan when --m-grid is absent")
    p.add_argument("--m-grid", default=None, help="Optional comma-separated explicit m values")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--metric",
        choices=["auto", "roc_auc", "spearman", "pearson", "r2"],
        default="auto",
        help="Inner CV metric. auto: binary->roc_auc, continuous->spearman.",
    )
    p.add_argument("--ridge-alpha", type=float, default=1.0, help="Ridge alpha for continuous target")
    p.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="balanced",
        help="Class weight for binary LogisticRegression in inner CV.",
    )
    p.add_argument(
        "--fallback-top-signal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If no eigenvalue exceeds lambda_plus, use the top eigenvector as a practical fallback.",
    )
    return p.parse_args()


def comma_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [x.strip() for x in value.split(",") if x.strip()]
    return items or None


def normalize_ids(series: pd.Series, mode: str) -> pd.Series:
    if mode == "none":
        return series
    raw = series.astype(str).str.strip()
    if mode == "string":
        return raw
    normalized = (
        raw.str.lower()
        .str.replace("ligand_", "", regex=False)
        .str.replace(r"_tau_\d+$", "", regex=True)
    )
    parsed = pd.to_numeric(normalized, errors="coerce")
    bad = parsed.isna()
    if bad.any():
        raise ValueError(f"Could not parse ID values as int-like IDs: {raw[bad].head(5).tolist()}")
    return parsed.astype("int64").astype(str)


def infer_feature_cols(df: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    explicit = comma_list(args.feature_cols)
    if explicit is not None:
        missing = [c for c in explicit if c not in df.columns]
        if missing:
            raise ValueError(f"Requested feature columns are absent: {missing[:10]}")
        return explicit

    exclude = set(comma_list(args.exclude_cols) or [])
    exclude.update({args.id_col, args.target_col, args.split_col, args.set_col})
    cols: list[str] = []
    for col in df.columns:
        if col in exclude:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().any():
            cols.append(col)
    if not cols:
        raise ValueError("No numeric feature columns were inferred")
    return cols


def numeric_matrix(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    return df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def fit_imputer_scaler(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(x_train, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    x_imp = np.where(np.isnan(x_train), mean[None, :], x_train)
    scale = x_imp.std(axis=0, ddof=1)
    scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0)
    return mean, scale


def apply_imputer(x: np.ndarray, mean: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(x), mean[None, :], x)


def zscore_with_stats(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (apply_imputer(x, mean) - mean[None, :]) / scale[None, :]


def fit_rmt(x_train: np.ndarray, feature_cols: list[str], *, fallback_top_signal: bool) -> RMTFit:
    if x_train.shape[0] < 3:
        raise ValueError("RMT requires at least 3 train rows")
    if x_train.shape[1] < 1:
        raise ValueError("RMT requires at least 1 feature")

    mean, scale = fit_imputer_scaler(x_train)
    xz = zscore_with_stats(x_train, mean, scale)
    n_rows, n_cols = xz.shape
    corr = (xz.T @ xz) / max(n_rows - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(corr)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    q = n_cols / n_rows
    lambda_plus = float((1.0 + np.sqrt(q)) ** 2)
    raw_signal_mask = eigvals > lambda_plus
    used_signal_mask = raw_signal_mask.copy()
    n_signal_raw = int(raw_signal_mask.sum())
    if n_signal_raw == 0 and fallback_top_signal:
        used_signal_mask[0] = True

    signal_eigvals = eigvals[used_signal_mask]
    signal_eigvecs = eigvecs[:, used_signal_mask]
    if signal_eigvecs.shape[1] == 0:
        s_i = np.zeros(n_cols, dtype=float)
    else:
        s_i = (signal_eigvecs * signal_eigvecs) @ signal_eigvals

    return RMTFit(
        feature_cols=list(feature_cols),
        mean=mean,
        scale=scale,
        eigvals=eigvals,
        eigvecs=eigvecs,
        raw_signal_mask=raw_signal_mask,
        used_signal_mask=used_signal_mask,
        lambda_plus=lambda_plus,
        q=float(q),
        n_signal_raw=n_signal_raw,
        n_signal_used=int(used_signal_mask.sum()),
        s_i=s_i,
    )


def reconstruct_from_signal(x: np.ndarray, fit: RMTFit) -> np.ndarray:
    xz = zscore_with_stats(x, fit.mean, fit.scale)
    v = fit.eigvecs[:, fit.used_signal_mask]
    if v.shape[1] == 0:
        x_signal_z = np.zeros_like(xz)
    else:
        x_signal_z = (xz @ v) @ v.T
    return x_signal_z * fit.scale[None, :] + fit.mean[None, :]


def encode_target(raw: pd.Series, requested: str) -> TargetInfo:
    raw = raw.reset_index(drop=True)
    if raw.isna().any():
        raise ValueError("Train target contains missing values")

    numeric = pd.to_numeric(raw, errors="coerce")
    numeric_ok = numeric.notna().all()
    n_unique = raw.nunique(dropna=True)

    if requested == "auto":
        if numeric_ok and n_unique > 2:
            kind = "continuous"
        elif n_unique <= 2:
            kind = "binary"
        else:
            raise ValueError("Could not infer target type; use --target-type")
    else:
        kind = requested

    if kind == "continuous":
        if not numeric_ok:
            raise ValueError("Continuous target must be numeric")
        return TargetInfo(kind=kind, y=numeric.to_numpy(dtype=float), raw_values=raw)

    if n_unique != 2:
        raise ValueError(f"Binary target requires exactly 2 classes, got {n_unique}")
    if numeric_ok:
        vals = sorted(numeric.unique().tolist())
        mapping = {str(vals[0]): 0, str(vals[1]): 1}
        y = numeric.map({vals[0]: 0, vals[1]: 1}).to_numpy(dtype=int)
    else:
        vals = sorted(raw.astype(str).unique().tolist())
        mapping = {vals[0]: 0, vals[1]: 1}
        y = raw.astype(str).map(mapping).to_numpy(dtype=int)
    return TargetInfo(kind=kind, y=y, raw_values=raw, class_mapping=mapping)


def pearson_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y_center = y - y.mean()
    x_center = x - x.mean(axis=0)
    denom = np.sqrt((x_center * x_center).sum(axis=0) * float((y_center * y_center).sum()))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = (x_center.T @ y_center) / denom
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def spearman_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    rx = np.apply_along_axis(rankdata, 0, x)
    ry = rankdata(y)
    return pearson_scores(rx, ry)


def target_relevance(x_train: np.ndarray, y: np.ndarray, method: str) -> np.ndarray:
    if method == "pearson":
        corr = pearson_scores(x_train, y)
    elif method == "spearman":
        corr = spearman_scores(x_train, y)
    else:
        raise ValueError(f"Unsupported target score: {method}")
    return np.abs(corr)


def feature_scores(
    x_train: np.ndarray,
    y: np.ndarray,
    feature_cols: list[str],
    *,
    rank_by: str,
    target_score_method: str,
    fallback_top_signal: bool,
) -> tuple[pd.DataFrame, np.ndarray, RMTFit]:
    fit = fit_rmt(x_train, feature_cols, fallback_top_signal=fallback_top_signal)
    x_imp = apply_imputer(x_train, fit.mean)
    t_score = target_relevance(x_imp, y, target_score_method)
    hybrid = fit.s_i * t_score
    if rank_by == "rmt":
        rank_score = fit.s_i
    elif rank_by == "target":
        rank_score = t_score
    else:
        rank_score = hybrid

    order = np.lexsort((-t_score, -fit.s_i, -rank_score))
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    rmt_order = np.argsort(-fit.s_i)
    rmt_rank = np.empty_like(rmt_order)
    rmt_rank[rmt_order] = np.arange(1, len(rmt_order) + 1)
    target_order = np.argsort(-t_score)
    target_rank = np.empty_like(target_order)
    target_rank[target_order] = np.arange(1, len(target_order) + 1)

    scores = pd.DataFrame(
        {
            "feature": feature_cols,
            "feature_rank": ranks.astype(int),
            "rank_score": rank_score,
            "rmt_rank": rmt_rank.astype(int),
            "rmt_s_i": fit.s_i,
            "target_rank": target_rank.astype(int),
            "target_score": t_score,
            "hybrid_score": hybrid,
        }
    ).sort_values("feature_rank", ascending=True).reset_index(drop=True)
    return scores, order, fit


def metric_name(target_kind: str, requested: str) -> str:
    if requested != "auto":
        return requested
    return "roc_auc" if target_kind == "binary" else "spearman"


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    if metric == "roc_auc":
        if len(np.unique(y_true)) < 2:
            return 0.5
        try:
            return float(roc_auc_score(y_true, y_pred))
        except ValueError:
            return 0.5
    if metric == "r2":
        return float(r2_score(y_true, y_pred))
    if metric == "pearson":
        corr = pearson_scores(y_pred.reshape(-1, 1), y_true)[0]
        return float(corr)
    if metric == "spearman":
        corr = spearmanr(y_true, y_pred).correlation
        return float(0.0 if not np.isfinite(corr) else corr)
    raise ValueError(f"Unsupported metric: {metric}")


def evaluate_top_m(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    selected_idx: np.ndarray,
    target_kind: str,
    metric: str,
    *,
    class_weight: str,
    ridge_alpha: float,
) -> float:
    xtr = x_train[:, selected_idx]
    xva = x_valid[:, selected_idx]
    scaler = StandardScaler()
    xtr_sc = scaler.fit_transform(xtr)
    xva_sc = scaler.transform(xva)

    if target_kind == "binary":
        if len(np.unique(y_train)) < 2:
            return 0.5
        weight = None if class_weight == "none" else "balanced"
        model = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight=weight)
        model.fit(xtr_sc, y_train)
        y_pred = model.predict_proba(xva_sc)[:, 1]
        return score_predictions(y_valid, y_pred, metric)

    model = Ridge(alpha=ridge_alpha)
    model.fit(xtr_sc, y_train)
    y_pred = model.predict(xva_sc)
    return score_predictions(y_valid, y_pred, metric)


def make_m_values(n_features: int, args: argparse.Namespace) -> list[int]:
    max_m = n_features if args.max_m == 0 else min(args.max_m, n_features)
    if max_m < 1:
        raise ValueError("No m values to scan")
    explicit = comma_list(args.m_grid)
    if explicit:
        vals = sorted({int(x) for x in explicit if 1 <= int(x) <= max_m})
        if not vals:
            raise ValueError("--m-grid has no valid values")
        return vals
    step = max(1, int(args.m_step))
    vals = list(range(1, max_m + 1, step))
    if vals[-1] != max_m:
        vals.append(max_m)
    return vals


def inner_cv_select_m(
    x_train: np.ndarray,
    y_train: np.ndarray,
    feature_cols: list[str],
    target_kind: str,
    args: argparse.Namespace,
) -> tuple[int, pd.DataFrame, pd.DataFrame]:
    m_values = make_m_values(x_train.shape[1], args)
    metric = metric_name(target_kind, args.metric)

    if target_kind == "binary":
        splitter = StratifiedShuffleSplit(
            n_splits=args.n_inner_splits,
            train_size=args.inner_train_frac,
            random_state=args.seed,
        )
        split_iter: Iterable[tuple[np.ndarray, np.ndarray]] = splitter.split(x_train, y_train)
    else:
        splitter = ShuffleSplit(
            n_splits=args.n_inner_splits,
            train_size=args.inner_train_frac,
            random_state=args.seed,
        )
        split_iter = splitter.split(x_train)

    records: list[dict] = []
    for fold, (fit_idx, valid_idx) in enumerate(split_iter):
        print(f"    inner fold {fold + 1}/{args.n_inner_splits}", flush=True)
        x_fit = x_train[fit_idx]
        y_fit = y_train[fit_idx]
        x_valid = x_train[valid_idx]
        y_valid = y_train[valid_idx]

        scores, order, rmt = feature_scores(
            x_fit,
            y_fit,
            feature_cols,
            rank_by=args.rank_by,
            target_score_method=args.target_score,
            fallback_top_signal=args.fallback_top_signal,
        )
        del scores

        for m in m_values:
            selected_idx = order[:m]
            value = evaluate_top_m(
                x_fit,
                y_fit,
                x_valid,
                y_valid,
                selected_idx,
                target_kind,
                metric,
                class_weight=args.class_weight,
                ridge_alpha=args.ridge_alpha,
            )
            records.append(
                {
                    "fold": fold,
                    "m": int(m),
                    "metric": metric,
                    "score": float(value),
                    "lambda_plus": float(rmt.lambda_plus),
                    "n_signal_raw": int(rmt.n_signal_raw),
                    "n_signal_used": int(rmt.n_signal_used),
                    "q": float(rmt.q),
                }
            )

    scan = pd.DataFrame(records)
    summary = (
        scan.groupby("m", as_index=False)
        .agg(
            mean_score=("score", "mean"),
            std_score=("score", "std"),
            n_valid_folds=("score", "count"),
        )
        .sort_values("m")
        .reset_index(drop=True)
    )
    best = summary.sort_values(["mean_score", "m"], ascending=[False, True]).iloc[0]
    return int(best["m"]), scan, summary


def frame_from_matrix(
    rows: pd.DataFrame,
    matrix: np.ndarray,
    selected_cols: list[str],
    *,
    id_col: str,
    target_col: str,
    split_id: str,
    set_label: str,
) -> pd.DataFrame:
    out = pd.DataFrame(matrix, columns=selected_cols)
    out.insert(0, target_col, rows[target_col].reset_index(drop=True).values)
    out.insert(0, id_col, rows[id_col].reset_index(drop=True).values)
    out.insert(0, "set", set_label)
    out.insert(0, "split_id", split_id)
    return out


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def process_split(
    split_id: str,
    features: pd.DataFrame,
    split_rows: pd.DataFrame,
    feature_cols: list[str],
    args: argparse.Namespace,
    out_dir: Path,
) -> dict:
    split_dir = out_dir / split_id
    split_dir.mkdir(parents=True, exist_ok=True)
    print(f"Processing {split_id}...", flush=True)

    train_rows = split_rows[split_rows[args.set_col].astype(str) == args.train_label].copy()
    test_rows = split_rows[split_rows[args.set_col].astype(str) == args.test_label].copy()
    if train_rows.empty:
        raise ValueError(f"{split_id}: no train rows")

    train = train_rows.merge(features[[args.id_col, *feature_cols]], on=args.id_col, how="left", validate="one_to_one")
    test = test_rows.merge(features[[args.id_col, *feature_cols]], on=args.id_col, how="left", validate="one_to_one")

    missing_train = train[feature_cols].isna().all(axis=1)
    missing_test = test[feature_cols].isna().all(axis=1) if not test.empty else pd.Series(dtype=bool)
    if missing_train.any() or (not test.empty and missing_test.any()):
        examples = train.loc[missing_train, args.id_col].head(5).tolist()
        examples += test.loc[missing_test, args.id_col].head(5).tolist() if not test.empty else []
        raise ValueError(f"{split_id}: split IDs absent from feature CSV, examples={examples[:5]}")

    target_info = encode_target(train[args.target_col], args.target_type)
    x_train = numeric_matrix(train, feature_cols)
    x_test = numeric_matrix(test, feature_cols) if not test.empty else np.empty((0, len(feature_cols)))

    m_opt, scan, scan_summary = inner_cv_select_m(x_train, target_info.y, feature_cols, target_info.kind, args)
    scan.to_csv(split_dir / "m_scan.csv", index=False)
    scan_summary.to_csv(split_dir / "m_scan_summary.csv", index=False)

    scores, order, rmt = feature_scores(
        x_train,
        target_info.y,
        feature_cols,
        rank_by=args.rank_by,
        target_score_method=args.target_score,
        fallback_top_signal=args.fallback_top_signal,
    )
    selected_idx = order[:m_opt]
    selected_cols = [feature_cols[i] for i in selected_idx]
    scores["selected"] = scores["feature"].isin(selected_cols)
    scores.to_csv(split_dir / "feature_scores.csv", index=False)
    selected_scores = scores[scores["selected"]].sort_values("feature_rank").reset_index(drop=True)
    selected_scores.to_csv(split_dir / "selected_features.csv", index=False)

    x_train_imp = apply_imputer(x_train, rmt.mean)
    x_test_imp = apply_imputer(x_test, rmt.mean) if len(x_test) else x_test
    x_train_sel = x_train_imp[:, selected_idx]
    x_test_sel = x_test_imp[:, selected_idx] if len(x_test) else np.empty((0, len(selected_cols)))

    x_train_rec_all = reconstruct_from_signal(x_train, rmt)
    x_test_rec_all = reconstruct_from_signal(x_test, rmt) if len(x_test) else x_test
    x_train_rec = x_train_rec_all[:, selected_idx]
    x_test_rec = x_test_rec_all[:, selected_idx] if len(x_test) else np.empty((0, len(selected_cols)))

    modes = set(comma_list(args.modes) or MODES)
    bad_modes = sorted(modes - set(MODES))
    if bad_modes:
        raise ValueError(f"Unsupported modes: {bad_modes}")

    outputs: dict[str, dict[str, str]] = {}
    if "rmt_rte_sel" in modes:
        sel_train = frame_from_matrix(
            train, x_train_sel, selected_cols,
            id_col=args.id_col, target_col=args.target_col, split_id=split_id, set_label=args.train_label,
        )
        sel_train.to_csv(split_dir / "rmt_rte_sel_train.csv", index=False)
        outputs.setdefault("rmt_rte_sel", {})["train"] = "rmt_rte_sel_train.csv"
        if not test.empty:
            sel_test = frame_from_matrix(
                test, x_test_sel, selected_cols,
                id_col=args.id_col, target_col=args.target_col, split_id=split_id, set_label=args.test_label,
            )
            sel_test.to_csv(split_dir / "rmt_rte_sel_test.csv", index=False)
            outputs["rmt_rte_sel"]["test"] = "rmt_rte_sel_test.csv"

    if "rmt_rte_rec" in modes:
        rec_train = frame_from_matrix(
            train, x_train_rec, selected_cols,
            id_col=args.id_col, target_col=args.target_col, split_id=split_id, set_label=args.train_label,
        )
        rec_train.to_csv(split_dir / "rmt_rte_rec_train.csv", index=False)
        outputs.setdefault("rmt_rte_rec", {})["train"] = "rmt_rte_rec_train.csv"
        if not test.empty:
            rec_test = frame_from_matrix(
                test, x_test_rec, selected_cols,
                id_col=args.id_col, target_col=args.target_col, split_id=split_id, set_label=args.test_label,
            )
            rec_test.to_csv(split_dir / "rmt_rte_rec_test.csv", index=False)
            outputs["rmt_rte_rec"]["test"] = "rmt_rte_rec_test.csv"

    best_row = scan_summary[scan_summary["m"] == m_opt].iloc[0]
    spec = {
        "script": "rmt_filter.py",
        "split_id": split_id,
        "feature_space": "rmt_rte",
        "modes": sorted(modes),
        "definition": {
            "rmt_rte_sel": "selected original feature columns; values are train-mean-imputed raw inputs",
            "rmt_rte_rec": "same selected columns reconstructed from RMT signal eigenvectors in original feature coordinates",
        },
        "selection_rule": "train-only RMT score plus target relevance ranking, inner-CV m_opt, keep top-m_opt features",
        "leakage_guard": "RMT fit, ranking, target relevance, m_opt, imputation, scaling, and reconstruction are fit on train rows only.",
        "input_features_csv": args.features_csv,
        "input_split_csv": args.split_csv,
        "id_col": args.id_col,
        "target_col": args.target_col,
        "target_type": target_info.kind,
        "target_class_mapping": target_info.class_mapping,
        "rank_by": args.rank_by,
        "target_score": args.target_score,
        "inner_metric": metric_name(target_info.kind, args.metric),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_features_total": int(len(feature_cols)),
        "n_selected": int(len(selected_cols)),
        "m_opt": int(m_opt),
        "inner_mean_score_at_m_opt": float(best_row["mean_score"]),
        "inner_std_score_at_m_opt": None if pd.isna(best_row["std_score"]) else float(best_row["std_score"]),
        "lambda_plus": float(rmt.lambda_plus),
        "q": float(rmt.q),
        "n_signal_raw": int(rmt.n_signal_raw),
        "n_signal_used": int(rmt.n_signal_used),
        "fallback_top_signal": bool(args.fallback_top_signal),
        "selected_features": selected_cols,
        "outputs": {
            "feature_scores": "feature_scores.csv",
            "selected_features": "selected_features.csv",
            "m_scan": "m_scan.csv",
            "m_scan_summary": "m_scan_summary.csv",
            **outputs,
        },
    }
    write_json(split_dir / "transform_spec.json", spec)
    return spec


def requested_splits(split_df: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    all_splits = sorted(split_df[args.split_col].astype(str).unique().tolist())
    if args.split_id == "all":
        return all_splits
    wanted = comma_list(args.split_id) or []
    missing = sorted(set(wanted) - set(all_splits))
    if missing:
        raise ValueError(f"Requested split IDs are absent: {missing}")
    return wanted


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(args.features_csv)
    split = pd.read_csv(args.split_csv)
    if args.id_col not in features.columns:
        raise ValueError(f"Feature CSV lacks id column: {args.id_col}")
    for col in [args.split_col, args.set_col, args.id_col, args.target_col]:
        if col not in split.columns:
            raise ValueError(f"Split CSV lacks required column: {col}")

    features = features.copy()
    split = split.copy()
    features[args.id_col] = normalize_ids(features[args.id_col], args.id_normalization)
    split[args.id_col] = normalize_ids(split[args.id_col], args.id_normalization)
    split[args.split_col] = split[args.split_col].astype(str)
    split[args.set_col] = split[args.set_col].astype(str)

    if features[args.id_col].duplicated().any():
        examples = features.loc[features[args.id_col].duplicated(), args.id_col].head(5).tolist()
        raise ValueError(f"Feature CSV has duplicate IDs: {examples}")

    feature_cols = infer_feature_cols(features, args)
    split_ids = requested_splits(split, args)

    specs: list[dict] = []
    selected_rows: list[pd.DataFrame] = []
    for split_id in split_ids:
        rows = split[split[args.split_col] == split_id].copy()
        spec = process_split(split_id, features, rows, feature_cols, args, out_dir)
        specs.append(spec)
        selected = pd.read_csv(out_dir / split_id / "selected_features.csv")
        selected.insert(0, "split_id", split_id)
        selected_rows.append(selected)
        print(
            f"{split_id}: m_opt={spec['m_opt']} selected={spec['n_selected']} "
            f"signal={spec['n_signal_raw']}/{spec['n_signal_used']} "
            f"inner={spec['inner_mean_score_at_m_opt']:.4f}"
        )

    summary = pd.DataFrame(
        [
            {
                "split_id": s["split_id"],
                "target_type": s["target_type"],
                "n_train": s["n_train"],
                "n_test": s["n_test"],
                "n_features_total": s["n_features_total"],
                "m_opt": s["m_opt"],
                "n_selected": s["n_selected"],
                "inner_metric": s["inner_metric"],
                "inner_mean_score_at_m_opt": s["inner_mean_score_at_m_opt"],
                "inner_std_score_at_m_opt": s["inner_std_score_at_m_opt"],
                "lambda_plus": s["lambda_plus"],
                "q": s["q"],
                "n_signal_raw": s["n_signal_raw"],
                "n_signal_used": s["n_signal_used"],
            }
            for s in specs
        ]
    )
    summary.to_csv(out_dir / "summary.csv", index=False)
    if selected_rows:
        pd.concat(selected_rows, ignore_index=True).to_csv(out_dir / "selected_features_long.csv", index=False)

    metadata = {
        "script": "rmt_filter.py",
        "feature_space": "rmt_rte",
        "modes": comma_list(args.modes),
        "features_csv": args.features_csv,
        "split_csv": args.split_csv,
        "out_dir": str(out_dir),
        "id_col": args.id_col,
        "target_col": args.target_col,
        "split_col": args.split_col,
        "set_col": args.set_col,
        "target_type": args.target_type,
        "rank_by": args.rank_by,
        "target_score": args.target_score,
        "n_inner_splits": args.n_inner_splits,
        "inner_train_frac": args.inner_train_frac,
        "max_m": args.max_m,
        "m_step": args.m_step,
        "m_grid": args.m_grid,
        "metric": args.metric,
        "seed": args.seed,
        "n_splits_processed": len(specs),
        "n_features_total": len(feature_cols),
        "outputs": {
            "summary": "summary.csv",
            "selected_features_long": "selected_features_long.csv",
            "per_split": "<split_id>/transform_spec.json and feature CSVs",
        },
    }
    write_json(out_dir / "metadata.json", metadata)


if __name__ == "__main__":
    main()
