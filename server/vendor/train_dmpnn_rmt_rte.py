#!/usr/bin/env python3
"""One/few-split DMPNN + GBM benchmark for RMT-RTE feature spaces.

The runner follows the local DMPNN implementation used in LOG/36_dmpnn, but uses
new split-specific RMT-RTE features:

  rmt_rte_sel = selected raw residue-term features
  rmt_rte_rec = selected reconstructed residue-term features

Default global features concatenated to the DMPNN molecule vector:
  RMT-RTE + engineered docking + RDKit 2D descriptors

For ablation runs, --feature-sets can be used to compare:
  rdkit2d, rdkit2d_dock2, rdkit2d_rmt_sel, rdkit2d_rmt_rec

The scaler is fit on the training subset only; near-constant global features are
removed with std <= 1e-6. A HistGradientBoostingClassifier is trained on the same
global features and blended with DMPNN by validation ROC AUC.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


BASE = Path(__file__).resolve().parent.parent
DATA_CSV = BASE / "data" / "data.csv"
SPLIT_REG = BASE / "data" / "split_registry.csv"
RDKIT_CACHE = BASE / "data" / "fp_rdkit2d.npy"
RTE_CSV = BASE / "features" / "protein-related" / "rte.csv"

DOCK_COLS = ["2imi", "d8v7j0", "3rif", "8sfy", "8udb", "8v3d"]
DOCK2_COLS = ["d8v7j0", "3rif"]  # ACHE + GLC1, matching the current RMT-RTE proteins.

FEATURE_SET_SPECS = {
    "rdkit2d": {"rmt_mode": "rmt_rte_sel", "include_rmt": False, "include_raw_rte": False, "include_dock2": False},
    "rdkit2d_dock2": {"rmt_mode": "rmt_rte_sel", "include_rmt": False, "include_raw_rte": False, "include_dock2": True},
    "rdkit2d_raw_rte": {
        "rmt_mode": "rmt_rte_sel",
        "include_rmt": False,
        "include_raw_rte": True,
        "include_dock2": False,
        "disable_constant_filter": True,
    },
    "rdkit2d_rmt_sel": {"rmt_mode": "rmt_rte_sel", "include_rmt": True, "include_raw_rte": False, "include_dock2": False},
    "rdkit2d_rmt_rec": {"rmt_mode": "rmt_rte_rec", "include_rmt": True, "include_raw_rte": False, "include_dock2": False},
    "rdkit2d_rmt_sel_sum": {"rmt_mode": "rmt_rte_sel", "include_rmt_sum": True, "include_raw_rte": False, "include_dock2": False},
    "rdkit2d_rmt_rec_sum": {"rmt_mode": "rmt_rte_rec", "include_rmt_sum": True, "include_raw_rte": False, "include_dock2": False},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DMPNN+GBM on RMT-RTE features")
    p.add_argument("--rmt-dir", required=True, help="Output directory from scripts/rmt_filter.py")
    p.add_argument("--out-dir", default="pipeline_runs/dmpnn_rmt_rte_test")
    p.add_argument("--split-id", default="split_00", help="Split id or comma-list")
    p.add_argument("--modes", default="rmt_rte_sel,rmt_rte_rec")
    p.add_argument(
        "--feature-sets",
        default="",
        help=(
            "Optional comma-list of ablation feature sets. Supported: "
            "rdkit2d,rdkit2d_dock2,rdkit2d_raw_rte,rdkit2d_rmt_sel,rdkit2d_rmt_rec,"
            "rdkit2d_rmt_sel_sum,rdkit2d_rmt_rec_sum. "
            "When set, --modes is ignored."
        ),
    )
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--hidden-size", type=int, default=300)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--ffn-hidden", type=int, default=300)
    p.add_argument("--ffn-layers", type=int, default=2)
    p.add_argument("--hgb-max-iter", type=int, default=400)
    p.add_argument("--hgb-learning-rate", type=float, default=0.05)
    return p.parse_args()


def comma_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


# ============================================================ ATOM / BOND FEATURES
ATOM_SYMBOLS = ["B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"]
DEGREES = [0, 1, 2, 3, 4, 5]
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
NUM_HS = [0, 1, 2, 3, 4]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
CHIRAL_TAGS = [0, 1, 2, 3]


def onehot(value, choices):
    vec = [0] * (len(choices) + 1)
    vec[choices.index(value) if value in choices else len(choices)] = 1
    return vec


def atom_features(atom):
    return (
        onehot(atom.GetSymbol(), ATOM_SYMBOLS)
        + onehot(atom.GetTotalDegree(), DEGREES)
        + onehot(atom.GetFormalCharge(), FORMAL_CHARGES)
        + onehot(int(atom.GetChiralTag()), CHIRAL_TAGS)
        + onehot(int(atom.GetTotalNumHs()), NUM_HS)
        + onehot(atom.GetHybridization(), HYBRIDIZATIONS)
        + [1 if atom.GetIsAromatic() else 0]
        + [atom.GetMass() * 0.01]
    )


def bond_features(bond):
    bt = bond.GetBondType()
    return [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing(),
    ] + onehot(int(bond.GetStereo()), [0, 1, 2, 3, 4, 5])


_PROBE = Chem.MolFromSmiles("CC")
ATOM_FDIM = len(atom_features(_PROBE.GetAtomWithIdx(0)))
BOND_FDIM = len(bond_features(_PROBE.GetBondWithIdx(0)))


class MolGraph:
    def __init__(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        self.n_atoms = mol.GetNumAtoms()
        self.n_bonds = 0
        self.f_atoms = [atom_features(a) for a in mol.GetAtoms()]
        self.f_bonds = []
        self.a2b = [[] for _ in range(self.n_atoms)]
        self.b2a = []
        self.b2revb = []
        for bond in mol.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bf = bond_features(bond)
            self.f_bonds.append(self.f_atoms[a1] + bf)
            b1 = self.n_bonds
            self.f_bonds.append(self.f_atoms[a2] + bf)
            b2 = self.n_bonds + 1
            self.a2b[a2].append(b1)
            self.b2a.append(a1)
            self.a2b[a1].append(b2)
            self.b2a.append(a2)
            self.b2revb.append(b2)
            self.b2revb.append(b1)
            self.n_bonds += 2


_MG_CACHE: dict[str, MolGraph] = {}


def get_molgraph(smiles: str) -> MolGraph:
    mg = _MG_CACHE.get(smiles)
    if mg is None:
        mg = MolGraph(smiles)
        _MG_CACHE[smiles] = mg
    return mg


class BatchMolGraph:
    def __init__(self, items):
        f_atoms = [[0] * ATOM_FDIM]
        f_bonds = [[0] * (ATOM_FDIM + BOND_FDIM)]
        a2b, b2a, b2revb = [[]], [0], [0]
        atom_batch = [-1]
        n_atoms, n_bonds = 1, 1
        for mol_id, item in enumerate(items):
            mg = item if isinstance(item, MolGraph) else get_molgraph(item)
            for f in mg.f_atoms:
                f_atoms.append(f)
            for f in mg.f_bonds:
                f_bonds.append(f)
            for a in range(mg.n_atoms):
                a2b.append([b + n_bonds for b in mg.a2b[a]])
            for b in range(mg.n_bonds):
                b2a.append(n_atoms + mg.b2a[b])
                b2revb.append(n_bonds + mg.b2revb[b])
            atom_batch.extend([mol_id] * mg.n_atoms)
            n_atoms += mg.n_atoms
            n_bonds += mg.n_bonds
        self.n_atoms = n_atoms
        self.n_bonds = n_bonds
        self.n_mols = len(items)
        atom_batch[0] = self.n_mols
        self.atom_batch = np.array(atom_batch, dtype=np.int64)
        max_b = max(1, max(len(x) for x in a2b))
        self.f_atoms = np.array(f_atoms, dtype=np.float32)
        self.f_bonds = np.array(f_bonds, dtype=np.float32)
        self.a2b = np.array([row + [0] * (max_b - len(row)) for row in a2b], dtype=np.int64)
        self.b2a = np.array(b2a, dtype=np.int64)
        self.b2revb = np.array(b2revb, dtype=np.int64)


class DMPNNEncoder(nn.Module):
    def __init__(self, hidden_size=300, depth=3, dropout=0.0, aggregation="multi"):
        super().__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.aggregation = aggregation
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, hidden_size, bias=False)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(ATOM_FDIM + hidden_size, hidden_size)

    @property
    def out_dim(self):
        return self.hidden_size * (3 if self.aggregation == "multi" else 1)

    def _readout(self, atom_hiddens, atom_batch, n_mols):
        hd = atom_hiddens.size(1)
        idx = atom_batch.unsqueeze(1).expand(-1, hd)
        s = torch.zeros(n_mols + 1, hd, device=atom_hiddens.device).scatter_add(0, idx, atom_hiddens)
        cnt = torch.zeros(n_mols + 1, device=atom_hiddens.device).scatter_add(
            0, atom_batch, torch.ones_like(atom_batch, dtype=atom_hiddens.dtype)
        )
        s, cnt = s[:n_mols], cnt[:n_mols].clamp(min=1.0).unsqueeze(1)
        mx = torch.full((n_mols + 1, hd), float("-inf"), device=atom_hiddens.device)
        mx = mx.scatter_reduce(0, idx, atom_hiddens, reduce="amax", include_self=False)[:n_mols]
        mx = torch.nan_to_num(mx, nan=0.0)
        return torch.cat([s / cnt, s / 100.0, mx], dim=1)

    def forward(self, f_atoms, f_bonds, a2b, b2a, b2revb, atom_batch, n_mols):
        input_b = self.W_i(f_bonds)
        message = self.act(input_b)
        for _ in range(self.depth - 1):
            nei_message = message[a2b]
            a_msg = nei_message.sum(dim=1)
            rev_msg = message[b2revb]
            message = self.W_h(a_msg[b2a] - rev_msg)
            message = self.act(input_b + message)
            message = self.dropout(message)
        nei_message = message[a2b]
        a_msg = nei_message.sum(dim=1)
        a_input = torch.cat([f_atoms, a_msg], dim=1)
        atom_hiddens = self.dropout(self.act(self.W_o(a_input)))
        return self._readout(atom_hiddens, atom_batch, n_mols)


class DMPNNClassifier(nn.Module):
    def __init__(self, n_global_features, hidden_size=300, depth=3, dropout=0.15, ffn_hidden=300, ffn_num_layers=2):
        super().__init__()
        self.encoder = DMPNNEncoder(hidden_size, depth, dropout, aggregation="multi")
        layers = [nn.Dropout(dropout), nn.Linear(self.encoder.out_dim + n_global_features, ffn_hidden), nn.ReLU()]
        for _ in range(ffn_num_layers - 2):
            layers += [nn.Dropout(dropout), nn.Linear(ffn_hidden, ffn_hidden), nn.ReLU()]
        layers += [nn.Dropout(dropout), nn.Linear(ffn_hidden, 1)]
        self.ffn = nn.Sequential(*layers)

    def forward(self, batch, global_feats):
        f_atoms, f_bonds, a2b, b2a, b2revb, atom_batch, n_mols = batch
        mv = self.encoder(f_atoms, f_bonds, a2b, b2a, b2revb, atom_batch, n_mols)
        return self.ffn(torch.cat([mv, global_feats], dim=1)).squeeze(1)


class FeatureScaler:
    def fit(self, x, filter_constant: bool = True):
        self.median_ = np.nanmedian(x, axis=0)
        xf = np.where(np.isnan(x), self.median_, x)
        self.mean_ = xf.mean(axis=0)
        self.std_ = xf.std(axis=0)
        self.filter_constant_ = bool(filter_constant)
        self.keep_ = self.std_ > 1e-6 if self.filter_constant_ else np.ones_like(self.std_, dtype=bool)
        self.std_[~self.keep_] = 1.0
        self.std_[self.std_ <= 1e-12] = 1.0
        return self

    def transform(self, x):
        xf = np.where(np.isnan(x), self.median_, x)
        xz = np.clip((xf - self.mean_) / self.std_, -5.0, 5.0)
        return xz[:, self.keep_].astype(np.float32)


class NoamLR:
    def __init__(self, opt, warmup_epochs, total_epochs, steps_per_epoch, init_lr, max_lr, final_lr):
        self.opt = opt
        self.warmup_steps = max(1, int(warmup_epochs * steps_per_epoch))
        self.total_steps = max(self.warmup_steps + 1, int(total_epochs * steps_per_epoch))
        self.init_lr, self.max_lr, self.final_lr = init_lr, max_lr, final_lr
        self.step_num = 0
        self.linear_inc = (max_lr - init_lr) / self.warmup_steps
        decay_steps = self.total_steps - self.warmup_steps
        self.gamma = (final_lr / max_lr) ** (1.0 / decay_steps) if decay_steps > 0 else 1.0
        self._set(init_lr)

    def _set(self, lr):
        for g in self.opt.param_groups:
            g["lr"] = lr

    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            lr = self.init_lr + self.step_num * self.linear_inc
        else:
            lr = max(self.final_lr, self.max_lr * self.gamma ** (self.step_num - self.warmup_steps))
        self._set(lr)


def batch_tensors(smiles_subset, device):
    bmg = BatchMolGraph(list(smiles_subset))
    return (
        torch.from_numpy(bmg.f_atoms).to(device),
        torch.from_numpy(bmg.f_bonds).to(device),
        torch.from_numpy(bmg.a2b).to(device),
        torch.from_numpy(bmg.b2a).to(device),
        torch.from_numpy(bmg.b2revb).to(device),
        torch.from_numpy(bmg.atom_batch).to(device),
        bmg.n_mols,
    )


def predict_proba(model, smiles_arr, xg, indices, device, batch_size=256):
    model.eval()
    out = np.zeros(len(indices), dtype=np.float32)
    with torch.no_grad():
        for k in range(0, len(indices), batch_size):
            sub = indices[k:k + batch_size]
            batch = batch_tensors([smiles_arr[i] for i in sub], device)
            g = torch.from_numpy(xg[sub]).to(device)
            out[k:k + len(sub)] = torch.sigmoid(model(batch, g)).cpu().numpy()
    return out


def docking_engineered(dock):
    dmin = dock.min(axis=1, keepdims=True)
    dmax = dock.max(axis=1, keepdims=True)
    dmean = dock.mean(axis=1, keepdims=True)
    dstd = dock.std(axis=1, keepdims=True)
    drange = dmax - dmin
    best2 = np.sort(dock, axis=1)[:, :2].mean(axis=1, keepdims=True)
    ranks = dock.argsort(axis=1).argsort(axis=1).astype(np.float64) / (dock.shape[1] - 1)
    return np.concatenate([dock, dmin, dmax, dmean, dstd, drange, best2, ranks], axis=1)


def docking_engineered_names() -> list[str]:
    return DOCK_COLS + ["dock_min", "dock_max", "dock_mean", "dock_std", "dock_range", "dock_best2"] + [
        f"dock_rank_{c}" for c in DOCK_COLS
    ]


def load_rdkit2d(smiles: np.ndarray) -> np.ndarray:
    if RDKIT_CACHE.exists():
        arr = np.load(RDKIT_CACHE)
        if arr.shape[0] == len(smiles):
            return arr.astype(np.float64)
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append(np.full(len(Descriptors.descList), np.nan))
        else:
            v = np.array([fn(mol) for _, fn in Descriptors.descList], dtype=np.float64)
            v[~np.isfinite(v)] = np.nan
            rows.append(v)
    arr = np.array(rows, dtype=np.float64)
    RDKIT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(RDKIT_CACHE, arr)
    return arr


def rdkit2d_names() -> list[str]:
    return [name for name, _ in Descriptors.descList]


def load_raw_rte(all_ids: np.ndarray) -> tuple[np.ndarray, list[str]]:
    rte = pd.read_csv(RTE_CSV)
    rte["ligand_id"] = rte["ligand_id"].astype(int)
    rte_cols = [c for c in rte.columns if c != "ligand_id"]
    rte = rte.set_index("ligand_id")
    missing = sorted(set(map(int, all_ids)) - set(map(int, rte.index)))
    if missing:
        raise ValueError(f"Missing {len(missing)} ligand_id values in {RTE_CSV}: {missing[:10]}")
    return rte.loc[all_ids, rte_cols].to_numpy(dtype=np.float64), rte_cols


def split_number(split_id: str) -> int:
    return int(split_id.rsplit("_", 1)[1])


def set_run_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_global_features(
    feature_set: str | None,
    rmt_cols: list[str],
    all_rmt: pd.DataFrame,
    data: pd.DataFrame,
    full_idx: np.ndarray,
    all_ids: np.ndarray,
) -> tuple[np.ndarray, list[str], dict]:
    pieces = []
    names = []
    counts = {"n_rmt_features": 0, "n_docking_features": 0, "n_rdkit2d_features": 0}

    if feature_set is None:
        rmt_x = all_rmt[rmt_cols].to_numpy(dtype=np.float64)
        dock_eng = docking_engineered(data.loc[full_idx, DOCK_COLS].to_numpy(dtype=np.float64))
        rdkit2d_all = load_rdkit2d(data["SMILES"].to_numpy())[full_idx]
        pieces = [rmt_x, dock_eng, rdkit2d_all]
        names = rmt_cols + docking_engineered_names() + rdkit2d_names()
        counts = {
            "n_rmt_features": len(rmt_cols),
            "n_docking_features": len(docking_engineered_names()),
            "n_rdkit2d_features": len(rdkit2d_names()),
        }
    else:
        spec = FEATURE_SET_SPECS[feature_set]
        if spec.get("include_rmt"):
            pieces.append(all_rmt[rmt_cols].to_numpy(dtype=np.float64))
            names.extend(rmt_cols)
            counts["n_rmt_features"] = len(rmt_cols)
        if spec.get("include_raw_rte"):
            raw_rte, raw_rte_cols = load_raw_rte(all_ids)
            pieces.append(raw_rte)
            names.extend(raw_rte_cols)
            counts["n_rmt_features"] = len(raw_rte_cols)
        if spec.get("include_rmt_sum"):
            rmt_sum = all_rmt[rmt_cols].to_numpy(dtype=np.float64).sum(axis=1, keepdims=True)
            pieces.append(rmt_sum)
            names.append("rmt_rte_sum")
            counts["n_rmt_features"] = 1
        if spec.get("include_dock2"):
            pieces.append(data.loc[full_idx, DOCK2_COLS].to_numpy(dtype=np.float64))
            names.extend(DOCK2_COLS)
            counts["n_docking_features"] = len(DOCK2_COLS)
        rdkit2d_all = load_rdkit2d(data["SMILES"].to_numpy())[full_idx]
        pieces.append(rdkit2d_all)
        names.extend(rdkit2d_names())
        counts["n_rdkit2d_features"] = len(rdkit2d_names())

    return np.concatenate(pieces, axis=1).astype(np.float64), names, counts


def best_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f1 = f1_score(y_true, proba >= t, zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


def metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "threshold": float(threshold),
    }


def train_dmpnn(
    mode: str,
    split_id: str,
    data: pd.DataFrame,
    rmt_dir: Path,
    out_dir: Path,
    args: argparse.Namespace,
    device,
    feature_set: str | None = None,
):
    if feature_set is not None and feature_set not in FEATURE_SET_SPECS:
        raise ValueError(f"Unknown feature set: {feature_set}")

    spec = FEATURE_SET_SPECS[feature_set] if feature_set is not None else {}
    run_name = feature_set or mode
    rmt_mode = spec["rmt_mode"] if feature_set is not None else mode
    filter_constant = not bool(spec.get("disable_constant_filter", False))
    set_run_seed(args.seed + split_number(split_id))

    mode_dir = out_dir / run_name / split_id
    mode_dir.mkdir(parents=True, exist_ok=True)

    train_rmt = pd.read_csv(rmt_dir / split_id / f"{rmt_mode}_train.csv")
    test_rmt = pd.read_csv(rmt_dir / split_id / f"{rmt_mode}_test.csv")
    train_rmt["ligand_id"] = train_rmt["ligand_id"].astype(int)
    test_rmt["ligand_id"] = test_rmt["ligand_id"].astype(int)
    rmt_cols = [c for c in train_rmt.columns if c not in {"split_id", "set", "ligand_id", "activity"}]

    labelled = data[data["activity"].notna()].copy()
    labelled["activity"] = labelled["activity"].astype(int)
    id_to_full = {int(lid): i for i, lid in enumerate(data["ligand_id"].astype(int).values)}

    train_ids = train_rmt["ligand_id"].to_numpy(dtype=int)
    test_ids = test_rmt["ligand_id"].to_numpy(dtype=int)
    all_ids = np.concatenate([train_ids, test_ids])
    all_rmt = pd.concat([train_rmt, test_rmt], ignore_index=True)

    data_idx = data.set_index("ligand_id")
    smiles = data_idx.loc[all_ids, "SMILES"].to_numpy()
    y = data_idx.loc[all_ids, "activity"].to_numpy(dtype=np.float32)

    full_idx = np.array([id_to_full[int(lid)] for lid in all_ids], dtype=int)
    x_global, feature_names, feature_counts = build_global_features(feature_set, rmt_cols, all_rmt, data, full_idx, all_ids)

    n_train_full = len(train_ids)
    train_pool = np.arange(n_train_full)
    test_idx = np.arange(n_train_full, len(all_ids))

    rng = np.random.RandomState(args.seed + split_number(split_id))
    perm = train_pool.copy()
    rng.shuffle(perm)
    n_val = max(30, int(args.val_frac * len(perm)))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    for smi in smiles:
        get_molgraph(smi)

    scaler = FeatureScaler().fit(x_global[train_idx], filter_constant=filter_constant)
    x_scaled = scaler.transform(x_global)

    model = DMPNNClassifier(
        n_global_features=x_scaled.shape[1],
        hidden_size=args.hidden_size,
        depth=args.depth,
        dropout=args.dropout,
        ffn_hidden=args.ffn_hidden,
        ffn_num_layers=args.ffn_layers,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    steps_per_epoch = max(1, int(np.ceil(len(train_idx) / args.batch_size)))
    sched = NoamLR(opt, 2, args.epochs, steps_per_epoch, 1e-4, 1e-3, 1e-4)
    pos = float(y[train_idx].sum())
    neg = float(len(train_idx) - pos)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)], device=device))

    best_auc, best_state, no_improve = -1.0, None, 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        order = train_idx.copy()
        rng.shuffle(order)
        for k in range(0, len(order), args.batch_size):
            sub = order[k:k + args.batch_size]
            batch = batch_tensors([smiles[i] for i in sub], device)
            g = torch.from_numpy(x_scaled[sub]).to(device)
            target = torch.from_numpy(y[sub]).to(device)
            opt.zero_grad()
            loss_fn(model(batch, g), target).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
        val_p = predict_proba(model, smiles, x_scaled, val_idx, device)
        val_auc = float(roc_auc_score(y[val_idx], val_p))
        if val_auc > best_auc:
            best_auc, no_improve = val_auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        if no_improve >= args.patience:
            break
    train_time = time.time() - t0
    model.load_state_dict(best_state)

    dmpnn_val = predict_proba(model, smiles, x_scaled, val_idx, device)
    dmpnn_test = predict_proba(model, smiles, x_scaled, test_idx, device)

    sample_weight = np.where(y[train_idx] == 1, len(train_idx) / (2 * max(pos, 1.0)), len(train_idx) / (2 * max(neg, 1.0)))
    hgb = HistGradientBoostingClassifier(
        max_iter=args.hgb_max_iter,
        learning_rate=args.hgb_learning_rate,
        l2_regularization=0.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=args.seed,
    )
    hgb.fit(x_scaled[train_idx], y[train_idx].astype(int), sample_weight=sample_weight)
    hgb_val = hgb.predict_proba(x_scaled[val_idx])[:, 1]
    hgb_test = hgb.predict_proba(x_scaled[test_idx])[:, 1]

    best_w, best_blend_auc = 1.0, -1.0
    for w in np.linspace(0.0, 1.0, 51):
        blend_val = w * dmpnn_val + (1.0 - w) * hgb_val
        auc = float(roc_auc_score(y[val_idx], blend_val))
        if auc > best_blend_auc:
            best_w, best_blend_auc = float(w), auc
    blend_val = best_w * dmpnn_val + (1.0 - best_w) * hgb_val
    blend_test = best_w * dmpnn_test + (1.0 - best_w) * hgb_test

    thresholds = {}
    val_metrics = {}
    test_metrics = {}
    for name, val_p, test_p in [
        ("dmpnn", dmpnn_val, dmpnn_test),
        ("hgb", hgb_val, hgb_test),
        ("blend", blend_val, blend_test),
    ]:
        threshold, _ = best_threshold(y[val_idx].astype(int), val_p)
        thresholds[name] = threshold
        val_metrics[name] = metrics(y[val_idx].astype(int), val_p, threshold)
        test_metrics[name] = metrics(y[test_idx].astype(int), test_p, threshold)

    torch.save(model.state_dict(), mode_dir / "dmpnn_model.pt")
    joblib.dump(hgb, mode_dir / "hgb_model.pkl")
    joblib.dump(scaler, mode_dir / "feature_scaler.pkl")
    np.save(mode_dir / "selected_rmt_feature_names.npy", np.array(rmt_cols, dtype=object))
    np.save(mode_dir / "global_feature_names.npy", np.array(feature_names, dtype=object))
    np.save(mode_dir / "global_feature_names_after_filter.npy", np.array(feature_names, dtype=object)[scaler.keep_])

    pred = pd.DataFrame(
        {
            "ligand_id": all_ids[test_idx],
            "activity": y[test_idx].astype(int),
            "dmpnn_proba": dmpnn_test,
            "hgb_proba": hgb_test,
            "blend_proba": blend_test,
        }
    )
    pred.to_csv(mode_dir / "test_predictions.csv", index=False)

    result = {
        "mode": run_name,
        "rmt_mode": rmt_mode,
        "feature_set": feature_set or "full_legacy",
        "split_id": split_id,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "n_rmt_features": int(feature_counts["n_rmt_features"]),
        "n_docking_features": int(feature_counts["n_docking_features"]),
        "n_rdkit2d_features": int(feature_counts["n_rdkit2d_features"]),
        "n_global_before_filter": int(x_global.shape[1]),
        "n_global_after_filter": int(x_scaled.shape[1]),
        "constant_filter_threshold": 1e-6 if filter_constant else None,
        "constant_filter_enabled": bool(filter_constant),
        "epochs_run": int(epoch + 1),
        "train_time_s": round(train_time, 1),
        "best_val_auc_dmpnn_epoch": float(best_auc),
        "blend_weight_dmpnn": float(best_w),
        "blend_weight_hgb": float(1.0 - best_w),
        "thresholds": thresholds,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "outputs": {
            "dmpnn_model": "dmpnn_model.pt",
            "hgb_model": "hgb_model.pkl",
            "feature_scaler": "feature_scaler.pkl",
            "global_feature_names": "global_feature_names.npy",
            "global_feature_names_after_filter": "global_feature_names_after_filter.npy",
            "test_predictions": "test_predictions.csv",
        },
    }
    (mode_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"{run_name} {split_id}: test blend ROC={test_metrics['blend']['roc_auc']:.4f} "
        f"PR={test_metrics['blend']['pr_auc']:.4f} w={best_w:.2f} "
        f"features={x_scaled.shape[1]} time={train_time:.1f}s",
        flush=True,
    )
    return result


def main() -> None:
    args = parse_args()
    out_dir = BASE / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    rmt_dir = BASE / args.rmt_dir if not Path(args.rmt_dir).is_absolute() else Path(args.rmt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    data = pd.read_csv(DATA_CSV)
    data["ligand_id"] = data["ligand_id"].astype(int)
    modes = comma_list(args.modes)
    feature_sets = comma_list(args.feature_sets)
    splits = comma_list(args.split_id)

    results = []
    for split_id in splits:
        if feature_sets:
            for feature_set in feature_sets:
                if feature_set not in FEATURE_SET_SPECS:
                    raise ValueError(f"Unknown feature set: {feature_set}. Supported: {sorted(FEATURE_SET_SPECS)}")
                rmt_mode = FEATURE_SET_SPECS[feature_set]["rmt_mode"]
                results.append(train_dmpnn(rmt_mode, split_id, data, rmt_dir, out_dir, args, device, feature_set=feature_set))
        else:
            for mode in modes:
                results.append(train_dmpnn(mode, split_id, data, rmt_dir, out_dir, args, device))

    flat = []
    for r in results:
        for model_name, vals in r["test_metrics"].items():
            flat.append({
                "mode": r["mode"],
                "feature_set": r["feature_set"],
                "rmt_mode": r["rmt_mode"],
                "split_id": r["split_id"],
                "model": model_name,
                "roc_auc": vals["roc_auc"],
                "pr_auc": vals["pr_auc"],
                "f1": vals["f1"],
                "accuracy": vals["accuracy"],
                "threshold": vals["threshold"],
                "blend_weight_dmpnn": r["blend_weight_dmpnn"],
                "n_rmt_features": r["n_rmt_features"],
                "n_docking_features": r["n_docking_features"],
                "n_rdkit2d_features": r["n_rdkit2d_features"],
                "n_global_after_filter": r["n_global_after_filter"],
                "epochs_run": r["epochs_run"],
                "train_time_s": r["train_time_s"],
            })
    pd.DataFrame(flat).to_csv(out_dir / "metrics_summary.csv", index=False)
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
    print(f"Summary: {out_dir / 'metrics_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
