"""Load the canpest dataset (SMILES + activity + 6 docking scores) and RTE features."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from .config import get_settings

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)

# docking column -> (gene, protein name, PDB/UniProt, insecticidal role)
PROTEINS = {
    "3rif":   {"gene": "GluCl",    "name": "Glutamate-gated chloride channel", "pdb": "3RIF",  "role": "lethal (ion channel)"},
    "d8v7j0": {"gene": "AChE1",    "name": "Acetylcholinesterase 1",           "pdb": "D8V7J0", "role": "lethal (enzyme)"},
    "8sfy":   {"gene": "UGT202A2", "name": "UDP-glucuronosyltransferase 202A2", "pdb": "8SFY", "role": "detoxification (UGT)"},
    "8udb":   {"gene": "GSTM12",   "name": "Glutathione S-transferase m12",     "pdb": "8UDB", "role": "detoxification (GST)"},
    "2imi":   {"gene": "agGSTe2",  "name": "Glutathione S-transferase epsilon 2", "pdb": "2IMI", "role": "detoxification (GST)"},
    "8v3d":   {"gene": "OR28",     "name": "Odorant receptor 28 (AgamOR28)",   "pdb": "8V3D",  "role": "chemosensory (repellent)"},
}
DOCK_COLS = list(PROTEINS.keys())


@dataclass
class Dataset:
    df: pd.DataFrame
    smiles: list[str]
    mols: list                       # RDKit Mol per row (parsed; None dropped)
    activity: np.ndarray             # float, NaN = unlabelled C. sativa metabolite
    dock: np.ndarray                 # (n, 6) docking scores (kcal/mol)
    ligand_ids: list[str]
    _canon_index: dict[str, int] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.df)

    @property
    def labelled_mask(self) -> np.ndarray:
        return ~np.isnan(self.activity)

    @property
    def active_mask(self) -> np.ndarray:
        return self.activity == 1.0

    @property
    def inactive_mask(self) -> np.ndarray:
        return self.activity == 0.0

    @property
    def metabolite_mask(self) -> np.ndarray:
        """Unlabelled C. sativa metabolites (the prediction set, DS1)."""
        return np.isnan(self.activity)

    def index_for_smiles(self, smiles: str) -> int | None:
        m = Chem.MolFromSmiles(smiles)
        return self._canon_index.get(Chem.MolToSmiles(m)) if m is not None else None


@lru_cache(maxsize=1)
def load_dataset() -> Dataset:
    settings = get_settings()
    path = Path(settings.data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}.")
    df = pd.read_csv(path, dtype={"ligand_id": str})
    smiles = df["SMILES"].tolist()
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    keep = [i for i, m in enumerate(mols) if m is not None]
    if len(keep) != len(mols):
        df = df.iloc[keep].reset_index(drop=True)
        smiles = [smiles[i] for i in keep]
        mols = [mols[i] for i in keep]
    canon_index: dict[str, int] = {}
    for i, m in enumerate(mols):
        canon_index.setdefault(Chem.MolToSmiles(m), i)
    return Dataset(
        df=df,
        smiles=smiles,
        mols=mols,
        activity=df["activity"].to_numpy(dtype=float),
        dock=df[DOCK_COLS].to_numpy(dtype=float),
        ligand_ids=df["ligand_id"].tolist(),
        _canon_index=canon_index,
    )


@lru_cache(maxsize=1)
def load_rte() -> tuple[pd.DataFrame, list[str]]:
    """Return the 390-column residue-term-energy feature table (3171 labelled ligands)."""
    settings = get_settings()
    df = pd.read_csv(settings.rte_path, dtype={"ligand_id": str})
    feat_cols = [c for c in df.columns if c != "ligand_id"]
    return df, feat_cols


@lru_cache(maxsize=2)
def load_split(scaffold: bool = False) -> pd.DataFrame:
    settings = get_settings()
    path = settings.split_scaffold_path if scaffold else settings.split_path
    return pd.read_csv(path, dtype={"ligand_id": str})
