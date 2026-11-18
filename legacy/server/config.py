"""Configuration for the cannabis-biopesticide MCP server (pydantic-settings).

Values are read from environment / .env with the ``CANPEST_`` prefix, mirroring the
other CoScientist MCP servers (``HERACLEUM_``, ``TOX_`` ...).
"""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
DATA_DIR = PACKAGE_DIR / "data"
DEFAULT_ENV_FILE = PROJECT_DIR / ".env"
DEFAULT_ARTIFACTS_DIR = PROJECT_DIR / "artifacts"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CANPEST_",
        env_file=str(DEFAULT_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- MCP transport ---
    mcp_host: str = Field(default="0.0.0.0")
    mcp_port: int = Field(default=7331)
    mcp_path: str = Field(default="/mcp")

    # --- bundled data (from the authors' rmt_canpest package) ---
    data_path: str = Field(default=str(DATA_DIR / "canpest_data.csv"))          # 5920 x SMILES/activity/6-dock
    rte_path: str = Field(default=str(DATA_DIR / "rte.csv"))                     # 390 residue-term energies
    fp_rdkit2d_path: str = Field(default=str(DATA_DIR / "fp_rdkit2d.npy"))       # authors' exact 217 RDKit2D descriptors
    # Precomputed DMPNN-SD metabolite probabilities (the paper's headline model, open Chemprop).
    # Bundled like the docking scores: if present, `predict_biopesticides` uses these for the exact
    # ~41% candidate fraction (paper 1010/2465; 1152/2749 here); if absent, it falls back to the torch-free HGB analogue.
    dmpnn_pred_path: str = Field(default=str(DATA_DIR / "dmpnn_pred.csv"))       # cols: ligand_id, prob, proba_dmpnn, proba_hgb
    split_path: str = Field(default=str(DATA_DIR / "split_registry.csv"))       # 10 random splits
    split_scaffold_path: str = Field(default=str(DATA_DIR / "split_registry_scaffold.csv"))
    reference_dir: str = Field(default=str(DATA_DIR / "reference"))

    # --- docking columns -> proteins (PDB/gene) ---
    # 2imi=GSTE2, d8v7j0=ACHE1, 3rif=GluCl, 8sfy=UGT202A2, 8udb=GSTM12, 8v3d=OR28(AGAMOR28)
    binder_threshold: float = Field(default=-7.0)   # kcal/mol, "strong binder" cutoff

    # --- RMT feature selection (paper Section 3.3) ---
    rmt_inner_splits: int = Field(default=4)
    rmt_inner_train_frac: float = Field(default=0.7)
    rmt_rank_by: str = Field(default="hybrid")      # s_i * |rho_i|
    random_state: int = Field(default=42)

    # --- QSAR model stack (paper's DMPNN-SD = weighted DMPNN + HGB blend) ---
    # The production model is a soft-voting stack: p = w·p_DMPNN + (1-w)·p_HGB. The blend
    # weight and F1-optimal threshold below are the values selected on OOF CV in the authors'
    # pipeline; predict_biopesticides recomputes the blend from the bundled component columns.
    blend_w_dmpnn: float = Field(default=0.62)
    blend_threshold: float = Field(default=0.45)
    model_backend: str = Field(default="hgb")       # torch-free fallback when no stack preds bundled
    model_cache_dir: str = Field(default=str(PACKAGE_DIR / "model_cache"))
    retrain: bool = Field(default=False)

    # --- applicability domain / candidate cutoff ---
    ad_k_neighbors: int = Field(default=5)
    ad_threshold_sigma: float = Field(default=2.0)
    candidate_probability: float = Field(default=0.70)   # paper: >0.7 -> 1010 candidates

    # --- chemical-space clustering (differential scaffold fingerprint) ---
    morgan_radius: int = Field(default=2)
    morgan_nbits: int = Field(default=2048)
    cluster_fingerprint: str = Field(default="differential")
    tsne_perplexity: float = Field(default=30.0)

    # --- artifacts ---
    artifacts_dir: str = Field(default=str(DEFAULT_ARTIFACTS_DIR))
    artifact_url_base: str = Field(default="")
    s3_endpoint_url: str = Field(default="")
    s3_public_endpoint_url: str = Field(default="")
    s3_access_key: str = Field(default="")
    s3_secret_key: str = Field(default="")
    s3_bucket_name: str = Field(default="")
    s3_region: str = Field(default="us-east-1")
    s3_addressing_style: Literal["auto", "path", "virtual"] = Field(default="path")
    s3_ca_bundle: str = Field(default="")
    s3_allow_local_fallback: bool = Field(default=False)
    s3_url_expiration: int = Field(default=3600, ge=1, le=604800)

    @model_validator(mode="after")
    def validate_s3_configuration(self) -> "Settings":
        core = {
            "s3_endpoint_url": self.s3_endpoint_url.strip(),
            "s3_access_key": self.s3_access_key.strip(),
            "s3_secret_key": self.s3_secret_key.strip(),
            "s3_bucket_name": self.s3_bucket_name.strip(),
        }
        configured = [name for name, value in core.items() if value]
        if configured and len(configured) != len(core):
            missing = ", ".join(name for name, value in core.items() if not value)
            raise ValueError(f"partial S3 configuration; missing: {missing}")
        if (self.s3_public_endpoint_url.strip() or self.s3_ca_bundle.strip()) and not configured:
            raise ValueError("S3 public endpoint/CA bundle requires complete S3 configuration")
        return self

    @property
    def use_s3(self) -> bool:
        return bool(
            self.s3_endpoint_url.strip()
            and self.s3_bucket_name.strip()
            and self.s3_access_key.strip()
            and self.s3_secret_key.strip()
        )

    @property
    def s3_presign_endpoint_url(self) -> str:
        return self.s3_public_endpoint_url.strip() or self.s3_endpoint_url.strip()


@lru_cache
def get_settings() -> Settings:
    return Settings()
