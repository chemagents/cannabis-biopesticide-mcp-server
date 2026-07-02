"""Configuration for the cannabis-biopesticide MCP server (pydantic-settings).

Values are read from environment / .env with the ``CANPEST_`` prefix, mirroring the
other CoScientist MCP servers (``HERACLEUM_``, ``TOX_`` ...).
"""
from functools import lru_cache
from pathlib import Path

from pydantic import Field
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

    # --- QSAR model backend (open analogue of DMPNN-SD; HGB reproduces ~0.93) ---
    model_backend: str = Field(default="hgb")       # "hgb" | "catboost"
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
    s3_access_key: str = Field(default="")
    s3_secret_key: str = Field(default="")
    s3_bucket_name: str = Field(default="")
    s3_url_expiration: int = Field(default=3600)

    @property
    def use_s3(self) -> bool:
        return bool(self.s3_endpoint_url and self.s3_bucket_name and self.s3_access_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
