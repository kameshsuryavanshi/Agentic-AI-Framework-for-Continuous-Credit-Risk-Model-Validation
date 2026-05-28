import os

from pathlib import Path

from dotenv import load_dotenv

from pydantic import Field

from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict
)

# =========================================================
# LOAD ENV VARIABLES
# =========================================================

load_dotenv()

# =========================================================
# ROOT PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

# =========================================================
# DATA DIRECTORIES
# =========================================================

DATA_DIR = BASE_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

# =========================================================
# DOCUMENTS
# =========================================================

DOCS_DIR = BASE_DIR / "docs"

REGULATORY_DOCS_DIR = (
    DOCS_DIR / "regulatory_docs"
)

# =========================================================
# OUTPUT DIRECTORIES
# =========================================================

OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR = OUTPUT_DIR / "models"

REPORT_DIR = OUTPUT_DIR / "reports"

LOG_DIR = OUTPUT_DIR / "logs"

VECTOR_DB_PATH = (
    OUTPUT_DIR / "faiss_index"
)

GRAPH_OUTPUT_DIR = (
    OUTPUT_DIR / "graph"
)

# =========================================================
# MODEL FILES
# =========================================================

PD_MODEL_FILE = (
    MODEL_DIR / "pd_model.pkl"
)

XGB_MODEL_FILE = (
    MODEL_DIR / "xgb_model.pkl"
)

FEATURE_COLUMNS_FILE = (
    MODEL_DIR / "feature_columns.pkl"
)

# =========================================================
# CREATE DIRECTORIES
# =========================================================

for path in [

    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,

    DOCS_DIR,
    REGULATORY_DOCS_DIR,

    OUTPUT_DIR,
    MODEL_DIR,
    REPORT_DIR,
    LOG_DIR,
    VECTOR_DB_PATH,
    GRAPH_OUTPUT_DIR

]:
    path.mkdir(
        parents=True,
        exist_ok=True
    )

# =========================================================
# SETTINGS
# =========================================================


class Settings(BaseSettings):

    # =====================================================
    # GOOGLE / GEMINI
    # =====================================================

    google_api_key: str = Field(
        default=""
    )

    gemini_model: str = Field(
        default="gemini-2.5-flash"
    )

    gemini_temperature: float = Field(
        default=0.1
    )

    # =====================================================
    # NEO4J
    # =====================================================

    neo4j_uri: str = Field(
        default="bolt://localhost:7687"
    )

    neo4j_username: str = Field(
        default="neo4j"
    )

    neo4j_password: str = Field(
        default="password"
    )

    neo4j_database: str = Field(
        default="neo4j"
    )

    # =====================================================
    # IFRS9 THRESHOLDS
    # =====================================================

    stage_2_dpd: int = Field(
        default=30
    )

    stage_3_dpd: int = Field(
        default=90
    )

    sicr_pd_multiplier: float = Field(
        default=2.0
    )

    high_risk_threshold: float = Field(
        default=0.25
    )

    # =====================================================
    # SCENARIO WEIGHTS
    # =====================================================

    base_scenario_weight: float = Field(
        default=0.60
    )

    adverse_scenario_weight: float = Field(
        default=0.30
    )

    severe_scenario_weight: float = Field(
        default=0.10
    )

    # =====================================================
    # DRIFT / VALIDATION
    # =====================================================

    psi_threshold: float = Field(
        default=0.2
    )

    drift_alert_threshold: float = Field(
        default=0.3
    )

    # =====================================================
    # VECTOR DB
    # =====================================================

    embedding_model: str = Field(
        default="models/embedding-001"
    )

    vector_top_k: int = Field(
        default=5
    )

    # =====================================================
    # LANGGRAPH
    # =====================================================

    max_agent_iterations: int = Field(
        default=10
    )

    # =====================================================
    # Pydantic Config
    # =====================================================

    model_config = SettingsConfigDict(

        env_file=".env",

        env_file_encoding="utf-8",

        extra="ignore",

        case_sensitive=False
    )

# =========================================================
# INITIALIZE SETTINGS
# =========================================================


settings = Settings()

# =========================================================
# BACKWARD COMPATIBILITY CONSTANTS
# =========================================================

GOOGLE_API_KEY = settings.google_api_key

GEMINI_MODEL = settings.gemini_model

EMBEDDING_MODEL = settings.embedding_model

# =========================================================
# NEO4J
# =========================================================

NEO4J_URI = settings.neo4j_uri

NEO4J_USERNAME = settings.neo4j_username

NEO4J_PASSWORD = settings.neo4j_password

NEO4J_DATABASE = settings.neo4j_database

# =========================================================
# IFRS9
# =========================================================

STAGE_2_DPD = settings.stage_2_dpd

STAGE_3_DPD = settings.stage_3_dpd

SICR_PD_MULTIPLIER = (
    settings.sicr_pd_multiplier
)

HIGH_RISK_THRESHOLD = (
    settings.high_risk_threshold
)

# =========================================================
# SCENARIOS
# =========================================================

BASE_SCENARIO_WEIGHT = (
    settings.base_scenario_weight
)

ADVERSE_SCENARIO_WEIGHT = (
    settings.adverse_scenario_weight
)

SEVERE_SCENARIO_WEIGHT = (
    settings.severe_scenario_weight
)

# =========================================================
# VALIDATION
# =========================================================

PSI_THRESHOLD = (
    settings.psi_threshold
)

DRIFT_ALERT_THRESHOLD = (
    settings.drift_alert_threshold
)

# =========================================================
# VECTOR DB
# =========================================================

VECTOR_TOP_K = settings.vector_top_k
