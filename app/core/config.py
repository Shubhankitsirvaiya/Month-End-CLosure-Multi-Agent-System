"""Environment-backed settings for live model-driven orchestration."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed configuration loaded from ``.env`` and process environment.

    Pydantic performs type conversion and validation for us. For example, the
    text ``MAX_ORCHESTRATOR_ROUNDS=6`` becomes an integer and the constraints
    below reject unsafe values before an agent run starts.
    """

    # Tell Pydantic where local configuration comes from. Environment variables
    # still override .env values, which is useful in Docker and CI/CD.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Empty in source control because secrets never belong in Git. Runtime startup
    # fails clearly until a real key is supplied through .env or the environment.
    openai_api_key: str = ""
    # Chat and embedding models are separate because their jobs and pricing differ.
    openai_model: str = "gpt-5-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    # Keep policy retrieval local by default even though orchestration uses OpenAI.
    # Change to ``openai`` only when embedding-backed Chroma indexing is desired.
    policy_retrieval_mode: str = "local"

    # 0.0.0.0 exposes FastAPI outside its process/container; 127.0.0.1 would not.
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    # Streamlit uses this URL as an HTTP client; it never imports backend logic.
    streamlit_api_url: str = "http://localhost:8000"

    # This URL is the network boundary of the independently hosted A2A agent.
    valuation_agent_url: str = "http://localhost:9000"
    # Sent as an A2A-Version header so client/server contract changes are visible.
    a2a_version: str = "1.0"
    # A local fallback keeps month-end processing available during A2A downtime.
    a2a_fallback_local: bool = True

    # Source data is versionable; runtime output is local and ignored by Git.
    data_root: Path = Path("data")
    runtime_root: Path = Path("runtime")
    policy_root: Path = Path("data/policies")
    ground_truth_root: Path = Path("data/ground_truth")

    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "project-16-month-end-copilot"

    # The round budget prevents an autonomous supervisor from looping forever.
    max_orchestrator_rounds: int = Field(default=6, ge=2, le=20)
    # Retries are bounded for the same reason and should cover transient failures only.
    max_agent_retries: int = Field(default=2, ge=0, le=5)
    # Basis points are used because portfolio materiality scales with NAV.
    materiality_approval_bps: float = Field(default=10.0, ge=0)
    # Lower-confidence packages are escalated instead of being silently finalized.
    low_confidence_threshold: float = Field(default=0.72, ge=0, le=1)

    def ensure_runtime_directories(self) -> None:
        """Create local runtime directories if missing.

        ``parents=True`` creates intermediate folders; ``exist_ok=True`` makes
        startup idempotent, meaning repeated starts do not fail.
        """

        self.runtime_root.mkdir(parents=True, exist_ok=True)
        (self.runtime_root / "runs").mkdir(parents=True, exist_ok=True)
        (self.runtime_root / "chroma").mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return one cached settings object for the process.

    ``lru_cache(maxsize=1)`` avoids reparsing the environment on every FastAPI
    dependency injection while still keeping tests able to clear the cache.
    """

    # Constructing Settings triggers parsing, conversion, and validation.
    settings = Settings()
    # Create writable folders once so later components can assume they exist.
    settings.ensure_runtime_directories()
    return settings

