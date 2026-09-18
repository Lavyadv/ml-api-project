"""Central configuration, driven by environment variables.

The twelve-factor principle at work: the same code runs on a laptop, in
CI, and in production — only the environment differs. Nothing in this
app reads os.getenv() directly any more; everything goes through the
`settings` object below, so there is exactly one place to look up what
is configurable and what its default is.

Values are resolved in this order (first wins):
    1. a real environment variable       MAX_BATCH_SIZE=50 uvicorn ...
    2. a line in the .env file           MAX_BATCH_SIZE=50
    3. the default declared here

.env holds local values and is gitignored; .env.example is committed and
documents every variable without carrying real values.
"""
from __future__ import annotations

from pathlib import Path
from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Every value this app used to hardcode."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        # Ignore unrelated environment variables rather than failing to
        # boot because the shell happens to export something we don't know.
        extra="ignore",
    )

    # --- API metadata (shown on /docs) -----------------------------------
    api_title: str = "Iris Classifier API"
    api_description: str = "Serves a scikit-learn Iris species classifier over HTTP."
    api_version: str = "1.0.0"

    # --- model -----------------------------------------------------------
    model_path: Path = PROJECT_ROOT / "ml" / "saved_model" / "model.joblib"

    # --- logging ---------------------------------------------------------
    log_level: str = "INFO"
    log_dir: Path = PROJECT_ROOT / "logs"
    log_max_bytes: int = 1_000_000
    log_backup_count: int = 5

    # --- limits ----------------------------------------------------------
    # Guards against a caller posting 10 million rows and exhausting memory.
    max_batch_size: int = 100

    # Kept out of source control: provide it through .env locally and the
    # deployment environment in production. A missing value fails closed.
    api_key: SecretStr | None = None

    # Comma-separated origins avoid the JSON-list syntax environment variables
    # otherwise require. An empty value permits no browser origins.
    cors_origins: str = ""

    @property
    def allowed_cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @field_validator("model_path", "log_dir", mode="after")
    @classmethod
    def _resolve_against_project_root(cls, value: Path) -> Path:
        """Let .env use short relative paths without breaking on cwd.

        `MODEL_PATH=ml/saved_model/model.joblib` should mean the same
        thing no matter which directory uvicorn was started from.
        """
        return value if value.is_absolute() else (PROJECT_ROOT / value)

    @field_validator("log_level", mode="after")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        """Accept `debug`, `Debug`, `DEBUG` alike, and reject nonsense loudly."""
        level = value.upper()
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if level not in valid:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(valid)}, got {value!r}")
        return level


# One instance, imported everywhere. Building it at import time means a
# bad .env fails at startup with a clear Pydantic error, rather than
# halfway through serving a request.
settings = Settings()
