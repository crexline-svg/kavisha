"""Central configuration, loaded from environment / .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Smartphone ABSA Recommendation API"
    pipeline_version: str = "1.0.0"
    debug: bool = False
    # When true and the database has no phones, load the synthetic demo corpus
    # and run lexicon ABSA. Render sets this so the site is not empty.
    seed_demo: bool = False

    data_dir: Path = BASE_DIR / "data"
    database_url: str = ""

    # ---------- scraping ----------
    marketplace: str = "https://www.amazon.com"
    headless: bool = True
    browser_channel: str | None = None
    nav_timeout_ms: int = 45_000
    min_delay_s: float = 1.5
    max_delay_s: float = 4.0
    max_retries: int = 3
    block_resources: bool = True
    user_agent: str = DEFAULT_USER_AGENT
    locale: str = "en-US"
    timezone_id: str = "America/New_York"
    proxy_server: str | None = None

    # ---------- NLP ----------
    aspect_set: Literal["core", "extended"] = "core"
    absa_engine: Literal["auto", "llm", "lexicon"] = "auto"
    language_filter: str = "en"
    min_review_chars: int = 15

    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_batch_size: int = 10
    llm_timeout_s: int = 90
    llm_max_retries: int = 3
    llm_temperature: float = 0.0

    # ---------- aggregation / recommendation ----------
    neutral_weight: float = 0.5
    min_mentions_for_score: int = 3
    shrinkage_strength: float = 5.0

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"

    @property
    def storage_state_path(self) -> Path:
        """Reusable browser session (cookies) captured by `python run.py login`."""
        return self.data_dir / "storage_state.json"

    @property
    def sqlalchemy_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{(self.data_dir / 'smartphones.db').as_posix()}"

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.artifacts_dir, self.exports_dir):
            path.mkdir(parents=True, exist_ok=True)

    def llm_available(self) -> bool:
        return bool(self.llm_api_key and self.llm_api_key.strip())

    def resolved_absa_engine(self) -> Literal["llm", "lexicon"]:
        if self.absa_engine == "llm":
            return "llm"
        if self.absa_engine == "lexicon":
            return "lexicon"
        return "llm" if self.llm_available() else "lexicon"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings


settings = get_settings()
