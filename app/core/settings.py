"""Environment settings (pydantic-settings).

Values come from process environment variables only (Vercel project settings, CI, or an
exported shell). The .env file is deliberately not read here. Empty variables count as unset, so
the blank lines in .env.example fall back to the defaults below.

Validation errors never include the submitted value (hide_input_in_errors), so a key pasted into
the wrong variable is not echoed into logs.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.jev_client import JevClient
from app.core.model_config import DEFAULT_CONFIG_PATH
from app.core.policy import RoutingConfig

# vercel.json: functions."app/main.py".maxDuration. The request budget must end before it.
FUNCTION_MAX_DURATION_SECONDS = 60


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", env_ignore_empty=True, hide_input_in_errors=True)

    database_url: str | None = None  # DATABASE_URL, e.g. Neon pooled postgresql+psycopg://...
    redis_url: str | None = None  # REDIS_URL, e.g. Upstash rediss://...
    ai_gateway_api_key: SecretStr | None = None  # AI_GATEWAY_API_KEY
    models_config_path: Path = DEFAULT_CONFIG_PATH  # MODELS_CONFIG_PATH
    # CONFIDENCE_THRESHOLD; the default is RoutingConfig's, so the two cannot disagree.
    confidence_threshold: float = Field(default=RoutingConfig.confidence_threshold, ge=0, le=1, allow_inf_nan=False)
    strong_daily_cap: int = Field(default=100, ge=0)  # STRONG_DAILY_CAP; 0 disables strong routing
    request_budget_seconds: float = Field(  # REQUEST_BUDGET_SECONDS
        default=50.0, gt=0, lt=FUNCTION_MAX_DURATION_SECONDS, allow_inf_nan=False
    )
    # JEV_TIMEOUT_SECONDS: limit for the whole Jev call. Jev time counts against REQUEST_BUDGET_SECONDS.
    jev_timeout_seconds: float = Field(default=JevClient.DEFAULT_TIMEOUT_SECONDS, gt=0, allow_inf_nan=False)

    @field_validator("database_url", "redis_url", "ai_gateway_api_key", mode="before")
    @classmethod
    def blank_means_unset(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
