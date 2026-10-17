"""Environment settings (pydantic-settings).

Values come from process environment variables only (Vercel project settings, CI, or an
exported shell). The .env file is deliberately not read here. Empty variables count as unset, so
the blank lines in .env.example fall back to the defaults below.

Validation errors never include the submitted value (hide_input_in_errors), so a key pasted into
the wrong variable is not echoed into logs.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.executor import CALL_MARGIN_SECONDS, MIN_CALL_SECONDS
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
    jevstation_api_key: SecretStr | None = None  # JEVSTATION_API_KEY (environment only; .env is not read)
    models_config_path: Path = DEFAULT_CONFIG_PATH  # MODELS_CONFIG_PATH
    # CONFIDENCE_THRESHOLD; the default is RoutingConfig's, so the two cannot disagree.
    confidence_threshold: float = Field(default=RoutingConfig.confidence_threshold, ge=0, le=1, allow_inf_nan=False)
    strong_daily_cap: int = Field(default=100, ge=0)  # STRONG_DAILY_CAP; 0 disables strong routing
    request_budget_seconds: float = Field(  # REQUEST_BUDGET_SECONDS
        default=50.0, gt=0, lt=FUNCTION_MAX_DURATION_SECONDS, allow_inf_nan=False
    )
    # JEV_TIMEOUT_SECONDS: limit for the whole Jev call. Jev time counts against REQUEST_BUDGET_SECONDS.
    jev_timeout_seconds: float = Field(default=JevClient.DEFAULT_TIMEOUT_SECONDS, gt=0, allow_inf_nan=False)
    rate_limit_per_min: int = Field(default=10, ge=1)  # RATE_LIMIT_PER_MIN: /api/route requests per client IP
    # CLIENT_IP_HEADER: trusted header holding the client IP (x-real-ip on Vercel); unset = socket peer.
    client_ip_header: str | None = None
    rate_limit_ip_salt: SecretStr | None = None  # RATE_LIMIT_IP_SALT: HMAC key for rate-limit identities
    # S2.14 response cache (app.infra.cache): off unless REDIS_URL, CACHE_NAMESPACE and CACHE_KEY_SECRET are all set.
    # CACHE_NAMESPACE: the deployment environment in every cache key (e.g. production); ":" is not allowed.
    cache_namespace: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9_-]{0,31}$")
    cache_key_secret: SecretStr | None = Field(default=None, min_length=32)  # CACHE_KEY_SECRET: HMAC key for keys
    cache_ttl_seconds: int = Field(default=86400, ge=1)  # CACHE_TTL_SECONDS

    @field_validator(
        "database_url",
        "redis_url",
        "ai_gateway_api_key",
        "jevstation_api_key",
        "client_ip_header",
        "rate_limit_ip_salt",
        "cache_namespace",
        "cache_key_secret",
        mode="before",
    )
    @classmethod
    def blank_means_unset(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("cache_namespace", mode="before")
    @classmethod
    def strip_cache_namespace(cls, value: object) -> object:
        # Before the pattern check, so a padded name ("preview\n") is accepted as "preview".
        return value.strip() if isinstance(value, str) else value

    @field_validator("client_ip_header")
    @classmethod
    def strip_header_name(cls, value: str | None) -> str | None:
        # A padded name ("x-real-ip ") would never match a request header, silently falling back to the peer.
        return value.strip() if value is not None else None

    @model_validator(mode="after")
    def jev_timeout_leaves_room_for_a_model_call(self) -> "Settings":
        # Jev time counts against the request budget; after a Jev call that uses its whole timeout, the
        # executor must still be able to start one model call (S2.9). Field names only, never values.
        if self.jev_timeout_seconds + MIN_CALL_SECONDS + CALL_MARGIN_SECONDS >= self.request_budget_seconds:
            raise ValueError(
                "jev_timeout_seconds + MIN_CALL_SECONDS + CALL_MARGIN_SECONDS must be less than "
                "request_budget_seconds, so a model call can still start after a Jev call times out"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
