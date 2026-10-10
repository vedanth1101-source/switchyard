"""Environment settings (pydantic-settings).

Values come from process environment variables only (Vercel project settings, CI, or an
exported shell). The .env file is deliberately not read here.
"""

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    database_url: str | None = None  # DATABASE_URL, e.g. Neon pooled postgresql+psycopg://...
    redis_url: str | None = None  # REDIS_URL, e.g. Upstash rediss://...

    @field_validator("database_url", "redis_url", mode="before")
    @classmethod
    def blank_means_unset(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
