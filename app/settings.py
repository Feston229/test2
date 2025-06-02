from typing import Literal

from pydantic import ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    env: Literal["local", "stage", "prod"]
    model: str = "qwen3:8b"
    csv_path: str = "data/data.csv"

    @field_validator("env")
    def validate_env(cls, v: str) -> str:
        if v not in ["local", "stage", "prod"]:
            raise ValueError("Environment must be local, stage, or prod")
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",  # no prefix - variables match exactly
        env_file_encoding="utf-8",
        extra="ignore",  # ignore extra fields in .env
    )


# Instantiate settings (will raise ValidationError if config is invalid)
try:
    settings = Settings()
except ValidationError as e:
    print(f"Configuration error: {e}")
    raise
