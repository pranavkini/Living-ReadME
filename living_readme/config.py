from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    gemini_api_key: str
    gemini_model: str = "gemini-1.5-flash"  # default, but overridable in .env

    class Config:
        env_prefix = ""
        extra = "ignore"  # ignore unknown keys instead of raising errors

def load_settings() -> Settings:
    return Settings()
