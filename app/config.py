from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str
    azure_ai_project_endpoint: str
    azure_openai_endpoint: str
    azure_ai_services_endpoint: str
    openai_model: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()