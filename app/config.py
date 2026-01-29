from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://postgres:postgres@localhost:5432/hackathon"
    debug: bool = True
    anthropic_api_key: str = ""

    class Config:
        env_file = ".env"


settings = Settings()
