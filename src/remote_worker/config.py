from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    redis_host: str
    dep_queue: str
