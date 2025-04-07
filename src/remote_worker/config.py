import os
from functools import lru_cache


class BaseConfig:
    REDIS_HOST: str = os.environ.get("REDIS_HOST", "redis")
    QUEUE: str = os.environ.get("QUEUE", "queue1")


@lru_cache()
def get_settings():
    config_cls_dict = {
        "development": BaseConfig,
    }

    config_name = os.environ.get("WORKER_CONFIG", "development")
    config_cls = config_cls_dict[config_name]
    return config_cls()


settings = get_settings()
