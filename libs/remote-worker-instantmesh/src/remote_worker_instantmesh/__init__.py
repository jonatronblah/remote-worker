from remote_worker import Worker
from remote_worker.config import settings

w = Worker(worker_settings=settings)


def main():
    print(w.settings.REDIS_HOST)


if "__name__" == "__main__":
    main()
