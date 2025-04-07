from remote_worker import Worker
from remote_worker.config import settings


class MeshWorker(Worker):
    def __init__(self, worker_settings) -> None:
        super().__init__(worker_settings)


def main():
    mw = MeshWorker(worker_settings=settings)
    print(mw.settings.REDIS_HOST)


if "__name__" == "__main__":
    main()
