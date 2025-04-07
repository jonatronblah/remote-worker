from remote_worker import Worker
from remote_worker.config import settings


class MeshWorker(Worker):
    def __init__(self, worker_settings) -> None:
        super().__init__(worker_settings)
        self.worker_settings = worker_settings

    def generate(self):
        print(self.worker_settings.REDIS_HOST)


def main():
    mw = MeshWorker(worker_settings=settings)
    return mw


if "__name__" == "__main__":
    main()
