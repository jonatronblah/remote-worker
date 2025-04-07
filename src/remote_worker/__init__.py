from rq import get_current_job, Queue
from redis import Redis


class Worker:
    def __init__(self, worker_settings) -> None:
        self.settings = worker_settings
        self.redis_conn = self._get_conn()
        self.queue = self._get_queue()

    def _get_conn(self):
        return Redis(host=self.settings.REDIS_HOST, port=6379)

    def _get_queue(self):
        return Queue(name=self.settings.QUEUE, connection=self.redis_conn)
