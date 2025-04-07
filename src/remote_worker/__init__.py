from rq import get_current_job, Queue
from redis import Redis
from zipfile import ZipFile
import subprocess
import io
import os
from uuid import uuid4

from remote_worker.config import settings


class MeshWorker:
    def __init__(self, worker_settings) -> None:
        self.id = uuid4().hex
        self.settings = worker_settings
        self.redis_conn = self._get_conn()
        self.dep_queue = self._get_queue()

    def _get_conn(self):
        return Redis(host=self.settings.REDIS_HOST, port=6379)

    def _get_queue(self):
        return Queue(name=self.settings.DEP_QUEUE, connection=self.redis_conn)

    def generate(self):
        path = f"tmp/{self.id}.jpg"

        img_job = get_current_job(self.worker_settings.redis_conn)
        img_job_id = img_job.dependency.id
        img_job_res = self.dep_queue.fetch_job(img_job_id).result
        value = img_job_res
        value.save(path, format="JPEG")

        result = subprocess.run(
            [
                "python3",
                "run.py",
                "configs/instant-mesh-large.yaml",
                path,
                "--export_texmap",
            ],
            capture_output=True,
            text=True,
        )
        self.result = result

    def handle_results(self):
        id = self.id
        buffer = io.BytesIO()
        with ZipFile(buffer, "w") as zip:
            zip.write(f"outputs/{id}.obj")
            zip.write(f"outputs/{id}.mtl")
            zip.write(f"outputs/{id}.png")
        return buffer, id

    def cleanup(self):
        id = self.id
        os.remove(f"tmp/{id}.jpg")
        os.remove(f"outputs/{id}.obj")
        os.remove(f"outputs/{id}.mtl")
        os.remove(f"outputs/{id}.png")


def main():
    mw = MeshWorker(worker_settings=settings)
    mw.generate()
    buffer = mw.handle_results()
    mw.cleanup()
    return buffer