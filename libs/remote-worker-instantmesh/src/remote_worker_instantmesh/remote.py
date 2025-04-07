from uuid import uuid4
import subprocess
from rq import get_current_job, Queue
from redis import Redis
from zipfile import ZipFile
import os
import io


redis_conn = Redis(host="192.168.1.117", port=6379)
q = Queue(name="queue1", connection=redis_conn)


def mesh():
    r = uuid4().hex
    path = f"tmp/{r}.jpg"

    img_job = get_current_job(redis_conn)
    img_job_id = img_job.dependency.id
    img_job_res = q.fetch_job(img_job_id).result
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
    return result
    # zip_buffer = handle_output(r)
    # cleanup_output(r)
    # return zip_buffer


def handle_output(id):
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as zip:
        zip.write(f"outputs/{id}.obj")
        zip.write(f"outputs/{id}.mtl")
        zip.write(f"outputs/{id}.png")
    return buffer


def cleanup_output(id):
    os.remove(f"tmp/{id}.jpg")
    os.remove(f"outputs/{id}.obj")
    os.remove(f"outputs/{id}.mtl")
    os.remove(f"outputs/{id}.png")
