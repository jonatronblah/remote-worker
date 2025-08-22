from fastapi import FastAPI, Query
from typing import Annotated
from pydantic import BaseModel
import io
from fastapi.responses import FileResponse
import os
from magenta_rt import audio, system
import uuid

os.environ["HF_TOKEN"] = " "

app = FastAPI()


class Parameters(BaseModel):
    style: str = "bad drummer"
    length: int = 10


@app.get("/", response_model=None)
def predict(parameters: Annotated[Parameters, Query()]):
    mrt = system.MagentaRT(tag="large", device="gpu", skip_cache=False, lazy=False)
    style = mrt.embed_style(parameters.style)
    chunks = []
    state = None
    for i in range(round(parameters.length / mrt.config.chunk_length)):
        chunk, state = mrt.generate_chunk(state=state, style=style)
        chunks.append(chunk)
    generated = audio.concatenate(chunks, crossfade_time=0.001)
    id = uuid.uuid4().hex
    filename = f"{id}.wav"
    generated.write(filename)
    return FileResponse(
        filename,
        media_type="audio/wav",
        filename=filename,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
