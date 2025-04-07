FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

LABEL name="instantmesh" maintainer="instantmesh"

FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slimf


# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    # apt-get install -y python3.10 python3-pip tzdata build-essential git wget vim libegl1-mesa-dev libglib2.0-0 unzip && \
    apt-get install -y tzdata build-essential git wget vim libegl1-mesa-dev libglib2.0-0 unzip && \
    ln -fs /usr/share/zoneinfo/America/Chicago /etc/localtime && \ 
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


WORKDIR /app

RUN uv venv .venv
RUN . .venv/bin/activate
RUN uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 xformers==0.0.29.post1 --index-url https://download.pytorch.org/whl/cu124
RUN uv pip install triton ninja


# ADD . /app



RUN uv pip install .[instantmesh]

RUN git clone https://github.com/MrLemur/InstantMesh

RUN uv pip install -r InstantMesh/requirements.txt

ENV CONFIG="instant-mesh-large"

ENV PATH="/app/.venv/bin:$PATH"

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []

# Run the FastAPI application by default
# Uses `fastapi dev` to enable hot-reloading when the `watch` sync occurs
# Uses `--host 0.0.0.0` to allow access from outside the container