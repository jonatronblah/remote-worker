FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS base

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git build-essential \
    gcc wget \
    git \
    ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev \
    && mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

FROM base AS req

# install python with uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
RUN uv python install 3.12.10

# Install the project into `/app`
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

COPY src pyproject.toml uv.lock README.md /app/

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    # --mount=type=bind,source=uv.lock,target=uv.lock \
    # --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --no-dev

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
# ADD ./src /app/src
# RUN --mount=type=cache,target=/root/.cache/uv \
#     uv sync --no-dev

# RUN uv pip install autogluon
ENV PATH="$PATH:/opt/gtk/bin"
RUN uv pip install flash-attn --no-build-isolation
# RUN uv pip install "git+https://github.com/DataDog/toto"


RUN uv pip install jupyter

# Place executables in the environment at the front of the path
ENV PATH="/usr/local/cuda/bin:$PATH"



# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []

# Run the FastAPI application by default
# Uses `fastapi dev` to enable hot-reloading when the `watch` sync occurs
# Uses `--host 0.0.0.0` to allow access from outside the container


# RUN uv pip install "git+https://github.com/DataDog/toto"