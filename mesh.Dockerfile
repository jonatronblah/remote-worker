FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

LABEL name="instantmesh" maintainer="instantmesh"


ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y python3.10 python3-pip tzdata build-essential git wget vim libegl1-mesa-dev libglib2.0-0 unzip && \
    ln -fs /usr/share/zoneinfo/America/Chicago /etc/localtime && \ 
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


RUN python3.10 -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 xformers==0.0.29.post1 --index-url https://download.pytorch.org/whl/cu124
RUN python3.10 -m pip install triton ninja

RUN git clone https://github.com/MrLemur/InstantMesh /workspace/instantmesh
WORKDIR /workspace/instantmesh

RUN python3.10 -m pip install -r requirements.txt


ENV CONFIG="instant-mesh-large"


RUN python3.10 -m pip install "git+https://github.com/jonatronblah/remote-worker.git"


RUN mkdir /workspace/instantmesh/tmp 
RUN mkdir /workspace/instantmesh/outputs

# COPY run.py /workspace/instantmesh/run.py

ENTRYPOINT []