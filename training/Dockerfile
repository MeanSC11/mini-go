# Build from the repository root:  docker build -f training/Dockerfile .
# Default is CPU torch; for GPU training use a CUDA base image, e.g.
#   FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
FROM python:3.11-slim

WORKDIR /srv

COPY engine /srv/engine
RUN pip install --no-cache-dir /srv/engine

COPY training /srv/training
RUN pip install --no-cache-dir /srv/training \
    --extra-index-url https://download.pytorch.org/whl/cpu

WORKDIR /srv/training
CMD ["azero-train", "--config", "config.yaml"]
