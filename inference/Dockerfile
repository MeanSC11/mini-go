# Build from the repository root:  docker build -f inference/Dockerfile .
# Includes the training package (CPU torch) so AlphaZero checkpoints can be served.
FROM python:3.11-slim

WORKDIR /srv

COPY engine /srv/engine
RUN pip install --no-cache-dir /srv/engine

COPY training /srv/training
RUN pip install --no-cache-dir /srv/training \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY inference /srv/inference
RUN pip install --no-cache-dir /srv/inference

WORKDIR /srv/inference
ENV GOBOT_INF_CHECKPOINT_DIR=/checkpoints
EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
