# Build from the repository root:  docker build -f server/Dockerfile .
FROM python:3.11-slim

WORKDIR /srv

COPY engine /srv/engine
RUN pip install --no-cache-dir /srv/engine

COPY server /srv/server
RUN pip install --no-cache-dir /srv/server

WORKDIR /srv/server
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
