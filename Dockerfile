FROM python:3.12-slim

WORKDIR /src

# Install worker as a package
COPY worker_olmocr worker_olmocr
COPY pyproject.toml ./
RUN pip install -U pip
RUN pip install . --no-cache-dir

# Setup unprivileged user
RUN adduser --gid=65534 --uid=2000 --home=/home --no-create-home --shell=/bin/nologin worker
USER worker
