FROM python:3.11-slim
WORKDIR /src

# HuggingFace cache location - COHÉRENT
ENV HF_HOME=/models/huggingface
ENV HF_HUB_CACHE=/models/huggingface
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV XET_LOG_DIR=/models/hf/xet/logs

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
  git \
  libglib2.0-0 \
  libsm6 \
  libxrender1 \
  libxext6 \
  poppler-utils \
  fonts-crosextra-caladea \
  fonts-crosextra-carlito \
  gsfonts \
  lcdf-typetools \
  && rm -rf /var/lib/apt/lists/*

# Install worker as a package
COPY pyproject.toml ./
RUN pip install -U pip
RUN pip install --no-cache-dir $(python -c "import tomllib; f=open('pyproject.toml','rb'); deps=tomllib.load(f)['project']['dependencies']; print(' '.join(deps))")

COPY worker_olmocr worker_olmocr
RUN pip install . --no-cache-dir --no-deps
