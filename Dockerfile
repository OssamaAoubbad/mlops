FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_CACHE_DIR=/root/.cache/pip \
    GITHUB_USERNAME=container

# 1. Installer dépendances système (stable layer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 2. Installer pip tooling (stable layer)
RUN pip install --upgrade "pip<27" "setuptools==68.2.2" wheel

# 3. Copier requirements (cache layer IMPORTANT)
COPY requirement-ci.txt .

# 4. Installer dépendances Python (CACHE LAYER CRUCIAL)
RUN pip install --no-cache-dir -r requirement-ci.txt

# 5. Copier code (dernier layer = change souvent)
COPY . .

EXPOSE 8000 8265 5000