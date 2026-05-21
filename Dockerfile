FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GITHUB_USERNAME=container

COPY requirement-ci.txt .

RUN pip install --no-cache-dir --upgrade "pip<27" "setuptools==68.2.2" wheel && \
    pip install --no-cache-dir -r requirement-ci.txt && \
    pip install --no-cache-dir --force-reinstall "setuptools==68.2.2" "click==8.1.7"

COPY . .

EXPOSE 8000 8265 5000