ARG BASE_IMAGE=python:3.13-slim
FROM ${BASE_IMAGE}

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_EXTRA_INDEX_URL=
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL} \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/venv/bin:$PATH

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN python3 -m venv --system-site-packages --without-pip /venv && python -m pip install .

COPY scripts ./scripts
