# syntax=docker/dockerfile:1.7
#
# Traffic signal policy service.
#
# The image holds code and dependencies only — no model weights. Models arrive
# at runtime, either mounted from disk or resolved from the MLflow registry, so
# one image can serve any model version without a rebuild.

# ---------------------------------------------------------------------------
# Stage 1: install dependencies into an isolated virtualenv
# ---------------------------------------------------------------------------
# Pinned to an exact patch version and distro. `python:latest` would silently
# change underneath us between builds.
FROM python:3.11.16-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# Requirements are copied BEFORE the application code. Docker caches each layer,
# so editing app code does not invalidate the slow dependency install below.
COPY requirements.txt .

# CPU-only torch first. The default Linux wheel bundles CUDA (~2 GB) that this
# CPU inference service never uses. Once torch 2.13.0+cpu is installed, the
# `torch==2.13.0` pin in requirements.txt is already satisfied (PEP 440 ignores
# the +cpu local label), so pip does not replace it with the CUDA build.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 \
 && pip install -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: runtime image — only the finished virtualenv and the app code
# ---------------------------------------------------------------------------
FROM python:3.11.16-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Unprivileged user. The service only reads files and answers HTTP; it has no
# reason to run as root inside the container.
RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --create-home --home-dir /home/app app

WORKDIR /srv

COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app app/ app/

USER app

EXPOSE 8000

# No HEALTHCHECK instruction on purpose: Kubernetes ignores it and uses its own
# liveness/readiness probes against /health and /ready.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
