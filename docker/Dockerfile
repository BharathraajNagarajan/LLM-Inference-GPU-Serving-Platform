# docker/Dockerfile
#
# Containerizes the naive FastAPI baseline server (src/server/main.py).
# Not vLLM. Placed in docker/ (not repo root) to match the convention
# already assumed by k8s/deployment.yaml and k8s/README.md, which
# reference this exact path. Build context is still the repo root
# (needed for pyproject.toml and src/), so build from there:
#
#   docker build -t llm-inference-gpu-serving-platform:latest -f docker/Dockerfile .
#
# STATUS: written, NOT yet built or run. `docker build` has deliberately
# not been executed for this file (see README.md's Docker note) — it
# still needs an actual build + run to be verified, the same way the
# CUDA torch install itself was only trusted locally after being
# empirically verified (see README.md Section 5), not assumed from specs.

# python:3.10-slim, NOT a CUDA base image (e.g. nvidia/cuda:*-devel).
# This is deliberate and is a common point of confusion: the container
# does not need CUDA baked into its base OS/userspace, because GPU
# access at runtime is provided by the *host*, either via
# nvidia-container-toolkit (`docker run --gpus ...`) or the Kubernetes
# NVIDIA device plugin (k8s/deployment.yaml's `nvidia.com/gpu: 1`) —
# both inject the host's GPU driver + runtime libraries into the
# container at run time. What actually needs to be CUDA-aware here is
# the Python torch *wheel* installed below (a package concern), not the
# base image.
FROM python:3.10-slim

# requires-python in pyproject.toml is ">=3.10,<3.11" — this base image
# tag satisfies that exactly, matching the same Python 3.10 requirement
# documented in README.md Section 4 (the system default is 3.14, which
# does not work for this project's torch/vllm stack).

# ca-certificates: needed for HTTPS to the Hugging Face Hub, which
# src/server/main.py's lifespan hook calls at container startup to
# download model weights (weights are NOT baked into this image — first
# startup needs network access, or a mounted/pre-warmed HF cache volume).
# build-essential: kept as a safety net in case any dependency lacks a
# prebuilt wheel for this platform/Python combo and falls back to a
# source build. This has not been build-tested yet, so this is a
# precaution, not a confirmed requirement — it can likely be dropped
# once an actual build confirms every dependency resolves to a wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Only the necessary source: pyproject.toml (dependencies + package
# metadata) and src/ (the actual server code). Deliberately not
# `COPY . .` — venv/, results/, .git/, benchmark/, metrics/, tests/,
# k8s/, docker/, docs/ etc. are excluded both by not being COPYed here
# and, redundantly, by .dockerignore at the repo root (Docker only
# honors a .dockerignore at the build context root, not next to a
# Dockerfile referenced via -f, hence it lives at the repo root here).
COPY pyproject.toml ./
COPY src/ ./src/

# Same two-step install order as the verified local venv setup in
# README.md Section 5: install the CUDA (cu118) torch wheel explicitly
# and FIRST, so it is already satisfied — and NOT silently replaced by a
# CPU-only build — before `pip install .` resolves the rest of
# pyproject.toml's dependencies. `pip install torch` alone was verified
# on this machine to silently resolve a CPU-only wheel; the same failure
# mode applies inside a fresh container image just as much as it did in
# the local venv, so it is avoided the same explicit way here.
#
# `pip install .` here installs only pyproject.toml's base
# [project.dependencies] (torch, transformers, httpx, pydantic, pynvml,
# fastapi, uvicorn — exactly what src/server/ actually imports or needs
# to run, checked via grep). vllm lives in the separate "vllm" optional
# extra (see pyproject.toml) and is deliberately NOT installed here: this
# image only runs the naive FastAPI server, which never imports vllm, so
# pulling it in would bloat this image and risk failing the build on
# vllm's own install requirements for no benefit to this container.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu118 \
    && pip install --no-cache-dir .

# Basic container security hygiene: run as a non-root user, not as root.
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.server.main:app", "--host", "0.0.0.0", "--port", "8000"]
