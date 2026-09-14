# k8s/ — Kubernetes Manifests

## Status: written and syntax-validated only. NOT deployed.

These manifests (`deployment.yaml`, `service.yaml`) have been **written and validated for syntax only** — parsed as well-formed YAML, and checked with `kubectl apply --dry-run=client` against the locally available `kubectl` client. They have **not** been applied to, or run on, any live Kubernetes cluster, GPU-enabled or otherwise. No pod described here has ever actually started, and no `/health`, `/generate`, or `/v1/chat/completions` request has ever been served through this Service.

This is deliberate, not an oversight: there is no GPU-enabled Kubernetes cluster available in this project's current environment. If this section is ever rewritten to describe a real deployment, it must say explicitly whether that was:
- applied to a live cluster with `kubectl apply` and actually reached a `Running`/`Ready` pod, versus
- only `--dry-run`'d or schema-linted, as is the case today.

These are materially different claims (see also [README.md Section 10](../README.md#10-kubernetes-deployment)) and must not be conflated.

## What's here

- **`deployment.yaml`** — a single-replica `Deployment` for `src/server/main.py`, referencing a placeholder image (`llm-inference-gpu-serving-platform:latest`) that assumes a not-yet-written `docker/Dockerfile`. Requests/limits `nvidia.com/gpu: 1` (requires a cluster with the NVIDIA device plugin on its GPU nodes) plus CPU/memory requests and limits. Readiness/liveness probes hit the `/health` endpoint added to `src/server/main.py` for this purpose; `initialDelaySeconds` is set from real model-load times observed during local dev server runs this session (6.4s–10.9s across 4 startups), not guessed.
- **`service.yaml`** — a `ClusterIP` Service exposing the Deployment's port 8000.

## Known gaps before this could actually be applied

- `docker/` has no `Dockerfile` yet — the image this Deployment references does not exist and cannot be pulled.
- No GPU-enabled cluster has been targeted, so the `nvidia.com/gpu` resource request and the NVIDIA device plugin dependency are unverified in practice — only understood from the Kubernetes GPU scheduling docs.
- No namespace, `ImagePullSecrets`, `ConfigMap`/`Secret` for `.env`-style config, or `HorizontalPodAutoscaler` are included — autoscaling is explicitly out of scope for this project phase (see [README.md Section 13](../README.md#13-future-work)).
