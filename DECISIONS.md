# Decision log

Each entry: what was chosen, what the alternative was, and what it costs.

---

### 1. Separate `/health` and `/ready`

**Chosen:** two endpoints with different semantics. `/health` returns 200 as soon
as the process serves HTTP. `/ready` returns 503 until model weights are loaded.

**Alternative:** a single `/health` that checks everything.

**Why:** the two probes drive different kubelet actions. A failed liveness probe
restarts the container; a failed readiness probe only removes the pod from
Service endpoints. Loading model weights takes real time, so a combined endpoint
under an aggressive liveness probe produces a permanent restart loop — the pod
never survives long enough to finish loading.

**Cost:** slightly more surface to maintain, and probe timings must be configured
correctly in the Deployment or the separation buys nothing.

---

### 2. Load the model at startup, not lazily on first request

**Chosen:** load during application lifespan startup.

**Alternative:** load on first request.

**Why:** lazy loading means the first real user request absorbs the full cold-start
latency, and the pod reports itself ready while it cannot actually serve. Loading
at startup makes readiness honest.

**Cost:** slower pod startup, which must be accounted for in
`initialDelaySeconds` and in rolling-update surge settings.

---

### 3. Inference failures degrade to a fallback phase rather than returning 500

**Chosen:** catch inference errors, return a safe default phase, and label the
response `action_source: "fallback"`.

**Alternative:** propagate the error as a 500.

**Why:** the consumer is a signal controller. Losing the optimiser should mean
falling back to a valid fixed plan, not losing signal control entirely. Labelling
the source keeps this visible instead of silent — the fallback rate is a metric,
not a hidden behaviour.

**Cost:** a degraded model can serve fallbacks indefinitely without an outage
being obvious. This *must* be alerted on, or it hides a real failure.

---

### 4. Explicit named schema instead of a raw observation vector

**Chosen:** `TrafficState` with named, range-validated fields.

**Alternative:** accept `{"obs": [...]}` and pass it straight to the model.

**Why:** a malformed request fails at the edge with a 422 rather than producing a
plausible-looking but wrong control action. Field ranges (`current_phase` 0–15,
non-negative queues) catch integration errors at the boundary.

**Cost:** the schema is coupled to the observation layout. `build_observation()`
must match training-time ordering, and changing it is a breaking model-contract
change.

---

### 5. Model source isolated in `model.py`

**Chosen:** all knowledge of *where a model comes from* lives in one module
behind a `Policy` protocol.

**Alternative:** load the model inside the route handler.

**Why:** switching from local checkpoint to MLflow registry is then a one-file
change with no API impact, and tests can inject a stub policy instead of loading
real weights.

**Cost:** one extra layer of indirection.

---

### 6. Configuration entirely from environment variables

**Chosen:** `pydantic-settings` with a `TSC_` prefix, no environment-specific
values in code.

**Why:** this is what makes the build-once-promote-many pipeline valid. The image
is byte-identical across staging and production; only the environment differs.
Rebuilding per environment means the artifact you tested is not the artifact you
shipped.

**Cost:** more environment configuration to manage, which raises the importance of
getting secrets handling right.

---

### 7. Tests inject a stub policy rather than loading a real checkpoint

**Chosen:** `StubPolicy`, `BrokenPolicy`, `OutOfRangePolicy` injected via
`policy_service.set_policy()`.

**Why:** these are contract tests. Model *quality* is validated in the training
pipeline against held-out data; the API test suite should verify request
validation, status codes and degradation paths, and should run in seconds without
torch weights.

**Cost:** no test coverage of real checkpoint loading. That needs a separate
integration test against a real artifact before the pipeline can gate on it.

---

## Open questions

- Bake weights into the image (immutable, reproducible, larger image, rebuild to
  change model) or pull from the registry at startup (swap models without
  rebuilding, but adds a startup network dependency)? Leaning toward registry
  pull with the image as fallback — to be decided in the Docker phase.
- Drift detection currently only exposes input distributions as histograms.
  Comparison against the training baseline is not yet automated.
