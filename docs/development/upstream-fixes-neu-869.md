# NEU-869 Router implementation boundaries

PR base: `0cb19847f0a93a9a6e3604fac28bac3b6c3e37ce` of the Neutree fork.
Implementation started at `cd2158edbfdacb9dbae35b5ea1c7e362e282abd1`; the intervening
base commits only update the Helm chart version.

## Stage 1: selected upstream fixes

| Source | Fixed revision | Adaptation |
| --- | --- | --- |
| [#768](https://github.com/vllm-project/production-stack/pull/768) | `eb9526f20eaac96f56696174c6568afe57d08a5a` | Publish the application loop and initialize discovery fields before the watcher starts |
| [#1040](https://github.com/vllm-project/production-stack/pull/1040) | `a4e6c0a21047de70671ece33f88753e27f79fa2d` | Cancel backend header waits on disconnect; close suspended response generators on send failure |
| [#1168](https://github.com/vllm-project/production-stack/pull/1168) | `8edddd14b1aa0055d07d52c6591c2ff1f947ca1a` | Initialize CPU sampling and collect without blocking the request loop |

Retain Neutree's request-stat cleanup, tenant filtering, passthrough admission
and per-host connection limits. Do not include #1169 header deduplication or
`#1110` latency-window expiry. The #1040 revision includes unrelated merged
content, so adapt the cancellation changes rather than cherry-picking it whole.

## Stage 2: reliability, native configuration and hooks

Adapt [#1092](https://github.com/vllm-project/production-stack/pull/1092) at
`7adf99af02de37b3c62b32cba80db616491c2527` inside both existing Kubernetes
discovery classes. Complete the List before dispatching existing resource
events, preserve callbacks, and remove confirmed missing resources. Retry
after five seconds and keep cached backends usable during failures. Pass List
resourceVersion through to Watch without adding a version-comparison layer.

Configure the existing `ClientSession`/`TCPConnector` using one native
`ClientTimeout(total=None, connect=..., sock_connect=..., sock_read=...)`.
P/D shares that session and captures selected endpoint metadata per request.
Discovery changes do not close an in-flight backend connection.

Only add `router:connection_pool_waiting`, through public native queued/start/end
and request-exception signals. Do not add body/finally metric probes, approximate
socket occupancy, full-body terminal counters or another HTTP wrapper. The
configuration defaults and metric limitations are documented in the Router README.

## Stage 3: upstream audio implementation and thin API routes

Use the audio implementation present at fixed upstream
`b4e79de329d15ea291b46928b28517c1c919db51`, including its transcription/raw
multipart entry points and `proxy_multipart_request` response handling.
Streaming and non-JSON responses already exist in its PR base
`7b1164dbe71b1f02ede85152bdfc2f383376dc2d`;
[#1156](https://github.com/vllm-project/production-stack/pull/1156) adds the
remaining text-field and repeated-key forwarding rules.

Adapt the complete relevant audio chain to local scope filtering, existing
headers, native timeouts, request-stat cleanup and cancellation helpers. Close
upload forms and backend responses on normal and interrupted paths. Retain the
existing transcription language default; translation preserves the original
multipart form. Do not import unrelated orchestrated P/D, provider or metric
features from the newer upstream file.

Adapt the Pooling route from [#1112](https://github.com/vllm-project/production-stack/pull/1112)
at `4955696dcd4acff35b7a7ccbea183784371a0eaf` to the scoped prefix. Classification
is a local thin route. Add scoped translations and transcriptions while keeping
the existing transcription entry. No Responses, speech, engine task schemas,
control-plane changes or UI changes are included.

## Verification scope

New tests cover real normal or failure scenarios absent from the existing suite:
startup/discovery recovery, cancellation/resource ownership, native timeout and
configuration behavior, connection-pool waiting, and the scoped API protocols.
Do not add migration assertions about removed private fields or tests that only
mirror internal implementation. Local HTTP/ASGI tests do not establish real
Kubernetes, gateway or engine compatibility.

## Local validation

Implemented on `codex/neu-869-router-stages` from the base above.

- Full Router suite: **144 passed**, including a real CLI subprocess using the
  Kubernetes SDK against a local API stub and real local HTTP/ASGI scenarios.
- Baseline suite before implementation: 72 passed. Added cases cover missing
  normal paths and production boundaries, including CLI/file hot reload,
  discovery startup/recovery, sleep/wake, cancellation, scoped input APIs,
  multipart fields/streaming and P/D response compatibility.
- Executed changed production lines: **528/628 (84.1%)**, measured against the
  base diff, including the new collector. This is changed-line coverage, not
  whole-repository coverage or a real-cluster acceptance result.
- All applicable pre-commit hooks and `ruff check src` passed.
- Production Python diff: **+1131/-591, net +540**. This includes adopted upstream
  audio code and moved event handlers; tests and documentation are separate.

No image build, deployment, real Kubernetes/engine/gateway E2E, database migration
or UI implementation was performed.

The current product does not support LoRA or P/D. Their compatibility findings
from local review remain outside this delivery by the agreed scope.
