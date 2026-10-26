# NEU-869: isolated upstream fixes

This change adapts six approved upstream fixes to the current Neutree Router.
The source revisions below are fixed review inputs. Open upstream PRs are not
represented as merged releases, and these changes are adapted ports rather
than literal cherry-picks of the upstream commits.

| Upstream PR | Source revision | Current code and adaptation |
| --- | --- | --- |
| [#768](https://github.com/vllm-project/production-stack/pull/768) | `eb9526f20eaac96f56696174c6568afe57d08a5a` (merged commit) | `app.py:lifespan` publishes the event loop before session initialization. `service_discovery.py:K8sPodIPServiceDiscovery` initializes fields before starting the watcher and defers session creation until the loop is ready. Scheduling failures close the coroutine; future failures are logged without blocking the watcher. An async lock serializes session replacement across startup and discovery bursts. |
| [#1040](https://github.com/vllm-project/production-stack/pull/1040) | `a4e6c0a21047de70671ece33f88753e27f79fa2d` (PR head) | `request.py:_anext_or_disconnect` races backend headers against client disconnect, preserving independent caller cancellation. `_ProxyStreamingResponse` closes suspended upstream generators on ASGI send failure. Existing NEU-555 terminal accounting is retained. |
| [#1169](https://github.com/vllm-project/production-stack/pull/1169) | `bb877c277bb80b11ebb05ce4f8dfd662161ee7d6` (PR head) | `request.py:_response_headers` removes backend Date/Request-ID duplicates and hop-by-hop headers. JSON, multipart and the existing passthrough path share it. Content type is retained; unsupported content encodings are preserved. |
| [#1156](https://github.com/vllm-project/production-stack/pull/1156) | `b4e79de329d15ea291b46928b28517c1c919db51` (PR head) | `request.py:route_general_transcriptions` forwards extra text fields and repeated keys instead of silently dropping them. Its multipart response path owns uploads and the backend response until completion/cancellation and relays status, content type and error bodies. |
| [#1168](https://github.com/vllm-project/production-stack/pull/1168) | `8edddd14b1aa0055d07d52c6591c2ff1f947ca1a` (PR head) | `metrics_router.py:metrics` uses nonblocking CPU sampling; startup primes the measurement window. CPU is still host-wide, averaged since the previous sample, rather than a 100 ms blocking snapshot. Existing Neutree metric names and labels remain. |
| [#1110](https://github.com/vllm-project/production-stack/pull/1110) | `27f771226b08ea86c810a6d2f17f7675c055edc9` (PR head) | `request_stats.py:get_request_stats` expires latency samples before reading their average, matching the existing QPS/TTFT sliding-window behavior when a backend becomes idle. |

Neutree already includes NEU-573 per-backend pool isolation and NEU-555 request
statistics cleanup. Those fixes are retained, not re-imported from upstream.
Workspace/endpoint routing and passthrough admission are unchanged. The existing
JSON/passthrough no-total-timeout and multipart 300-second timeout are retained;
P/D and sleep/wakeup request functions remain unchanged.

Transport configuration, forwarding metric additions, full List/Watch recovery
(#1092), new APIs (including Pooling #1112), P/D authentication and Cluster/UI
changes are outside this isolated change. They stay on the separate NEU-869
feature branch and are not prerequisites for these fixes.

Regression tests cover startup ordering, pre-header disconnect, cancellation
races, backend/header/upload ownership, multipart repeated fields, wire-level
status/content-type preservation, nonblocking metrics and idle latency expiry.
Real Kubernetes and engine deployment validation is still required separately.
