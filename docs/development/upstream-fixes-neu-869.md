# NEU-869 Router implementation boundaries

PR base: `0cb19847f0a93a9a6e3604fac28bac3b6c3e37ce` of the Neutree fork.
Only changes already merged into upstream `main` are eligible for adoption.
Checked on 2026-11-10 against upstream main
`405a2fa79992754d31f754c31b1eabb7183a540e`. A PR head SHA or a non-null
GitHub `merge_commit_sha` alone is not evidence that a PR was merged.

## Confirmed upstream sources

| Source | Adopted revision | Mainline evidence / adaptation |
| --- | --- | --- |
| [#768](https://github.com/vllm-project/production-stack/pull/768) | `eb9526f20eaac96f56696174c6568afe57d08a5a` | Merged 2025-12-17; commit is an ancestor of checked main. Publish the startup loop and initialize Pod-IP watcher fields before starting the thread. |
| [#1112](https://github.com/vllm-project/production-stack/pull/1112) | `4955696dcd4acff35b7a7ccbea183784371a0eaf` | Merged 2026-10-23; commit is an ancestor of checked main. Adapt pooling to Neutree's scoped path. |
| [Merged audio implementation](https://github.com/vllm-project/production-stack/blob/7b1164dbe71b1f02ede85152bdfc2f383376dc2d/src/vllm_router/services/request_service/request.py) | `7b1164dbe71b1f02ede85152bdfc2f383376dc2d` | Snapshot is an ancestor of checked main (8 commits ahead, 0 behind). Adopt its shared multipart proxy, JSON/text/streaming responses and normal response cleanup. Exclude unrelated P/D/provider changes. |

## Withdrawn unmerged changes

All four PRs below were still open with `merged=false` and no `merged_at` on
2026-11-10. Their implementations were removed rather than relabelled as local fixes.

| Source | Removed behavior |
| --- | --- |
| [#1040](https://github.com/vllm-project/production-stack/pull/1040) | Disconnect/header races, active upstream cancellation, `_ProxyStreamingResponse` and ASGI send-failure cleanup derived from that work. JSON/passthrough use native StreamingResponse and ordinary header waits again. |
| [#1092](https://github.com/vllm-project/production-stack/pull/1092) | Complete relists, pagination/reconciliation, fixed five-second retries, stop events and cache-preservation changes. Discovery returns to the fork's baseline plus merged #768. |
| [#1168](https://github.com/vllm-project/production-stack/pull/1168) | CPU priming and nonblocking samples. `/metrics` returns to the fork's `cpu_percent(interval=0.1)`. |
| [#1156](https://github.com/vllm-project/production-stack/pull/1156) | Forwarding unknown/additional and repeated text fields when rebuilding transcription forms. Raw translation forwarding remains part of the merged audio implementation. |

## Neutree-specific features

These are explicit product changes, not adopted upstream fixes:

- Native `ClientSession`/`TCPConnector` configuration with one
  `ClientTimeout(total=None, connect=..., sock_connect=..., sock_read=...)`.
  Per-request legacy timeouts do not override these settings.
- `router:connection_pool_waiting`, maintained by native aiohttp queued hooks.
  Queued-start increments; queued-end or request-exception decrements an active
  wait once. `/metrics` exports values and prunes departed-backend series only
  after five idle minutes with no queued calls.
- Inference-replica labels on backend metrics, preserving the existing incoming
  request counter dimensions and original request-stat accounting.
- Scoped pooling, classification, transcription and translation paths.
  Classification is a local thin route. Input-only APIs have no text prefix;
  the local PrefixAware adapter reuses the existing low-QPS fallback.

The existing transcription handler is extended to expose only
`/{workspace}/{endpoint}/v1/audio/transcriptions`. No unscoped transcription
alias is retained. Required workspace/endpoint values come from path arguments.

`route_general_transcriptions` parses and rebuilds the merged implementation's
known transcription fields, retaining Neutree's existing `language=en` default.
`route_multipart_request` forwards the original multipart bytes for translation
without injecting that default. As in merged upstream, only transcription passes
its `stream` option to the proxy; translation uses the buffered response path.
Workspace/endpoint only filter backend candidates and select the Endpoint's routing
policy. Multipart parsing and response handling are shared across scopes.

Other retained fork/product differences from that snapshot are the existing
`language=en` default, input validation and form cleanup, native session timeout
settings instead of the legacy per-request 300-second deadline, queue trace labels,
and the fork's success-aware request statistics. Transport timeouts return 504;
connection errors return 503. These changes apply equally to every scope.

The fork's existing request-stat `finally` cleanup is retained; the withdrawn
active-disconnect and send-failure fixes are not reimplemented under another name.
There are no new body hooks, socket occupancy approximations or full-body counters.

## Verification and compatibility boundary

Unit/integration tests cover the retained configuration, scoped protocols,
normal resource cleanup, native timeout and pool-wait behavior. Tests requiring
withdrawn fixes and duplicate parameter combinations are removed. The CLI smoke
subprocess test is removed; startup/Watch integration belongs to deployment E2E,
not this unit suite. No P/D adapter is included in the new multipart paths, and
no new P/D routing bypass is retained. The new client fields
are startup snapshots excluded from dynamic equality checks; the existing watcher
is unchanged and does not apply those fields at runtime.

The old unscoped transcription URL is intentionally replaced by the scoped
Neutree path. Unknown transcription fields are not guaranteed to be forwarded.
The former relist recovery, cache-during-outage, nonblocking CPU and active
client-disconnect guarantees are no longer part of this change.

Historical image/E2E results obtained before this revision describe the earlier
implementation. They do not establish those withdrawn guarantees for this source.
Native file configuration/reload, sleep/wake, LoRA and P/D remain outside product
support. No engine task schema or new global model-routing entry is added.
