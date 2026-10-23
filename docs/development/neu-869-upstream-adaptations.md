# NEU-869 upstream adaptations

The Neutree fork retains per-backend connection isolation, NEU-555 request-state
cleanup, workspace/endpoint routing and explicit passthrough workloads. Changes
below adapt the approved sources rather than replace these fork integrations.

| Source | Fixed source commit | Adaptation |
| --- | --- | --- |
| [#768](https://github.com/vllm-project/production-stack/pull/768) | `eb9526f20eaac96f56696174c6568afe57d08a5a` | Start discovery after the application event loop and discovery fields are ready. |
| [#1040](https://github.com/vllm-project/production-stack/pull/1040) | `a4e6c0a21047de70671ece33f88753e27f79fa2d` | Cancel the pending backend request on client disconnect; preserve independent cancellation and NEU-555 terminal cleanup. |
| [#1168](https://github.com/vllm-project/production-stack/pull/1168) | `8edddd14b1aa0055d07d52c6591c2ff1f947ca1a` | Sample CPU without blocking the forwarding event loop. |
| [#1169](https://github.com/vllm-project/production-stack/pull/1169) | `bb877c277bb80b11ebb05ce4f8dfd662161ee7d6` | Filter Router-owned and hop-by-hop response headers in JSON, multipart and passthrough paths. |
| [#1110](https://github.com/vllm-project/production-stack/pull/1110) | `27f771226b08ea86c810a6d2f17f7675c055edc9` | Expire idle-backend latency samples when reading the statistics window. |
| [#1156](https://github.com/vllm-project/production-stack/pull/1156) | `b4e79de329d15ea291b46928b28517c1c919db51` | Preserve extra/repeated multipart fields and stream status, content type and error bodies. |
| [#1092](https://github.com/vllm-project/production-stack/pull/1092) | `7adf99af02de37b3c62b32cba80db616491c2527` | Reconcile complete List snapshots before Watch, including identities, fork labels, callbacks and bounded recovery. |
| [#1112](https://github.com/vllm-project/production-stack/pull/1112) | `4955696dcd4acff35b7a7ccbea183784371a0eaf` | Pooling route with the fork's workspace/endpoint prefix. |

Responses API (#691) is deferred. P/D authentication (#1153) is outside this
change. #1001 duplicates NEU-555; #1082 conflicts with Cluster stage-timeout
configuration; #1127 changes an orchestrated P/D path absent from this fork;
#1170 duplicates the engine metrics collection path managed by Neutree. These
patches are not imported.

Baseline: 72 existing Router tests passed before adaptations. CPU sampling and
latency-window regressions were reproduced before their fixes. Per-change and
combined validation results are recorded when each stage is complete; upstream
PR test reports are not treated as tests of this fork.


Local adaptations also ensure P/D prefill consumes a non-streaming one-token
response while decode preserves the client stream setting, and reject new P/D
requests when discovery is stale. These are local request-lifecycle fixes;
P/D Authorization behavior remains unchanged as requested.
