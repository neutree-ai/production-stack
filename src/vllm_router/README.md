# Router

The source code for the request router.

## Key features

- Support routing to endpoints that run different models
- Exporting observability metrics for each serving engine instance, including QPS, time-to-first-token (TTFT), number of pending/running/finished requests, and uptime
- Support automatic service discovery and fault tolerance by Kubernetes API
- Model aliases
- Multiple different routing algorithms
  - Round-robin routing
  - Session-ID based routing
  - (WIP) prefix-aware routing

## Running the router

The router can be configured using command-line arguments. Below are the available options:

### Basic Options

- `--host`: The host to run the server on. Default is `0.0.0.0`.
- `--port`: The port to run the server on. Default is `8001`.

### Service Discovery Options

- `--service-discovery`: The service discovery type. Options are `static` or `k8s`. This option is required.
- `--static-backends`: The URLs of static serving engines, separated by commas (e.g., `http://localhost:8000,http://localhost:8001`).
- `--static-models`: The models running in the static serving engines, separated by commas (e.g., `model1,model2`).
- `--static-aliases`: The aliases of the models running in the static serving engines, separated by commas and associated using colons (e.g., `model_alias1:model,mode_alias2:model`).
- `--static-backend-health-checks`: Enable this flag to make vllm-router check periodically if the models work by sending dummy requests to their endpoints.
- `--k8s-port`: The port of vLLM processes when using K8s service discovery. Default is `8000`.
- `--k8s-namespace`: The namespace of vLLM pods when using K8s service discovery. Default is `default`.
- `--k8s-label-selector`: The label selector to filter vLLM pods when using K8s service discovery.

### Routing Logic Options

- `--routing-logic`: The routing logic to use. Options are `roundrobin` or `session`. This option is required.
- `--session-key`: The key (in the header) to identify a session.

### Backend HTTP Options

These options configure the existing aiohttp client at startup. Neutree supplies
them as Router process arguments.

| CLI option | Default | Meaning |
| --- | --- | --- |
| `--max-connections-per-backend` | `100` | Connection slots per host, per Router process; `0` is unlimited |
| `--connect-timeout-seconds` | `60` | Connection acquisition, including pool waiting and establishing a connection |
| `--sock-connect-timeout-seconds` | `10` | Socket connection establishment |
| `--sock-read-timeout-seconds` | `600` | Idle wait for response headers or subsequent response data |

There is no total request deadline. Continuous streaming may exceed these
durations. Timeout values accept fractional seconds from `0` to `604800`, with
`0` disabling that timeout. Connection limits are integers from `0` to `65535`.
Omitted arguments use Router defaults; changing these four fields requires
a process restart. Kubernetes discovery otherwise retains the fork baseline;
no relist/reconnect or cache-during-outage fix is adopted from an unmerged PR.

### Monitoring Options

- `--engine-stats-interval`: The interval in seconds to scrape engine statistics. Default is `30`.
- `--request-stats-window`: The sliding window seconds to compute request statistics. Default is `60`.

Backend Gauges use the labels `workspace`, `endpoint`, `server` (backend URL),
and `replica` (inference Pod name). Static and Service-name discovery have no
Pod identity and export an empty `replica`. The scrape target's `pod` label
continues to identify the Router Pod. Request statistics remain indexed by URL,
so reuse of a backend URL can briefly attribute the existing window to a new Pod.

`/metrics` also exports `router:connection_pool_waiting` with these labels for
JSON and multipart audio requests. It counts calls waiting for an
aiohttp connection slot, including cleanup on connection timeout or cancellation.
It is separate from the inference engine's scheduler queue. Each Router has its
own pool; preserve the Router instance when diagnosing sustained waiting.
Thirty-second scrapes may miss brief waiting spikes. Backend labels absent from
discovery and inactive for five minutes are removed once no calls are waiting.

### Logging Options

- `--log-stats`: Log statistics every 30 seconds.

### Dynamic Config Options

- `--dynamic-config-yaml`: The path to the YAML file containing the dynamic configuration.
- `--dynamic-config-json`: The path to the JSON file containing the dynamic configuration.

### Sentry Options

- `--sentry-dsn`: The Sentry Data Source Name to use for error reporting.
- `--sentry-traces-sample-rate`: The sample rate for Sentry traces (0.0 to 1.0). Default is 0.1 (10%).
- `--sentry-profile-session-sample-rate`: The sample rate for Sentry profiling sessions (0.0 to 1.0). Default is 1.0 (100%).

## Build docker image

```bash
docker build -t <image_name>:<tag> -f docker/Dockerfile .
```

## Example commands to run the router

You can install the router using the following command:

```bash
pip install -e .
```

If you want to run the router with the semantic cache, you can install the dependencies using the following command:

```bash
pip install -e .[semantic_cache]
```

**Example 1:** running the router locally at port 8000 in front of multiple serving engines:

```bash
vllm-router --port 8000 \
    --service-discovery static \
    --static-backends "http://localhost:9001,http://localhost:9002,http://localhost:9003" \
    --static-models "facebook/opt-125m,meta-llama/Llama-3.1-8B-Instruct,facebook/opt-125m" \
    --static-aliases "gpt4:meta-llama/Llama-3.1-8B-Instruct" \
    --static-model-types "chat,chat,chat" \
    --static-backend-health-checks \
    --engine-stats-interval 10 \
    --log-stats \
    --routing-logic roundrobin
```

## Backend health checks

By enabling the `--static-backend-health-checks` flag, **vllm-router** will send a simple request to
your LLM nodes every minute to verify that they still work.
If a node is down, it will output a warning and exclude the node from being routed to.

If you enable this flag, its also required that you specify `--static-model-types` as we have to use
different endpoints for each model type.

> Enabling this flag will put some load on your backend every minute as real requests are send to the nodes
> to test their functionality.

## Additional scoped inference APIs

| POST path | Backend path | Body |
| --- | --- | --- |
| `/{workspace}/{endpoint}/pooling` | `/pooling` | JSON object with `model` and task input |
| `/{workspace}/{endpoint}/classify` | `/classify` | JSON object with `model` and task input |
| `/{workspace}/{endpoint}/v1/audio/transcriptions` | `/v1/audio/transcriptions` | Multipart `file` and `model` |
| `/{workspace}/{endpoint}/v1/audio/translations` | `/v1/audio/translations` | Multipart `file` and `model` |

All four paths select backends within the requested workspace and endpoint.
The existing transcription handler now exposes only the Neutree-scoped path.
Transcription rebuilds the known fields supported by the merged upstream audio
implementation, retaining the existing language=en default. Forwarding arbitrary
additional transcription fields is not included. Translation preserves the
original multipart body, including its repeated fields. As in merged upstream,
transcription supports streaming; translation buffers its backend response.
Audio responses preserve the backend status and content type. Requests use the
same native HTTP configuration as other inference calls.
The backend engine must support the requested task; adding a Router path does
not add an engine task or a global model-routing entry.

Static health checks also accept the `pooling`, `classification` and
`translation` model types.

## Dynamic Router Config

The router can be configured dynamically using a config file when passing the `--dynamic-config-yaml` or
`--dynamic-config-json` options. Please note that these are two mutually exclusive options.
The router will watch the config file for changes and update the configuration accordingly (every 10 seconds).

Currently, the dynamic config supports the following fields:

**Required fields:**

- `service_discovery`: The service discovery type. Options are `static` or `k8s`.
- `routing_logic`: The routing logic to use. Options are `roundrobin` or `session`.

**Optional fields:**

- `callbacks`: The path to the callback instance extending CustomCallbackHandler.
- (When using `static` service discovery) `static_backends`: The URLs of static serving engines, separated by commas (e.g., `http://localhost:9001,http://localhost:9002,http://localhost:9003`).
- (When using `static` service discovery) `static_models`: The models running in the static serving engines, separated by commas (e.g., `model1,model2`).
- (When using `static` service discovery) `static_aliases`: The aliases of the models running in the static serving engines, separated by commas and associated using colons (e.g., `model_alias1:model,mode_alias2:model`).
- (When using `static` service discovery and if you enable the `--static-backend-health-checks` flag) `static_model_types`: The model types running in the static serving engines, separated by commas (e.g., `chat,chat`).
- (When using `k8s` service discovery) `k8s_port`: The port of vLLM processes when using K8s service discovery. Default is `8000`.
- (When using `k8s` service discovery) `k8s_namespace`: The namespace of vLLM pods when using K8s service discovery. Default is `default`.
- (When using `k8s` service discovery) `k8s_label_selector`: The label selector to filter vLLM pods when using K8s service discovery.
- `session_key`: The key (in the header) to identify a session when using session-based routing.

Here is an example of a dynamic YAML config file:

```yaml
service_discovery: static
routing_logic: roundrobin
callbacks: module.custom.callback_handler
static_models:
    facebook/opt-125m:
        static_backends:
            - http://localhost:9001
            - http://localhost:9003
        static_model_type: completion
    meta-llama/Llama-3.1-8B-Instruct:
        static_backends:
            - http://localhost:9002
        static_model_type: chat
static_aliases:
    "my-alias": "facebook/opt-125m"
    "my-other-alias": "meta-llama/Llama-3.1-8B-Instruct"
```

Here is an example of a dynamic JSON config file:

```json
{
    "service_discovery": "static",
    "routing_logic": "roundrobin",
    "callbacks": "module.custom.callback_handler",
    "static_backends": "http://localhost:9001,http://localhost:9002,http://localhost:9003",
    "static_models": "facebook/opt-125m,meta-llama/Llama-3.1-8B-Instruct,facebook/opt-125m",
    "static_model_types": "completion,chat,completion",
    "static_aliases": "my-alias:meta-llama/Llama-3.1-8B-Instruct,my-other-alias:meta-llama/Llama-3.1-8B-Instruct"
}
```

### Get current dynamic config

If the dynamic config is enabled, the router will reflect the current dynamic config in the `/health` endpoint.

```bash
curl http://<router_host>:<router_port>/health
```

The response will be a JSON object with the current dynamic config.

```json
{
    "status": "healthy",
    "dynamic_config": <current_dynamic_config (JSON object)>
}
```
