from prometheus_client import Counter, Gauge

# --- Prometheus Gauges ---
# Existing metrics
num_requests_running = Gauge(
    "router:num_requests_running",
    "Number of running requests",
    ["workspace", "endpoint", "server"],
)
num_requests_waiting = Gauge(
    "router:num_requests_waiting",
    "Number of waiting requests",
    ["workspace", "endpoint", "server"],
)
gpu_prefix_cache_hit_rate = Gauge(
    "router:gpu_prefix_cache_hit_rate",
    "GPU Prefix Cache Hit Rate",
    ["workspace", "endpoint", "server"],
)
gpu_prefix_cache_hits_total = Gauge(
    "router:gpu_prefix_cache_hits_total",
    "Total GPU Prefix Cache Hits",
    ["workspace", "endpoint", "server"],
)
gpu_prefix_cache_queries_total = Gauge(
    "router:gpu_prefix_cache_queries_total",
    "Total GPU Prefix Cache Queries",
    ["workspace", "endpoint", "server"],
)
current_qps = Gauge(
    "router:current_qps",
    "Current Queries Per Second",
    ["workspace", "endpoint", "server"],
)
avg_decoding_length = Gauge(
    "router:avg_decoding_length",
    "Average Decoding Length",
    ["workspace", "endpoint", "server"],
)
num_prefill_requests = Gauge(
    "router:num_prefill_requests",
    "Number of Prefill Requests",
    ["workspace", "endpoint", "server"],
)
num_decoding_requests = Gauge(
    "router:num_decoding_requests",
    "Number of Decoding Requests",
    ["workspace", "endpoint", "server"],
)
num_incoming_requests_total = Counter(
    "router:num_incoming_requests",
    "Total valid incoming requests to router (including when no backends available).",
    ["workspace", "endpoint"],
)

# New metrics per dashboard update
healthy_pods_total = Gauge(
    "router:healthy_pods_total",
    "Number of healthy inference pods",
    ["workspace", "endpoint", "server"],
)
avg_latency = Gauge(
    "router:avg_latency",
    "Average end-to-end request latency",
    ["workspace", "endpoint", "server"],
)
avg_itl = Gauge(
    "router:avg_itl", "Average Inter-Token Latency", ["workspace", "endpoint", "server"]
)
num_requests_swapped = Gauge(
    "router:num_requests_swapped",
    "Number of swapped requests",
    ["workspace", "endpoint", "server"],
)
