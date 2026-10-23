"""Bounded List/Watch recovery shared by the two Kubernetes discovery modes.

Adapted from production-stack #1092 (7adf99af), including snapshot reconciliation
and watching from the list resourceVersion; preserves Neutree event callbacks.
"""

import random
import threading
import time

from kubernetes import watch
from kubernetes.client.exceptions import ApiException

from vllm_router.log import init_logger

logger = init_logger(__name__)


class KubernetesWatch:
    def __init__(self, owner, list_resources):
        self.owner = owner
        self.list_resources = list_resources
        self.watcher = watch.Watch(return_type=getattr(owner, "_watch_kind", "V1Pod"))
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.timeout = min(owner.watcher_timeout_seconds or 30, 30)
        self.resource_version = None
        self.last_success = None
        self.last_error = None
        self.reconnects = 0
        self.expirations = 0
        self._response = None

    def start(self):
        if not self.stop_event.is_set() and self.thread.ident is None:
            self.thread.start()

    def reconcile(self):
        resources = {}
        token = None
        version = None
        while not self.stop_event.is_set():
            page = self.list_resources(
                namespace=self.owner.namespace,
                label_selector=self.owner.label_selector,
                limit=500,
                _continue=token,
                _request_timeout=(3, 10),
            )
            if version is not None and page.metadata.resource_version != version:
                raise RuntimeError(
                    "Kubernetes list resourceVersion changed between pages"
                )
            version = page.metadata.resource_version
            resources.update((obj.metadata.name, obj) for obj in page.items)
            token = page.metadata._continue
            if not token:
                break
        if self.stop_event.is_set():
            return None
        if not version:
            raise RuntimeError("Kubernetes list returned no resourceVersion")

        # Do not modify routes until every page of the snapshot has succeeded.
        with self.owner.available_engines_lock:
            stale = set(self.owner.available_engines) - resources.keys()
        for name in stale:
            self.owner._delete_engine(name)
        for resource in resources.values():
            if self.stop_event.is_set():
                return None
            try:
                self.owner._handle_resource(resource, "MODIFIED")
            except Exception:
                logger.exception(
                    "Failed to reconcile Kubernetes resource %s", resource.metadata.name
                )
        self.resource_version = version
        self.last_success = time.monotonic()
        if hasattr(self.owner, "_schedule_sessions"):
            self.owner._schedule_sessions()
        return version

    def handle_event(self, event):
        event_type = event["type"]
        raw = event.get("raw_object") or {}
        if event_type == "ERROR":
            raise ApiException(
                status=raw.get("code", 500), reason=raw.get("message", "Watch error")
            )
        if event_type == "BOOKMARK":
            self.resource_version = raw.get("metadata", {}).get(
                "resourceVersion", self.resource_version
            )
        elif event_type in {"ADDED", "MODIFIED", "DELETED"}:
            resource = event["object"]
            self.owner._handle_resource(resource, event_type)
            self.resource_version = resource.metadata.resource_version
        else:
            raise RuntimeError(f"Unexpected Kubernetes watch event: {event_type}")
        self.last_success = time.monotonic()

    def _watch_request(self, *args, **kwargs):
        response = self.list_resources(*args, **kwargs)
        self._response = response
        if self.stop_event.is_set():
            response.close()
        return response

    def run(self):
        failures = 0
        while not self.stop_event.is_set():
            started = time.monotonic()
            try:
                version = self.reconcile()
                if self.stop_event.is_set():
                    break
                for event in self.watcher.stream(
                    self._watch_request,
                    namespace=self.owner.namespace,
                    label_selector=self.owner.label_selector,
                    resource_version=version,
                    allow_watch_bookmarks=True,
                    timeout_seconds=self.timeout,
                    _request_timeout=(3, self.timeout + 5),
                ):
                    if self.stop_event.is_set():
                        break
                    self.handle_event(event)
                self.last_error = None
                if time.monotonic() - started >= self.timeout * 0.8:
                    failures = 0
                    continue
            except Exception as exc:
                if self.stop_event.is_set():
                    break
                self.last_error = f"{type(exc).__name__}: {exc}"
                if isinstance(exc, ApiException) and exc.status == 410:
                    self.expirations += 1
                logger.warning("Kubernetes watch reconnect: %s", self.last_error)
            finally:
                self._response = None
            self.reconnects += 1
            # A short EOF is also a retry, to avoid a tight reconnect loop.
            delay = min(0.5 * 2 ** min(failures, 5), 15)
            failures += 1
            if self.stop_event.wait(random.uniform(delay / 2, delay)):
                break

    def healthy(self):
        return (
            not self.stop_event.is_set()
            and self.thread.is_alive()
            and self.last_success is not None
            and time.monotonic() - self.last_success <= 60
        )

    def close(self):
        self.stop_event.set()
        self.watcher.stop()
        response = self._response
        if response is not None:
            response.close()
        if self.thread.ident is not None:
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                logger.warning(
                    "Kubernetes watcher shutdown is waiting for bounded network I/O"
                )
