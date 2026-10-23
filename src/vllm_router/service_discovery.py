# Copyright 2024-2025 The vLLM Production Stack Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import abc
import asyncio
import enum
import hashlib
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set

import requests
from kubernetes import client, config

from vllm_router import utils
from vllm_router.backend_http import BackendEndpoint
from vllm_router.kubernetes_watch import KubernetesWatch
from vllm_router.log import init_logger

logger = init_logger(__name__)

_global_service_discovery: "Optional[ServiceDiscovery]" = None


class ServiceDiscoveryType(enum.Enum):
    STATIC = "static"
    K8S = "k8s"


class ServiceDiscoveryEventType(enum.Enum):
    """Event types for service discovery callbacks.

    Only two events are needed for simple routing logic:
    - ENGINE_ADDED: Engine becomes available (pod is ready and healthy)
    - ENGINE_DELETED: Engine becomes unavailable (pod deleted or not ready)
    """

    ENGINE_ADDED = "engine_added"
    ENGINE_DELETED = "engine_deleted"


@dataclass
class ModelInfo:
    """Information about a model including its relationships and metadata."""

    id: str
    object: str
    created: int = 0
    owned_by: str = "vllm"
    root: Optional[str] = None
    parent: Optional[str] = None
    is_adapter: bool = False

    @classmethod
    def from_dict(cls, data: Dict) -> "ModelInfo":
        """Create a ModelInfo instance from a dictionary."""
        return cls(
            id=data.get("id"),
            object=data.get("object", "model"),
            created=data.get("created", int(time.time())),
            owned_by=data.get("owned_by", "vllm"),
            root=data.get("root", None),
            parent=data.get("parent", None),
            is_adapter=data.get("parent") is not None,
        )

    def to_dict(self) -> Dict:
        """Convert the ModelInfo instance to a dictionary."""
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "owned_by": self.owned_by,
            "root": self.root,
            "parent": self.parent,
            "is_adapter": self.is_adapter,
        }


@dataclass
class EndpointInfo:
    # Endpoint's url
    url: str

    # Model names
    model_names: List[str]

    # Endpoint Id
    Id: str

    # Added timestamp
    added_timestamp: float

    # Model label
    model_label: str

    # Endpoint's sleep status
    sleep: bool

    # Pod name
    pod_name: Optional[str] = None

    # Service name
    service_name: Optional[str] = None

    # Namespace
    namespace: Optional[str] = None

    # Model information including relationships
    model_info: Dict[str, ModelInfo] = None

    # Workspace
    workspace: Optional[str] = None

    # Endpoint
    endpoint: Optional[str] = None

    # Routing logic
    routing_logic: Optional[str] = None

    # Whether this engine opts out of OpenAI-shaped routing. Passthrough
    # engines are admitted without a /v1/models probe and are reachable only
    # through the catch-all router, keyed on (workspace, endpoint).
    passthrough: bool = False
    resource_uid: Optional[str] = None
    resource_signature: Optional[tuple] = None

    def __str__(self):
        return f"EndpointInfo(url={self.url}, model_names={self.model_names}, added_timestamp={self.added_timestamp}, model_label={self.model_label}, service_name={self.service_name},pod_name={self.pod_name}, namespace={self.namespace}, workspace={self.workspace}, endpoint={self.endpoint}, routing_logic={self.routing_logic}, passthrough={self.passthrough})"

    def get_base_models(self) -> List[str]:
        """
        Get the list of base models (models without parents) available on this endpoint.
        """
        if not self.model_info:
            return []
        return [
            model_id for model_id, info in self.model_info.items() if not info.parent
        ]

    def get_adapters(self) -> List[str]:
        """
        Get the list of adapters (models with parents) available on this endpoint.
        """
        if not self.model_info:
            return []
        return [model_id for model_id, info in self.model_info.items() if info.parent]

    def get_adapters_for_model(self, base_model: str) -> List[str]:
        """
        Get the list of adapters available for a specific base model.

        Args:
            base_model: The ID of the base model

        Returns:
            List of adapter IDs that are based on the specified model
        """
        if not self.model_info:
            return []
        return [
            model_id
            for model_id, info in self.model_info.items()
            if info.parent == base_model
        ]

    def has_model(self, model_id: str) -> bool:
        """
        Check if a specific model (base model or adapter) is available on this endpoint.

        Args:
            model_id: The ID of the model to check

        Returns:
            True if the model is available, False otherwise
        """
        return model_id in self.model_names

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """
        Get detailed information about a specific model.

        Args:
            model_id: The ID of the model to get information for

        Returns:
            ModelInfo object containing model information if available, None otherwise
        """
        if not self.model_info:
            return None
        return self.model_info.get(model_id)


# Type definition for event callback function
# Callback receives: event_type, engine_name, endpoint_info (optional)
EventCallback = Callable[[ServiceDiscoveryEventType, str, Optional[EndpointInfo]], None]


class ServiceDiscovery(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_endpoint_info(self) -> List[EndpointInfo]:
        """
        Get the URLs of the serving engines that are available for
        querying.

        Returns:
            a list of engine URLs
        """
        pass

    def get_health(self) -> bool:
        """
        Check if the service discovery module is healthy.

        Returns:
            True if the service discovery module is healthy, False otherwise
        """
        return True

    def close(self) -> None:
        """
        Close the service discovery module.
        """
        pass


class StaticServiceDiscovery(ServiceDiscovery):
    def __init__(
        self,
        app,
        urls: List[str],
        models: List[str],
        aliases: List[str] | None = None,
        model_labels: List[str] | None = None,
        model_types: List[str] | None = None,
        static_backend_health_checks: bool = False,
        prefill_model_labels: List[str] | None = None,
        decode_model_labels: List[str] | None = None,
    ):
        self.app = app
        assert len(urls) == len(models), "URLs and models should have the same length"
        self.urls = urls
        self.models = models
        self.aliases = aliases
        self.model_labels = model_labels
        self.model_types = model_types
        self.engines_id = [str(uuid.uuid4()) for i in range(0, len(urls))]
        self.added_timestamp = int(time.time())
        self.unhealthy_endpoint_hashes = []
        self._running = True
        if static_backend_health_checks:
            self.start_health_check_task()
        self.prefill_model_labels = prefill_model_labels
        self.decode_model_labels = decode_model_labels

    def get_unhealthy_endpoint_hashes(self) -> list[str]:
        unhealthy_endpoints = []
        try:
            for url, model, model_type in zip(
                self.urls, self.models, self.model_types, strict=True
            ):
                if utils.is_model_healthy(url, model, model_type):
                    logger.debug(f"{model} at {url} is healthy")
                else:
                    logger.warning(f"{model} at {url} not healthy!")
                    unhealthy_endpoints.append(self.get_model_endpoint_hash(url, model))
        except ValueError:
            logger.error(
                "To perform health check, each model has to define a static_model_type and at least one static_backend. "
                "Skipping health checks for now."
            )
        return unhealthy_endpoints

    async def check_model_health(self):
        while self._running:
            try:
                self.unhealthy_endpoint_hashes = self.get_unhealthy_endpoint_hashes()
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                logger.debug("Health check task cancelled")
                break
            except Exception as e:
                logger.error(e)

    def start_health_check_task(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        asyncio.run_coroutine_threadsafe(self.check_model_health(), self.loop)
        logger.info("Health check thread started")

    def get_model_endpoint_hash(self, url: str, model: str) -> str:
        return hashlib.md5(f"{url}{model}".encode()).hexdigest()

    def _get_model_info(self, model: str) -> Dict[str, ModelInfo]:
        """
        Get detailed model information. For static serving engines, we don't query the engine, instead we use predefined
        static model info.

        Args:
            model: the model name

        Returns:
            Dictionary mapping model IDs to their information, including parent-child relationships
        """
        return {
            model: ModelInfo(
                id=model,
                object="model",
                owned_by="vllm",
                parent=None,
                is_adapter=False,
                root=None,
                created=int(time.time()),
            )
        }

    def get_endpoint_info(self) -> List[EndpointInfo]:
        """
        Get the URLs of the serving engines that are available for
        querying.

        Returns:
            a list of engine URLs
        """
        endpoint_infos = []
        for i, (url, model) in enumerate(zip(self.urls, self.models)):
            if (
                self.get_model_endpoint_hash(url, model)
                in self.unhealthy_endpoint_hashes
            ):
                continue
            model_label = self.model_labels[i] if self.model_labels else "default"
            endpoint_info = EndpointInfo(
                url=url,
                model_names=[model],  # Convert single model to list
                Id=self.engines_id[i],
                sleep=False,
                added_timestamp=self.added_timestamp,
                model_label=model_label,
                model_info=self._get_model_info(model),
            )
            endpoint_infos.append(endpoint_info)
        return endpoint_infos

    async def initialize_client_sessions(self) -> None:
        """
        Select P/D endpoint metadata; networking uses the shared backend wrapper.
        """
        if self.prefill_model_labels is None or self.decode_model_labels is None:
            return
        endpoints = self.get_endpoint_info()
        for role, labels in (
            ("prefill", self.prefill_model_labels),
            ("decode", self.decode_model_labels),
        ):
            selected = next((ep for ep in endpoints if ep.model_label in labels), None)
            name = f"{role}_client"
            current = getattr(self.app.state, name, None)
            if selected is None:
                setattr(self.app.state, name, None)
            else:
                target = BackendEndpoint(
                    selected.url, selected.workspace or "", selected.endpoint or ""
                )
                if current != target:
                    setattr(self.app.state, name, target)

    def close(self):
        """
        Close the service discovery module and clean up health check resources.
        """
        self._running = False
        if hasattr(self, "loop") and self.loop.is_running():
            # Schedule a coroutine to gracefully shut down the event loop
            async def shutdown():
                tasks = [
                    t
                    for t in asyncio.all_tasks(self.loop)
                    if t is not asyncio.current_task()
                ]
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                self.loop.stop()

            future = asyncio.run_coroutine_threadsafe(shutdown(), self.loop)
            try:
                future.result(timeout=15.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "Timed out waiting for shutdown(loop might already be closed)"
                )
            except Exception as e:
                logger.warning(f"Error during health check shutdown: {e}")

        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join(timeout=5.0)

        if hasattr(self, "loop") and not self.loop.is_closed():
            self.loop.close()


class _KubernetesDiscovery:
    """Own watcher/session lifetime; no thread waits for the FastAPI loop."""

    def _init_watcher(self, list_resources, kind):
        self.event_loop = None
        self._session_refresh_pending = False
        self._session_lock = None
        self._watch_kind = kind
        self._watch = KubernetesWatch(self, list_resources)
        self.k8s_watcher = self._watch.watcher
        self.watcher_thread = self._watch.thread
        self.running = True
        loop = getattr(self.app.state, "event_loop", None)
        if loop is not None and loop.is_running():
            self.set_event_loop(loop)

    def set_event_loop(self, loop):
        self.event_loop = loop
        self._watch.start()
        self._schedule_sessions()

    def _schedule_sessions(self):
        if not self.running or self.event_loop is None or self._session_refresh_pending:
            return
        if self.prefill_model_labels is None or self.decode_model_labels is None:
            return
        self._session_refresh_pending = True

        async def refresh():
            try:
                await self.initialize_client_sessions()
            finally:
                self._session_refresh_pending = False

        if not self.event_loop.is_closed():
            future = asyncio.run_coroutine_threadsafe(refresh(), self.event_loop)

            def finished(result):
                try:
                    result.result()
                except Exception:
                    logger.exception("Failed to refresh P/D clients")

            future.add_done_callback(finished)
        else:
            self._session_refresh_pending = False

    def _watch_engines(self):
        self._watch.run()

    def _reconcile_engines(self):
        return self._watch.reconcile()

    def get_health(self):
        return self._watch.healthy()

    def get_endpoint_info(self):
        # Keep a failed List from destroying the cache, but stop routing stale data.
        if self._watch.last_success is not None and not self._watch.healthy():
            return []
        with self.available_engines_lock:
            return list(self.available_engines.values())

    def close(self):
        self.running = False
        self._watch.close()

    async def initialize_client_sessions(self):
        if self.prefill_model_labels is None or self.decode_model_labels is None:
            return
        if self._session_lock is None:
            self._session_lock = asyncio.Lock()
        async with self._session_lock:
            if not self.running:
                return
            endpoints = self.get_endpoint_info()
            for role, labels in (
                ("prefill", self.prefill_model_labels),
                ("decode", self.decode_model_labels),
            ):
                name = f"{role}_client"
                selected = next(
                    (ep for ep in endpoints if ep.model_label in labels), None
                )
                old = getattr(self.app.state, name, None)
                if selected is None:
                    setattr(self.app.state, name, None)
                    continue
                target = BackendEndpoint(
                    selected.url, selected.workspace or "", selected.endpoint or ""
                )
                if old != target:
                    setattr(self.app.state, name, target)

    async def close_client_sessions(self):
        self.app.state.prefill_client = None
        self.app.state.decode_client = None


class K8sPodIPServiceDiscovery(_KubernetesDiscovery, ServiceDiscovery):
    def __init__(
        self,
        app,
        namespace: str,
        port: str,
        label_selector=None,
        prefill_model_labels: List[str] | None = None,
        decode_model_labels: List[str] | None = None,
        watcher_timeout_seconds: int = 0,
        health_check_timeout_seconds: int = 10,
        event_callbacks: List[EventCallback] | None = None,
    ):
        """
        Initialize the Kubernetes service discovery module. This module
        assumes all serving engine pods are in the same namespace, listening
        on the same port, and have the same label selector.

        It will start a daemon thread to watch the engine pods and update
        the url of the available engines.

        Args:
            namespace: the namespace of the engine pods
            port: the port of the engines
            label_selector: the label selector of the engines
            watcher_timeout_seconds: timeout in seconds for Kubernetes watcher streams (default: 0)
            event_callbacks: list of callback functions to be invoked on service discovery events
        """
        self.app = app
        self.namespace = namespace
        self.port = port
        self.available_engines: Dict[str, EndpointInfo] = {}
        self.available_engines_lock = threading.Lock()
        self.known_models: Set[str] = set()
        self.known_models_lock = threading.Lock()
        self.label_selector = label_selector
        self.watcher_timeout_seconds = watcher_timeout_seconds
        self.health_check_timeout_seconds = health_check_timeout_seconds
        self.event_callbacks = event_callbacks or []

        # Init kubernetes watcher
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        self.k8s_api = client.CoreV1Api()
        self.prefill_model_labels = prefill_model_labels
        self.decode_model_labels = decode_model_labels
        self._init_watcher(self.k8s_api.list_namespaced_pod, "V1Pod")

    def _trigger_callbacks(
        self,
        event_type: ServiceDiscoveryEventType,
        engine_name: str,
        endpoint_info: Optional[EndpointInfo] = None,
    ) -> None:
        """
        Trigger all registered event callbacks.

        Args:
            event_type: the type of event that occurred
            engine_name: the name of the engine
            endpoint_info: optional endpoint information
        """
        for callback in self.event_callbacks:
            try:
                callback(event_type, engine_name, endpoint_info)
            except Exception as e:
                logger.error(
                    f"Error executing callback for event {event_type} on engine {engine_name}: {e}"
                )

    def register_callback(self, callback: EventCallback) -> None:
        """
        Register a new event callback.

        Args:
            callback: the callback function to register
        """
        if callback not in self.event_callbacks:
            self.event_callbacks.append(callback)
            logger.info(f"Registered new event callback: {callback.__name__}")

    def unregister_callback(self, callback: EventCallback) -> None:
        """
        Unregister an event callback.

        Args:
            callback: the callback function to unregister
        """
        if callback in self.event_callbacks:
            self.event_callbacks.remove(callback)
            logger.info(f"Unregistered event callback: {callback.__name__}")

    @staticmethod
    def _check_pod_ready(container_statuses):
        """
        Check if all containers in the pod are ready by reading the
        k8s container statuses.
        """
        if not container_statuses:
            return False
        ready_count = sum(1 for status in container_statuses if status.ready)
        return ready_count == len(container_statuses)

    @staticmethod
    def _is_pod_terminating(pod):
        """
        Check if the pod is in terminating state by checking
        deletion timestamp.
        """
        return pod.metadata.deletion_timestamp is not None

    def _get_engine_sleep_status(self, pod_ip) -> Optional[bool]:
        """
        Get the engine sleeping status by querying the engine's
        '/is_sleeping' endpoint.

        Args:
            pod_ip: the IP address of the pod running the engine

        Returns:
            the sleep status of the target engine
        """
        url = f"http://{pod_ip}:{self.port}/is_sleeping"
        try:
            headers = None
            if VLLM_API_KEY := os.getenv("VLLM_API_KEY"):
                logger.info("Using vllm server authentication")
                headers = {"Authorization": f"Bearer {VLLM_API_KEY}"}
            response = requests.get(
                url, headers=headers, timeout=self.health_check_timeout_seconds
            )
            response.raise_for_status()
            sleep = response.json()["is_sleeping"]
            return sleep
        except Exception as e:
            logger.warning(
                f"Failed to get the sleep status for engine at {url} - sleep status is set to `False`: {e}"
            )
            return False

    def _check_engine_sleep_mode(self, pod_name) -> Optional[bool]:
        try:
            enable_sleep_mode = False
            pod = self.k8s_api.read_namespaced_pod(
                name=pod_name, namespace=self.namespace, _request_timeout=(3, 10)
            )
            for container in pod.spec.containers:
                if container.name == "vllm":
                    if (
                        not container.command
                        or "--enable-sleep-mode" in container.command
                    ):
                        enable_sleep_mode = True
                    break
            return enable_sleep_mode
        except client.rest.ApiException as e:
            logger.error(
                f"Error checking if sleep-mode is enable for pod {pod_name}: {e}"
            )
            return False

    def add_sleep_label(self, pod_name):
        try:
            pod = self.k8s_api.read_namespaced_pod(
                name=pod_name, namespace=self.namespace, _request_timeout=(3, 10)
            )
            # pod.metadata.labels = {"sleeping": "true"}
            pod.metadata.labels.update({"sleeping": "true"})
            self.k8s_api.patch_namespaced_pod(
                name=pod_name, namespace=self.namespace, body=pod
            )
            logger.info(f"Sleeping label added to the pod: {pod_name}")

        except client.rest.ApiException as e:
            logger.error(f"Error adding sleeping label to the pod {pod_name}: {e}")

    def remove_sleep_label(self, pod_name):
        try:
            label_key = "sleeping"
            body = {"metadata": {"labels": {label_key: None}}}

            pod = self.k8s_api.read_namespaced_pod(
                name=pod_name, namespace=self.namespace, _request_timeout=(3, 10)
            )
            if label_key in pod.metadata.labels:
                self.k8s_api.patch_namespaced_pod(
                    name=pod_name, namespace=self.namespace, body=body
                )
                logger.info(f"Label `sleeping=true` removed from pod '{pod_name}'")
            else:
                logger.info(
                    f"Label `sleeping=true` not found on pod '{pod_name}' in namespace '{self.namespace}'"
                )

        except client.rest.ApiException as e:
            logger.error(f"Error removing sleeping label: {e}")

    def _get_model_names(self, pod_ip) -> List[str]:
        """
        Get the model names of the serving engine pod by querying the pod's
        '/v1/models' endpoint.

        Args:
            pod_ip: the IP address of the pod

        Returns:
            List of model names available on the serving engine, including both base models and adapters
        """
        url = f"http://{pod_ip}:{self.port}/v1/models"
        try:
            headers = None
            if VLLM_API_KEY := os.getenv("VLLM_API_KEY"):
                logger.info("Using vllm server authentication")
                headers = {"Authorization": f"Bearer {VLLM_API_KEY}"}
            response = requests.get(
                url, headers=headers, timeout=self.health_check_timeout_seconds
            )
            response.raise_for_status()
            models = response.json()["data"]

            # Collect all model names, including both base models and adapters
            model_names = []
            for model in models:
                model_id = model["id"]
                model_names.append(model_id)

            logger.info(f"Found models on pod {pod_ip}: {model_names}")
            return model_names
        except Exception as e:
            logger.error(f"Failed to get model names from {url}: {e}")
            return []

    def _get_model_info(self, pod_ip) -> Dict[str, ModelInfo]:
        """
        Get detailed model information from the serving engine pod.

        Args:
            pod_ip: the IP address of the pod

        Returns:
            Dictionary mapping model IDs to their ModelInfo objects, including parent-child relationships
        """
        url = f"http://{pod_ip}:{self.port}/v1/models"
        try:
            headers = None
            if VLLM_API_KEY := os.getenv("VLLM_API_KEY"):
                logger.info("Using vllm server authentication")
                headers = {"Authorization": f"Bearer {VLLM_API_KEY}"}
            response = requests.get(
                url, headers=headers, timeout=self.health_check_timeout_seconds
            )
            response.raise_for_status()
            models = response.json()["data"]
            # Create a dictionary of model information
            model_info = {}
            for model in models:
                model_id = model["id"]
                model_info[model_id] = ModelInfo.from_dict(model)

            return model_info
        except Exception as e:
            logger.error(f"Failed to get model info from {url}: {e}")
            return {}

    def _get_model_label(self, pod) -> Optional[str]:
        """
        Get the model label from the pod's metadata labels.

        Args:
            pod: The Kubernetes pod object

        Returns:
            The model label if found, None otherwise
        """
        if not pod.metadata.labels:
            return None
        return pod.metadata.labels.get("model")

    def _get_workspace(self, pod) -> Optional[str]:
        """
        Get the workspace from the pod's metadata labels.

        Args:
            pod: The Kubernetes pod object

        Returns:
            The workspace if found, None otherwise
        """
        if not pod.metadata.labels:
            return None
        return pod.metadata.labels.get("workspace")

    def _get_endpoint(self, pod) -> Optional[str]:
        """
        Get the endpoint from the pod's metadata labels.

        Args:
            pod: The Kubernetes pod object

        Returns:
            The endpoint if found, None otherwise
        """
        if not pod.metadata.labels:
            return None
        return pod.metadata.labels.get("endpoint")

    def _get_routing_logic(self, pod) -> Optional[str]:
        """
        Get the routing logic from the pod's metadata labels.

        Args:
            pod: The Kubernetes pod object

        Returns:
            The routing logic if found, None otherwise
        """
        if not pod.metadata.labels:
            return None
        return pod.metadata.labels.get("routing_logic")

    def _get_passthrough(self, pod) -> bool:
        """
        Get the passthrough flag from the pod's metadata labels.

        A pod labelled ``passthrough=true`` serves a workload that is not
        OpenAI-compatible. It is admitted to the pool without a /v1/models
        probe and is only reachable through the catch-all router.

        Args:
            pod: The Kubernetes pod object

        Returns:
            True if the pod opts into passthrough routing, False otherwise
        """
        if not pod.metadata.labels:
            return False
        return pod.metadata.labels.get("passthrough", "").lower() == "true"

    def _handle_resource(self, pod, event_type):
        name = pod.metadata.name
        uid = pod.metadata.uid
        with self.available_engines_lock:
            old = self.available_engines.get(name)
        if event_type == "DELETED":
            if old is not None and (
                old.resource_uid is None or old.resource_uid == uid
            ):
                self._delete_engine(name)
            return
        ip = pod.status.pod_ip
        ready = (
            bool(ip)
            and self._check_pod_ready(pod.status.container_statuses)
            and not self._is_pod_terminating(pod)
        )
        if not ready:
            self._delete_engine(name)
            return
        labels = pod.metadata.labels or {}
        lora_modified = (pod.metadata.annotations or {}).get("lora-modified")
        signature = (uid, ip, tuple(sorted(labels.items())), lora_modified)
        if old is not None and old.resource_signature == signature:
            return
        # Invalidate changed identities before probing, but retain same-identity routes
        # across a transient probe failure (the changed signature will be retried).
        identity_changed = old is not None and (
            old.resource_uid != uid
            or old.url != f"http://{ip}:{self.port}"
            or old.workspace != self._get_workspace(pod)
            or old.endpoint != self._get_endpoint(pod)
            or old.passthrough != self._get_passthrough(pod)
        )
        if identity_changed:
            self._delete_engine(name)
        passthrough = self._get_passthrough(pod)
        models = [] if passthrough else self._get_model_names(ip)
        if not self._is_admissible(
            is_pod_ready=ready,
            model_names=models,
            passthrough=passthrough,
            workspace=self._get_workspace(pod),
            endpoint=self._get_endpoint(pod),
        ):
            return
        if not self.running:
            return
        if old is not None and not identity_changed:
            self._delete_engine(name)
        self._add_engine(
            name,
            ip,
            models,
            self._get_model_label(pod),
            self._get_workspace(pod),
            self._get_endpoint(pod),
            self._get_routing_logic(pod),
            passthrough,
            resource_uid=uid,
            resource_signature=signature,
            sleep_status=(
                (labels["sleeping"] == "true") if "sleeping" in labels else None
            ),
        )

    def _add_engine(
        self,
        engine_name: str,
        engine_ip: str,
        model_names: List[str],
        model_label: str,
        workspace: str,
        endpoint: str,
        routing_logic: str,
        passthrough: bool,
        resource_uid=None,
        resource_signature=None,
        sleep_status=None,
    ):
        logger.info(
            f"Discovered new serving engine {engine_name} at "
            f"{engine_ip}, running models: {model_names}, passthrough: {passthrough}"
        )

        # Get detailed model information. Passthrough engines serve no
        # /v1/models, so skip the probe rather than pay its timeout.
        model_info = {} if passthrough else self._get_model_info(engine_ip)
        if not self.running:
            return

        # Check if engine is enabled with sleep mode and set engine sleep status
        if sleep_status is None:
            if not passthrough and self._check_engine_sleep_mode(engine_name):
                sleep_status = self._get_engine_sleep_status(engine_ip)
            else:
                sleep_status = False
        if not self.running:
            return

        with self.available_engines_lock:
            self.available_engines[engine_name] = EndpointInfo(
                url=f"http://{engine_ip}:{self.port}",
                model_names=model_names,
                added_timestamp=int(time.time()),
                Id=str(uuid.uuid5(uuid.NAMESPACE_DNS, engine_name)),
                model_label=model_label,
                sleep=sleep_status,
                pod_name=engine_name,
                namespace=self.namespace,
                model_info=model_info,
                workspace=workspace,
                endpoint=endpoint,
                routing_logic=routing_logic,
                passthrough=passthrough,
                resource_uid=resource_uid,
                resource_signature=resource_signature,
            )

            # Store model information in the endpoint info
            self.available_engines[engine_name].model_info = model_info

            # Get the endpoint info for callback
            endpoint_info = self.available_engines[engine_name]

        # Trigger callbacks after releasing lock
        self._trigger_callbacks(
            ServiceDiscoveryEventType.ENGINE_ADDED,
            engine_name,
            endpoint_info,
        )

        self._schedule_sessions()

        # Track all models we've ever seen
        with self.known_models_lock:
            self.known_models.update(model_names)

    def _delete_engine(self, engine_name: str):
        with self.available_engines_lock:
            endpoint_info = self.available_engines.pop(engine_name, None)
        if endpoint_info is None:
            return
        self._trigger_callbacks(
            ServiceDiscoveryEventType.ENGINE_DELETED, engine_name, endpoint_info
        )
        self._schedule_sessions()

    @staticmethod
    def _is_admissible(
        *,
        is_pod_ready: bool,
        model_names: List[str],
        passthrough: bool,
        workspace: Optional[str],
        endpoint: Optional[str],
    ) -> bool:
        """
        Decide whether a pod belongs in the routable pool.

        An OpenAI-compatible engine qualifies by advertising at least one model.
        A passthrough engine has no models to advertise, so it qualifies on its
        routing key instead: without both labels it could never be selected,
        and admitting it would only add an unreachable entry to the pool.
        """
        if not is_pod_ready:
            return False
        if passthrough:
            return bool(workspace and endpoint)
        return bool(model_names)

    def _on_engine_update(
        self,
        engine_name: str,
        engine_ip: Optional[str],
        event: str,
        is_pod_ready: bool,
        model_names: List[str],
        model_label: Optional[str],
        workspace: Optional[str],
        endpoint: Optional[str],
        routing_logic: Optional[str],
        passthrough: bool,
    ) -> None:
        """
        Handle engine update events from Kubernetes watcher.

        Simple state machine:
        - If pod becomes ready and healthy -> trigger ENGINE_ADDED
        - If pod becomes unavailable (deleted or not ready) -> trigger ENGINE_DELETED
        """
        if event == "ADDED":
            if engine_ip is None:
                return

            if not self._is_admissible(
                is_pod_ready=is_pod_ready,
                model_names=model_names,
                passthrough=passthrough,
                workspace=workspace,
                endpoint=endpoint,
            ):
                return

            self._add_engine(
                engine_name,
                engine_ip,
                model_names,
                model_label,
                workspace,
                endpoint,
                routing_logic,
                passthrough,
            )

        elif event == "DELETED":
            if engine_name not in self.available_engines:
                return

            self._delete_engine(engine_name)

        elif event == "MODIFIED":
            if engine_ip is None:
                return

            # Check if engine availability status changed
            was_available = engine_name in self.available_engines
            is_now_available = self._is_admissible(
                is_pod_ready=is_pod_ready,
                model_names=model_names,
                passthrough=passthrough,
                workspace=workspace,
                endpoint=endpoint,
            )

            if is_now_available and not was_available:
                # Engine became available: trigger ENGINE_ADDED
                self._add_engine(
                    engine_name,
                    engine_ip,
                    model_names,
                    model_label,
                    workspace,
                    endpoint,
                    routing_logic,
                    passthrough,
                )
            elif not is_now_available and was_available:
                # Engine became unavailable: trigger ENGINE_DELETED
                self._delete_engine(engine_name)

    def has_ever_seen_model(self, model_name: str) -> bool:
        """Check if we've ever seen this model, even if currently scaled to zero."""
        with self.known_models_lock:
            return model_name in self.known_models

    def get_known_models(self) -> Set[str]:
        """Get all models that have ever been discovered."""
        with self.known_models_lock:
            return self.known_models.copy()


class K8sServiceNameServiceDiscovery(_KubernetesDiscovery, ServiceDiscovery):
    def __init__(
        self,
        app,
        namespace: str,
        port: str,
        label_selector=None,
        prefill_model_labels: List[str] | None = None,
        decode_model_labels: List[str] | None = None,
        watcher_timeout_seconds: int = 0,
        health_check_timeout_seconds: int = 10,
    ):
        """
        Initialize the Kubernetes service discovery module. This module
        assumes all serving engine services are in the same namespace, listening
        on the same port, and have the same label selector.

        For the routing logic, this approach cannot perform advanced routing
        strategies such as kvaware or PD routing. Instead, it relies on
        Kubernetes'native service-level load-balancing mechanisms, such
        as round-robin or session affinity-based routing.

        Regarding metrics collection, the monitoring system typically scrapes vLLM
        metrics directly from individual pods. When metrics collection is performed
        at the service level, there is no guarantee that the collected metrics
        correspond to the specific pod that handled a given inference request,
        especially in multi-replica deployments.

        Therefore, to ensure full functionality of the production stack including
        accurate routing and metrics, this service discovery mechanismis recommended
        for deployments where each service maps to a single pod(1:1 service-to-pod ratio).

        It will start a daemon thread to watch the engine services and update
        the url of the available engines.

        Args:
            namespace: the namespace of the engine services
            port: the port of the engines
            label_selector: the label selector of the engines
            watcher_timeout_seconds: timeout in seconds for Kubernetes watcher streams (default: 0)
            health_check_timeout_seconds: timeout in seconds for health check requests (default: 10)
        """
        self.app = app
        self.namespace = namespace
        self.port = port
        self.available_engines: Dict[str, EndpointInfo] = {}
        self.available_engines_lock = threading.Lock()
        self.label_selector = label_selector
        self.watcher_timeout_seconds = watcher_timeout_seconds
        self.health_check_timeout_seconds = health_check_timeout_seconds

        # Init kubernetes watcher
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        self.k8s_api = client.CoreV1Api()
        self.prefill_model_labels = prefill_model_labels
        self.decode_model_labels = decode_model_labels
        self._init_watcher(self.k8s_api.list_namespaced_service, "V1Service")

    def _check_service_ready(self, service_name, namespace):
        endpoints = self.k8s_api.read_namespaced_endpoints(
            service_name, namespace, _request_timeout=(3, 10)
        )
        if not endpoints.subsets:
            return False

        for subset in endpoints.subsets:
            if subset.addresses:
                return True
        return False

    def _get_engine_sleep_status(self, service_name) -> Optional[bool]:
        """
        Get the engine sleeping status by querying the engine's
        '/is_sleeping' endpoint.

        Args:
            service_name: the name of the service running the engine

        Returns:
            the sleep status of the target engine
        """
        url = f"http://{service_name}:{self.port}/is_sleeping"
        try:
            headers = None
            if VLLM_API_KEY := os.getenv("VLLM_API_KEY"):
                logger.info("Using vllm server authentication")
                headers = {"Authorization": f"Bearer {VLLM_API_KEY}"}
            response = requests.get(
                url, headers=headers, timeout=self.health_check_timeout_seconds
            )
            response.raise_for_status()
            sleep = response.json()["is_sleeping"]
            return sleep
        except Exception as e:
            logger.warning(
                f"Failed to get the sleep status for engine at {url} - sleep status is set to `False`: {e}"
            )
            return False

    def _check_engine_sleep_mode(self, service_name) -> Optional[bool]:
        try:
            service = self.k8s_api.read_namespaced_service(
                service_name, self.namespace, _request_timeout=(3, 10)
            )
            if not service.spec.selector:
                return False

            selector = ",".join([f"{k}={v}" for k, v in service.spec.selector.items()])
            pods = self.k8s_api.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=selector,
                _request_timeout=(3, 10),
            )

            if not pods.items:
                logger.warning(
                    f"No pods found for service {service_name} in namespace {self.namespace}"
                )
                return False

            enable_sleep_mode = False
            for container in pods.items[0].spec.containers:
                if container.name == "vllm":
                    for arg in container.command or []:
                        if arg == "--enable-sleep-mode":
                            enable_sleep_mode = True
                            break
            return enable_sleep_mode
        except client.rest.ApiException as e:
            logger.error(
                f"Error checking if sleep-mode is enable for service {service_name}: {e}"
            )
            return False

    def add_sleep_label(self, service_name):
        try:
            body = {"metadata": {"labels": {"sleeping": "true"}}}
            self.k8s_api.patch_namespaced_service(
                name=service_name, namespace=self.namespace, body=body
            )
            logger.info(f"Sleeping label added to the service: {service_name}")

        except client.rest.ApiException as e:
            logger.error(
                f"Error adding sleeping label to the service {service_name}: {e}"
            )

    def remove_sleep_label(self, service_name):
        try:
            label_key = "sleeping"
            body = {"metadata": {"labels": {label_key: None}}}

            service = self.k8s_api.read_namespaced_service(
                name=service_name, namespace=self.namespace
            )
            if label_key in service.metadata.labels:
                self.k8s_api.patch_namespaced_service(
                    name=service_name, namespace=self.namespace, body=body
                )
                logger.info(
                    f"Label `sleeping=true` removed from service '{service_name}'"
                )
            else:
                logger.info(
                    f"Label `sleeping=true` not found on service '{service_name}' in namespace '{self.namespace}'"
                )

        except client.rest.ApiException as e:
            logger.error(f"Error removing sleeping label: {e}")

    def _get_model_names(self, service_name) -> List[str]:
        """
        Get the model names of the serving engine service by querying the service's
        '/v1/models' endpoint.

        Args:
            service_name: the name of the service

        Returns:
            List of model names available on the serving engine, including both base models and adapters
        """
        url = f"http://{service_name}:{self.port}/v1/models"
        try:
            headers = None
            if VLLM_API_KEY := os.getenv("VLLM_API_KEY"):
                logger.info("Using vllm server authentication")
                headers = {"Authorization": f"Bearer {VLLM_API_KEY}"}
            response = requests.get(
                url, headers=headers, timeout=self.health_check_timeout_seconds
            )
            response.raise_for_status()
            models = response.json()["data"]

            # Collect all model names, including both base models and adapters
            model_names = []
            for model in models:
                model_id = model["id"]
                model_names.append(model_id)

            logger.info(f"Found models on service {service_name}: {model_names}")
            return model_names
        except Exception as e:
            logger.error(f"Failed to get model names from {url}: {e}")
            return []

    def _get_model_info(self, service_name) -> Dict[str, ModelInfo]:
        """
        Get detailed model information from the serving engine service.

        Args:
            service_name: the IP name of the service

        Returns:
            Dictionary mapping model IDs to their ModelInfo objects, including parent-child relationships
        """
        url = f"http://{service_name}:{self.port}/v1/models"
        try:
            headers = None
            if VLLM_API_KEY := os.getenv("VLLM_API_KEY"):
                logger.info("Using vllm server authentication")
                headers = {"Authorization": f"Bearer {VLLM_API_KEY}"}
            response = requests.get(
                url, headers=headers, timeout=self.health_check_timeout_seconds
            )
            response.raise_for_status()
            models = response.json()["data"]
            # Create a dictionary of model information
            model_info = {}
            for model in models:
                model_id = model["id"]
                model_info[model_id] = ModelInfo.from_dict(model)

            return model_info
        except Exception as e:
            logger.error(f"Failed to get model info from {url}: {e}")
            return {}

    def _get_model_label(self, service) -> Optional[str]:
        """
        Get the model label from the service's selector.

        Args:
            service: The Kubernetes service object

        Returns:
            The model selector if found, None otherwise
        """
        if not service.spec.selector:
            return None
        return service.spec.selector.get("model")

    def _handle_resource(self, service, event_type):
        name = service.metadata.name
        uid = service.metadata.uid
        with self.available_engines_lock:
            old = self.available_engines.get(name)
        if event_type == "DELETED":
            if old is not None and (
                old.resource_uid is None or old.resource_uid == uid
            ):
                self._delete_engine(name)
            return
        # Invalidate a replacement before any readiness/probe call can fail.
        if old is not None and old.resource_uid != uid:
            self._delete_engine(name)
        if (
            service.metadata.deletion_timestamp is not None
            or not self._check_service_ready(name, self.namespace)
        ):
            self._delete_engine(name)
            return
        labels = service.metadata.labels or {}
        signature = (
            uid,
            tuple(sorted(labels.items())),
            tuple(sorted((service.spec.selector or {}).items())),
            (service.metadata.annotations or {}).get("lora-modified"),
        )
        if old is not None and old.resource_signature == signature:
            return
        models = self._get_model_names(name)
        if models and self.running:
            if old is not None:
                self._delete_engine(name)
            self._add_engine(
                name,
                models,
                self._get_model_label(service),
                resource_uid=uid,
                resource_signature=signature,
            )

    def _add_engine(
        self,
        engine_name: str,
        model_names: List[str],
        model_label: str,
        resource_uid=None,
        resource_signature=None,
    ):
        logger.info(
            f"Discovered new serving engine {engine_name} at "
            f"running models: {model_names}"
        )

        # Get detailed model information
        model_info = self._get_model_info(engine_name)

        # Check if engine is enabled with sleep mode and set engine sleep status
        if self._check_engine_sleep_mode(engine_name):
            sleep_status = self._get_engine_sleep_status(engine_name)
        else:
            sleep_status = False

        if not self.running:
            return

        with self.available_engines_lock:
            self.available_engines[engine_name] = EndpointInfo(
                url=f"http://{engine_name}:{self.port}",
                model_names=model_names,
                added_timestamp=int(time.time()),
                Id=str(uuid.uuid5(uuid.NAMESPACE_DNS, engine_name)),
                model_label=model_label,
                sleep=sleep_status,
                service_name=engine_name,
                namespace=self.namespace,
                model_info=model_info,
                resource_uid=resource_uid,
                resource_signature=resource_signature,
            )

            # Store model information in the endpoint info
            self.available_engines[engine_name].model_info = model_info

        self._schedule_sessions()

    def _delete_engine(self, engine_name: str):
        with self.available_engines_lock:
            removed = self.available_engines.pop(engine_name, None)
        if removed is not None:
            self._schedule_sessions()

    def _on_engine_update(
        self,
        engine_name: str,
        event: str,
        is_service_ready: bool,
        model_names: List[str],
        model_label: Optional[str],
    ) -> None:
        if event == "ADDED":
            if not engine_name:
                return

            if not is_service_ready:
                return

            if not model_names:
                return

            self._add_engine(engine_name, model_names, model_label)

        elif event == "DELETED":
            if engine_name not in self.available_engines:
                return

            self._delete_engine(engine_name)

        elif event == "MODIFIED":
            if not engine_name:
                return

            if is_service_ready and model_names:
                self._add_engine(engine_name, model_names, model_label)
                return

            if (
                not is_service_ready or not model_names
            ) and engine_name in self.available_engines:
                self._delete_engine(engine_name)
                return


def _create_service_discovery(
    service_discovery_type: ServiceDiscoveryType, *args, **kwargs
) -> ServiceDiscovery:
    """
    Create a service discovery module with the given type and arguments.

    Args:
        service_discovery_type: the type of service discovery module
        *args: positional arguments for the service discovery module
        **kwargs: keyword arguments for the service discovery module

    Returns:
        the created service discovery module
    """

    if service_discovery_type == ServiceDiscoveryType.STATIC:
        return StaticServiceDiscovery(*args, **kwargs)
    elif service_discovery_type == ServiceDiscoveryType.K8S:
        k8s_discovery_type = kwargs.pop("k8s_service_discovery_type", "pod-ip")
        if k8s_discovery_type is None or not k8s_discovery_type.strip():
            normalized_type = "pod-ip"
        else:
            normalized_type = k8s_discovery_type.strip().lower()

        if normalized_type == "service-name":
            return K8sServiceNameServiceDiscovery(*args, **kwargs)
        else:
            return K8sPodIPServiceDiscovery(*args, **kwargs)
    else:
        raise ValueError("Invalid service discovery type")


def initialize_service_discovery(
    service_discovery_type: ServiceDiscoveryType, *args, **kwargs
) -> ServiceDiscovery:
    """
    Initialize the service discovery module with the given type and arguments.

    Args:
        service_discovery_type: the type of service discovery module
        *args: positional arguments for the service discovery module
        **kwargs: keyword arguments for the service discovery module

    Returns:
        the initialized service discovery module

    Raises:
        ValueError: if the service discovery module is already initialized
        ValueError: if the service discovery type is invalid
    """
    global _global_service_discovery
    if _global_service_discovery is not None:
        raise ValueError("Service discovery module already initialized")

    _global_service_discovery = _create_service_discovery(
        service_discovery_type, *args, **kwargs
    )
    return _global_service_discovery


def reconfigure_service_discovery(
    service_discovery_type: ServiceDiscoveryType, *args, **kwargs
) -> ServiceDiscovery:
    """
    Reconfigure the service discovery module with the given type and arguments.
    """
    global _global_service_discovery
    if _global_service_discovery is None:
        raise ValueError("Service discovery module not initialized")

    new_service_discovery = _create_service_discovery(
        service_discovery_type, *args, **kwargs
    )

    _global_service_discovery.close()
    _global_service_discovery = new_service_discovery
    return _global_service_discovery


def get_service_discovery() -> ServiceDiscovery:
    """
    Get the initialized service discovery module.

    Returns:
        the initialized service discovery module

    Raises:
        ValueError: if the service discovery module is not initialized
    """
    global _global_service_discovery
    if _global_service_discovery is None:
        raise ValueError("Service discovery module not initialized")

    return _global_service_discovery


if __name__ == "__main__":
    # Test the service discovery with event callbacks

    # Define a sample callback function
    def on_engine_event(
        event_type: ServiceDiscoveryEventType,
        engine_name: str,
        endpoint_info: Optional[EndpointInfo],
    ) -> None:
        """Sample callback function to handle service discovery events."""
        print(f"[CALLBACK] Event: {event_type.value}, Engine: {engine_name}")
        if endpoint_info:
            print(f"[CALLBACK] Endpoint URL: {endpoint_info.url}")
            print(f"[CALLBACK] Models: {endpoint_info.model_names}")
            print(f"[CALLBACK] Model Label: {endpoint_info.model_label}")
        print("-" * 50)

    # Define another callback for logging
    def log_engine_event(
        event_type: ServiceDiscoveryEventType,
        engine_name: str,
        endpoint_info: Optional[EndpointInfo],
    ) -> None:
        """Log engine events to a file or monitoring system."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {event_type.value}: {engine_name}"
        print(f"[LOG] {log_msg}")

    # k8s_sd = K8sServiceDiscovery("default", 8000, "release=test")
    initialize_service_discovery(
        ServiceDiscoveryType.K8S,
        namespace="default",
        port=8000,
        label_selector="release=test",
        event_callbacks=[on_engine_event, log_engine_event],
    )

    k8s_sd = get_service_discovery()

    time.sleep(1)
    while True:
        urls = k8s_sd.get_endpoint_info()
        print(f"\n[MAIN] Current endpoints: {len(urls)}")
        for endpoint in urls:
            print(f"  - {endpoint.url} (models: {endpoint.model_names})")
        time.sleep(2)
