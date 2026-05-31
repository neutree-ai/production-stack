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
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

import aiohttp
import requests
from kubernetes import client, config, watch

from vllm_router import utils
from vllm_router.log import init_logger

logger = init_logger(__name__)

_global_service_discovery: "Optional[ServiceDiscovery]" = None
PD_ROUTING_LOGIC = "pd"


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


@dataclass(frozen=True)
class PDTopologyUnit:
    """Schedulable P/D unit returned by a topology source."""

    role: str
    rank: int

    @classmethod
    def from_dict(cls, data: Dict) -> "PDTopologyUnit":
        return cls(role=data.get("role"), rank=int(data.get("rank")))


@dataclass(frozen=True)
class PDTopology:
    """Minimal P/D topology contract consumed by discovery."""

    group_id: str
    units: List[PDTopologyUnit]

    @classmethod
    def from_dict(cls, data: Dict) -> "PDTopology":
        units = [PDTopologyUnit.from_dict(unit) for unit in data.get("units", [])]
        return cls(group_id=data.get("group_id"), units=units)


class TopologyResolver(abc.ABC):
    """Abstract topology source for direct/group discovery targets."""

    @abc.abstractmethod
    def resolve(self, target: str) -> Optional[PDTopology]:
        raise NotImplementedError


class SidecarHTTPTopologyResolver(TopologyResolver):
    """Resolve P/D group topology from a sidecar HTTP endpoint."""

    def __init__(self, timeout_seconds: int):
        self.timeout_seconds = timeout_seconds

    def resolve(self, target: str) -> Optional[PDTopology]:
        try:
            response = requests.get(target, timeout=self.timeout_seconds)
            response.raise_for_status()
            topology = PDTopology.from_dict(response.json())
        except Exception as e:
            logger.warning("Failed to resolve P/D topology from %s: %s", target, e)
            return None

        valid_units = [
            unit
            for unit in topology.units
            if unit.role in {"prefill", "decode"} and unit.rank >= 0
        ]
        if not topology.group_id or not valid_units:
            logger.warning("Invalid P/D topology from %s: %s", target, topology)
            return None
        return PDTopology(group_id=topology.group_id, units=valid_units)


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

    # P/D group and rank metadata. Discovery expands group targets into one
    # EndpointInfo per schedulable P/D unit before the router sees them.
    group_id: Optional[str] = None
    group_uid: Optional[str] = None
    domain: Optional[str] = None
    pd_role: Optional[str] = None
    pd_rank: Optional[int] = None
    route_meta: Dict[str, int] = field(default_factory=dict)

    # Deprecated count-based metadata kept for compatibility with older callers.
    # PDRouter no longer expands these counts on the request path.
    role_group_id: Optional[str] = None
    prefill_count: int = 1
    decode_count: int = 1

    def __str__(self):
        return f"EndpointInfo(url={self.url}, model_names={self.model_names}, added_timestamp={self.added_timestamp}, model_label={self.model_label}, service_name={self.service_name},pod_name={self.pod_name}, namespace={self.namespace}, workspace={self.workspace}, endpoint={self.endpoint}, routing_logic={self.routing_logic}, group_id={self.group_id}, group_uid={self.group_uid}, domain={self.domain}, pd_role={self.pd_role}, pd_rank={self.pd_rank}, route_meta={self.route_meta})"

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
        Initialize aiohttp ClientSession objects for prefill and decode endpoints.
        This must be called from an async context during app startup.
        """
        if (
            self.prefill_model_labels is not None
            and self.decode_model_labels is not None
        ):
            endpoint_infos = self.get_endpoint_info()
            for endpoint_info in endpoint_infos:
                if endpoint_info.model_label in self.prefill_model_labels:
                    self.app.state.prefill_client = aiohttp.ClientSession(
                        base_url=endpoint_info.url,
                        timeout=aiohttp.ClientTimeout(total=None),
                    )
                elif endpoint_info.model_label in self.decode_model_labels:
                    self.app.state.decode_client = aiohttp.ClientSession(
                        base_url=endpoint_info.url,
                        timeout=aiohttp.ClientTimeout(total=None),
                    )

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


class K8sPodIPServiceDiscovery(ServiceDiscovery):
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
        self.topology_resolver = SidecarHTTPTopologyResolver(
            timeout_seconds=health_check_timeout_seconds
        )
        self.event_callbacks = event_callbacks or []

        # Init kubernetes watcher
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        self.k8s_api = client.CoreV1Api()
        self.k8s_watcher = watch.Watch()

        # Start watching engines
        self.running = True
        self.watcher_thread = threading.Thread(target=self._watch_engines, daemon=True)
        self.watcher_thread.start()
        self.prefill_model_labels = prefill_model_labels
        self.decode_model_labels = decode_model_labels

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
                name=pod_name, namespace=self.namespace
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
                name=pod_name, namespace=self.namespace
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
                name=pod_name, namespace=self.namespace
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

    def _get_model_names(self, pod_ip, port: Optional[int] = None) -> List[str]:
        """
        Get the model names of the serving engine pod by querying the pod's
        '/v1/models' endpoint.

        Args:
            pod_ip: the IP address of the pod

        Returns:
            List of model names available on the serving engine, including both base models and adapters
        """
        target_port = port or self.port
        url = f"http://{pod_ip}:{target_port}/v1/models"
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

    def _get_model_info(
        self, pod_ip, port: Optional[int] = None
    ) -> Dict[str, ModelInfo]:
        """
        Get detailed model information from the serving engine pod.

        Args:
            pod_ip: the IP address of the pod

        Returns:
            Dictionary mapping model IDs to their ModelInfo objects, including parent-child relationships
        """
        target_port = port or self.port
        url = f"http://{pod_ip}:{target_port}/v1/models"
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

    @staticmethod
    def _get_metadata_value(pod, keys: List[str]) -> Optional[str]:
        labels = pod.metadata.labels or {}
        annotations = pod.metadata.annotations or {}
        for key in keys:
            if key in labels:
                return labels[key]
            if key in annotations:
                return annotations[key]
        return None

    @staticmethod
    def _parse_nonnegative_int(value: Optional[str], default: int) -> int:
        if value is None:
            return default
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            logger.warning("Invalid P/D metadata integer %s, using %s", value, default)
            return default
        if parsed < 0:
            logger.warning("Negative P/D metadata integer %s, using %s", value, default)
            return default
        return parsed

    @staticmethod
    def _parse_positive_int(value: Optional[str], field_name: str) -> Optional[int]:
        if value is None:
            logger.warning("Missing required P/D metadata field %s", field_name)
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            logger.warning("Invalid P/D metadata field %s=%s", field_name, value)
            return None
        if parsed <= 0:
            logger.warning(
                "P/D metadata field %s must be positive: %s", field_name, value
            )
            return None
        return parsed

    def _get_role_group_id(self, pod) -> str:
        """Get the stable P/D group id from Kubernetes object identity."""
        return pod.metadata.name

    def _get_pd_group_uid(self, pod) -> Optional[str]:
        return getattr(pod.metadata, "uid", None)

    def _get_pd_domain(self, pod) -> str:
        return pod.metadata.name or self._get_pd_group_uid(pod)

    def _get_pd_deployment_type(self, pod) -> Optional[str]:
        return self._get_metadata_value(
            pod,
            [
                "neutree.io/pd-deployment-type",
                "neutree.ai/pd-deployment-type",
                "pd_deployment_type",
                "pd-deployment-type",
            ],
        )

    def _get_pd_sidecar_port(self, pod) -> Optional[int]:
        """
        Get the sidecar port override for a collocated P/D Pod.
        """
        value = self._get_metadata_value(
            pod,
            [
                "neutree.io/pd-sidecar-port",
                "neutree.ai/pd-sidecar-port",
                "pd_sidecar_port",
                "pd-sidecar-port",
            ],
        )
        if value is None:
            logger.warning("Missing required P/D metadata field pd_sidecar_port")
            return None
        return self._parse_positive_int(value, "pd_sidecar_port")

    def _get_pd_metadata(
        self, pod, routing_logic: Optional[str]
    ) -> tuple[Optional[str], Optional[str], Optional[str], Optional[int]]:
        if routing_logic != PD_ROUTING_LOGIC:
            return None, None, None, None

        role_group_id = self._get_role_group_id(pod)
        group_uid = self._get_pd_group_uid(pod)
        domain = self._get_pd_domain(pod)
        pd_sidecar_port = self._get_pd_sidecar_port(pod)
        if self._get_pd_deployment_type(pod) != "group" or pd_sidecar_port is None:
            logger.warning(
                "P/D pod %s is missing valid group routing metadata; marking unavailable",
                pod.metadata.name,
            )
            return role_group_id, group_uid, domain, None

        return role_group_id, group_uid, domain, pd_sidecar_port

    def _get_pd_topology(
        self, pod_ip: str, pd_sidecar_port: Optional[int]
    ) -> Optional[PDTopology]:
        if pd_sidecar_port is None:
            return None
        topology_url = f"http://{pod_ip}:{pd_sidecar_port}/v1/pd/topology"
        resolver = getattr(self, "topology_resolver", None)
        if resolver is None:
            resolver = SidecarHTTPTopologyResolver(self.health_check_timeout_seconds)
        return resolver.resolve(topology_url)

    def _engine_needs_refresh(
        self,
        engine_name: str,
        engine_ip: str,
        model_names: List[str],
        model_label: Optional[str],
        workspace: Optional[str],
        endpoint: Optional[str],
        routing_logic: Optional[str],
        role_group_id: Optional[str],
        group_uid: Optional[str],
        domain: Optional[str],
        pd_sidecar_port: Optional[int],
        pd_topology: Optional[PDTopology],
    ) -> bool:
        target_port = pd_sidecar_port or self.port
        target_url = f"http://{engine_ip}:{target_port}"

        if routing_logic == PD_ROUTING_LOGIC:
            expected = self._build_pd_endpoint_signatures(
                target_url,
                model_names,
                model_label,
                workspace,
                endpoint,
                routing_logic,
                role_group_id,
                group_uid,
                domain,
                pd_topology,
            )
            with self.available_engines_lock:
                current = {
                    self._endpoint_signature(endpoint_info)
                    for endpoint_info in self.available_engines.values()
                    if endpoint_info.pod_name == engine_name
                }
            return current != expected

        with self.available_engines_lock:
            existing = self.available_engines.get(engine_name)

        if existing is None:
            return True

        return (
            existing.url != target_url
            or existing.model_names != model_names
            or existing.model_label != model_label
            or existing.workspace != workspace
            or existing.endpoint != endpoint
            or existing.routing_logic != routing_logic
        )

    @staticmethod
    def _pd_unit_engine_name(engine_name: str, unit: PDTopologyUnit) -> str:
        return f"{engine_name}:{unit.role}:{unit.rank}"

    @staticmethod
    def _route_meta_for_pd_unit(unit: PDTopologyUnit) -> Dict[str, int]:
        return {f"{unit.role}_index": unit.rank}

    def _build_pd_endpoint_info(
        self,
        engine_name: str,
        engine_ip: str,
        model_names: List[str],
        model_label: str,
        workspace: str,
        endpoint: str,
        routing_logic: str,
        role_group_id: str,
        group_uid: Optional[str],
        domain: Optional[str],
        pd_sidecar_port: int,
        model_info: Dict[str, ModelInfo],
        sleep_status: bool,
        unit: PDTopologyUnit,
    ) -> tuple[str, EndpointInfo]:
        unit_engine_name = self._pd_unit_engine_name(engine_name, unit)
        url = f"http://{engine_ip}:{pd_sidecar_port}"
        endpoint_info = EndpointInfo(
            url=url,
            model_names=model_names,
            added_timestamp=int(time.time()),
            Id=str(uuid.uuid5(uuid.NAMESPACE_DNS, unit_engine_name)),
            model_label=model_label,
            sleep=sleep_status,
            pod_name=engine_name,
            namespace=self.namespace,
            model_info=model_info,
            workspace=workspace,
            endpoint=endpoint,
            routing_logic=routing_logic,
            group_id=role_group_id,
            group_uid=group_uid,
            domain=domain,
            pd_role=unit.role,
            pd_rank=unit.rank,
            route_meta=self._route_meta_for_pd_unit(unit),
            role_group_id=role_group_id,
            prefill_count=0,
            decode_count=0,
        )
        return unit_engine_name, endpoint_info

    @staticmethod
    def _endpoint_signature(endpoint_info: EndpointInfo) -> tuple:
        return (
            endpoint_info.url,
            tuple(endpoint_info.model_names),
            endpoint_info.model_label,
            endpoint_info.workspace,
            endpoint_info.endpoint,
            endpoint_info.routing_logic,
            endpoint_info.group_id,
            endpoint_info.group_uid,
            endpoint_info.domain,
            endpoint_info.pd_role,
            endpoint_info.pd_rank,
            tuple(sorted(endpoint_info.route_meta.items())),
        )

    def _build_pd_endpoint_signatures(
        self,
        target_url: str,
        model_names: List[str],
        model_label: Optional[str],
        workspace: Optional[str],
        endpoint: Optional[str],
        routing_logic: Optional[str],
        role_group_id: Optional[str],
        group_uid: Optional[str],
        domain: Optional[str],
        pd_topology: Optional[PDTopology],
    ) -> Set[tuple]:
        if pd_topology is None:
            return set()
        return {
            (
                target_url,
                tuple(model_names),
                model_label,
                workspace,
                endpoint,
                routing_logic,
                role_group_id,
                group_uid,
                domain,
                unit.role,
                unit.rank,
                tuple(sorted(self._route_meta_for_pd_unit(unit).items())),
            )
            for unit in pd_topology.units
        }

    def _watch_engines(self):
        while self.running:
            try:
                for event in self.k8s_watcher.stream(
                    self.k8s_api.list_namespaced_pod,
                    namespace=self.namespace,
                    label_selector=self.label_selector,
                    timeout_seconds=self.watcher_timeout_seconds,
                ):
                    pod = event["object"]
                    event_type = event["type"]
                    pod_name = pod.metadata.name
                    pod_ip = pod.status.pod_ip

                    if event_type == "DELETED":
                        if any(
                            key == pod_name or endpoint_info.pod_name == pod_name
                            for key, endpoint_info in self.available_engines.items()
                        ):
                            self._delete_engine(pod_name)
                        continue

                    # Check if pod is terminating
                    is_pod_terminating = self._is_pod_terminating(pod)
                    is_container_ready = self._check_pod_ready(
                        pod.status.container_statuses
                    )

                    # Pod is ready if container is ready and pod is not terminating
                    is_pod_ready = is_container_ready and not is_pod_terminating

                    role_group_id = None
                    group_uid = None
                    domain = None
                    pd_sidecar_port = None
                    pd_topology = None

                    if is_pod_ready:
                        routing_logic = self._get_routing_logic(pod)
                        (
                            role_group_id,
                            group_uid,
                            domain,
                            pd_sidecar_port,
                        ) = self._get_pd_metadata(pod, routing_logic)
                        if routing_logic == PD_ROUTING_LOGIC:
                            pd_topology = self._get_pd_topology(pod_ip, pd_sidecar_port)
                        if routing_logic == PD_ROUTING_LOGIC and (
                            pd_sidecar_port is None or pd_topology is None
                        ):
                            is_pod_ready = False
                            model_names = []
                        else:
                            model_names = self._get_model_names(pod_ip, pd_sidecar_port)
                    if is_pod_ready:
                        model_label = self._get_model_label(pod)
                        workspace = self._get_workspace(pod)
                        endpoint = self._get_endpoint(pod)
                    else:
                        model_names = []
                        model_label = None
                        workspace = None
                        endpoint = None
                        routing_logic = None
                        pd_topology = None

                    # Record pod status for debugging
                    if is_container_ready and is_pod_terminating:
                        logger.info(
                            f"Pod {pod_name} has ready containers but is terminating - marking as unavailable"
                        )

                    self._on_engine_update(
                        pod_name,
                        pod_ip,
                        event_type,
                        is_pod_ready,
                        model_names,
                        model_label,
                        workspace,
                        endpoint,
                        routing_logic,
                        role_group_id,
                        group_uid,
                        domain,
                        pd_sidecar_port,
                        pd_topology,
                    )
            except Exception as e:
                logger.error(f"K8s watcher error: {e}")
                time.sleep(0.5)

    def _add_engine(
        self,
        engine_name: str,
        engine_ip: str,
        model_names: List[str],
        model_label: str,
        workspace: str,
        endpoint: str,
        routing_logic: str,
        role_group_id: Optional[str],
        group_uid: Optional[str],
        domain: Optional[str],
        pd_sidecar_port: Optional[int],
        pd_topology: Optional[PDTopology],
    ):
        logger.info(
            f"Discovered new serving engine {engine_name} at "
            f"{engine_ip}, running models: {model_names}"
        )

        # Get detailed model information
        model_info = self._get_model_info(engine_ip, pd_sidecar_port)

        # Check if engine is enabled with sleep mode and set engine sleep status
        if self._check_engine_sleep_mode(engine_name):
            sleep_status = self._get_engine_sleep_status(engine_ip)
        else:
            sleep_status = False

        with self.available_engines_lock:
            target_port = pd_sidecar_port or self.port
            if routing_logic == PD_ROUTING_LOGIC and pd_topology is not None:
                endpoint_infos = []
                for unit in pd_topology.units:
                    unit_engine_name, endpoint_info = self._build_pd_endpoint_info(
                        engine_name,
                        engine_ip,
                        model_names,
                        model_label,
                        workspace,
                        endpoint,
                        routing_logic,
                        role_group_id or pd_topology.group_id,
                        group_uid,
                        domain,
                        target_port,
                        model_info,
                        sleep_status,
                        unit,
                    )
                    self.available_engines[unit_engine_name] = endpoint_info
                    endpoint_infos.append((unit_engine_name, endpoint_info))
            else:
                self.available_engines[engine_name] = EndpointInfo(
                    url=f"http://{engine_ip}:{target_port}",
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
                )
                endpoint_infos = [(engine_name, self.available_engines[engine_name])]

        # Trigger callbacks after releasing lock
        for endpoint_name, endpoint_info in endpoint_infos:
            self._trigger_callbacks(
                ServiceDiscoveryEventType.ENGINE_ADDED,
                endpoint_name,
                endpoint_info,
            )

        try:
            fut = asyncio.run_coroutine_threadsafe(
                self.initialize_client_sessions(),
                self.app.state.event_loop,
            )
            fut.result()
        except Exception as e:
            logger.error(f"Error initializing client sessions: {e}")

        # Track all models we've ever seen
        with self.known_models_lock:
            self.known_models.update(model_names)

    def _delete_engine(self, engine_name: str):
        logger.info(f"Serving engine {engine_name} is deleted")
        with self.available_engines_lock:
            delete_items = [
                (key, endpoint_info)
                for key, endpoint_info in self.available_engines.items()
                if key == engine_name or endpoint_info.pod_name == engine_name
            ]
            for key, _ in delete_items:
                del self.available_engines[key]

        # Trigger callbacks after releasing lock
        for endpoint_name, endpoint_info in delete_items:
            self._trigger_callbacks(
                ServiceDiscoveryEventType.ENGINE_DELETED,
                endpoint_name,
                endpoint_info,
            )

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
        role_group_id: Optional[str],
        group_uid: Optional[str],
        domain: Optional[str],
        pd_sidecar_port: Optional[int],
        pd_topology: Optional[PDTopology],
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

            # Only add engine if pod is ready and has models
            if not is_pod_ready or not model_names:
                return

            self._add_engine(
                engine_name,
                engine_ip,
                model_names,
                model_label,
                workspace,
                endpoint,
                routing_logic,
                role_group_id,
                group_uid,
                domain,
                pd_sidecar_port,
                pd_topology,
            )

        elif event == "DELETED":
            if not any(
                key == engine_name or endpoint_info.pod_name == engine_name
                for key, endpoint_info in self.available_engines.items()
            ):
                return

            self._delete_engine(engine_name)

        elif event == "MODIFIED":
            if engine_ip is None:
                return

            # Check if engine availability status changed
            was_available = any(
                key == engine_name or endpoint_info.pod_name == engine_name
                for key, endpoint_info in self.available_engines.items()
            )
            is_now_available = is_pod_ready and model_names

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
                    role_group_id,
                    group_uid,
                    domain,
                    pd_sidecar_port,
                    pd_topology,
                )
            elif is_now_available and was_available:
                if self._engine_needs_refresh(
                    engine_name,
                    engine_ip,
                    model_names,
                    model_label,
                    workspace,
                    endpoint,
                    routing_logic,
                    role_group_id,
                    group_uid,
                    domain,
                    pd_sidecar_port,
                    pd_topology,
                ):
                    self._delete_engine(engine_name)
                    self._add_engine(
                        engine_name,
                        engine_ip,
                        model_names,
                        model_label,
                        workspace,
                        endpoint,
                        routing_logic,
                        role_group_id,
                        group_uid,
                        domain,
                        pd_sidecar_port,
                        pd_topology,
                    )
            elif not is_now_available and was_available:
                # Engine became unavailable: trigger ENGINE_DELETED
                self._delete_engine(engine_name)

    def get_endpoint_info(self) -> List[EndpointInfo]:
        """
        Get the URLs of the serving engines that are available for
        querying.

        Returns:
            a list of engine URLs
        """
        with self.available_engines_lock:
            return list(self.available_engines.values())

    def get_health(self) -> bool:
        """
        Check if the service discovery module is healthy.

        Returns:
            True if the service discovery module is healthy, False otherwise
        """
        return self.watcher_thread.is_alive()

    def close(self):
        """
        Close the service discovery module.
        """
        self.running = False
        self.k8s_watcher.stop()
        self.watcher_thread.join()

    async def initialize_client_sessions(self) -> None:
        """
        Initialize aiohttp ClientSession objects for prefill and decode endpoints.
        This must be called from an async context during app startup.
        """
        if (
            self.prefill_model_labels is not None
            and self.decode_model_labels is not None
        ):
            endpoint_infos = self.get_endpoint_info()
            for endpoint_info in endpoint_infos:
                if endpoint_info.model_label in self.prefill_model_labels:
                    if (
                        hasattr(self.app.state, "prefill_client")
                        and self.app.state.prefill_client is not None
                    ):
                        await self.app.state.prefill_client.close()
                    self.app.state.prefill_client = aiohttp.ClientSession(
                        base_url=endpoint_info.url,
                        timeout=aiohttp.ClientTimeout(total=None),
                    )
                elif endpoint_info.model_label in self.decode_model_labels:
                    if (
                        hasattr(self.app.state, "decode_client")
                        and self.app.state.decode_client is not None
                    ):
                        await self.app.state.decode_client.close()
                    self.app.state.decode_client = aiohttp.ClientSession(
                        base_url=endpoint_info.url,
                        timeout=aiohttp.ClientTimeout(total=None),
                    )

    def has_ever_seen_model(self, model_name: str) -> bool:
        """Check if we've ever seen this model, even if currently scaled to zero."""
        with self.known_models_lock:
            return model_name in self.known_models

    def get_known_models(self) -> Set[str]:
        """Get all models that have ever been discovered."""
        with self.known_models_lock:
            return self.known_models.copy()


class K8sServiceNameServiceDiscovery(ServiceDiscovery):
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
        self.k8s_watcher = watch.Watch()

        # Start watching engines
        self.running = True
        self.watcher_thread = threading.Thread(target=self._watch_engines, daemon=True)
        self.watcher_thread.start()
        self.prefill_model_labels = prefill_model_labels
        self.decode_model_labels = decode_model_labels

    def _check_service_ready(self, service_name, namespace):
        endpoints = self.k8s_api.read_namespaced_endpoints(service_name, namespace)
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
            service = self.k8s_api.read_namespaced_service(service_name, self.namespace)
            if not service.spec.selector:
                return False

            selector = ",".join([f"{k}={v}" for k, v in service.spec.selector.items()])
            pods = self.k8s_api.list_namespaced_pod(
                namespace=self.namespace, label_selector=selector
            )

            if not pods.items:
                logger.warning(
                    f"No pods found for service {service_name} in namespace {self.namespace}"
                )
                return False

            enable_sleep_mode = False
            for container in pods.items[0].spec.containers:
                if container.name == "vllm":
                    for arg in container.command:
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

    def _watch_engines(self):
        while self.running:
            try:
                for event in self.k8s_watcher.stream(
                    self.k8s_api.list_namespaced_service,
                    namespace=self.namespace,
                    label_selector=self.label_selector,
                    timeout_seconds=self.watcher_timeout_seconds,
                ):
                    service = event["object"]
                    event_type = event["type"]
                    if event_type == "DELETED":
                        if service.metadata.name in self.available_engines:
                            self._delete_engine(service.metadata.name)
                        continue
                    service_name = service.metadata.name
                    is_service_ready = self._check_service_ready(
                        service_name, self.namespace
                    )
                    if is_service_ready:
                        model_names = self._get_model_names(service_name)
                        model_label = self._get_model_label(service)
                    else:
                        model_names = []
                        model_label = None
                    self._on_engine_update(
                        service_name,
                        event_type,
                        is_service_ready,
                        model_names,
                        model_label,
                    )
            except Exception as e:
                logger.error(f"K8s watcher error: {e}")
                time.sleep(0.5)

    def _add_engine(self, engine_name: str, model_names: List[str], model_label: str):
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
            )

            # Store model information in the endpoint info
            self.available_engines[engine_name].model_info = model_info

    def _delete_engine(self, engine_name: str):
        logger.info(f"Serving engine {engine_name} is deleted")
        with self.available_engines_lock:
            del self.available_engines[engine_name]

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

    def get_endpoint_info(self) -> List[EndpointInfo]:
        """
        Get the URLs of the serving engines that are available for
        querying.

        Returns:
            a list of engine URLs
        """
        with self.available_engines_lock:
            return list(self.available_engines.values())

    def get_health(self) -> bool:
        """
        Check if the service discovery module is healthy.

        Returns:
            True if the service discovery module is healthy, False otherwise
        """
        return self.watcher_thread.is_alive()

    def close(self):
        """
        Close the service discovery module.
        """
        self.running = False
        self.k8s_watcher.stop()
        self.watcher_thread.join()

    async def initialize_client_sessions(self) -> None:
        """
        Initialize aiohttp ClientSession objects for prefill and decode endpoints.
        This must be called from an async context during app startup.
        """
        if (
            self.prefill_model_labels is not None
            and self.decode_model_labels is not None
        ):
            endpoint_infos = self.get_endpoint_info()
            for endpoint_info in endpoint_infos:
                if endpoint_info.model_label in self.prefill_model_labels:
                    self.app.state.prefill_client = aiohttp.ClientSession(
                        base_url=endpoint_info.url,
                        timeout=aiohttp.ClientTimeout(total=None),
                    )
                elif endpoint_info.model_label in self.decode_model_labels:
                    self.app.state.decode_client = aiohttp.ClientSession(
                        base_url=endpoint_info.url,
                        timeout=aiohttp.ClientTimeout(total=None),
                    )


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
