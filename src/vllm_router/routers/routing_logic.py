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
import bisect
import concurrent.futures
import enum
import hashlib
import json
import math
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import requests
from fastapi import Request

try:
    from transformers import AutoTokenizer
except ImportError:
    pass

try:
    from lmcache.v1.cache_controller import controller_manager
    from lmcache.v1.cache_controller.message import (
        LookupMsg,
        QueryInstMsg,
    )
except ImportError:
    pass
from uhashring import HashRing

from vllm_router.log import init_logger
from vllm_router.service_discovery import (
    EndpointInfo,
    ServiceDiscoveryEventType,
    get_service_discovery,
)
from vllm_router.stats.engine_stats import EngineStats
from vllm_router.stats.request_stats import RequestStats
from vllm_router.utils import SingletonABCMeta

logger = init_logger(__name__)


class RoutingLogic(str, enum.Enum):
    ROUND_ROBIN = "roundrobin"
    SESSION_BASED = "session"
    KVAWARE = "kvaware"
    PREFIXAWARE = "prefixaware"
    DISAGGREGATED_PREFILL = "disaggregated_prefill"
    PD = "pd"
    CONSISTENT_HASH = "consistent_hash"
    STATIC_HASH = "static_hash"


ROUTING_LOGIC_TO_CLASS = {
    RoutingLogic.ROUND_ROBIN: "RoundRobinRouter",
    RoutingLogic.SESSION_BASED: "SessionRouter",
    RoutingLogic.KVAWARE: "KvawareRouter",
    RoutingLogic.PREFIXAWARE: "PrefixAwareRouter",
    RoutingLogic.DISAGGREGATED_PREFILL: "DisaggregatedPrefillRouter",
    RoutingLogic.PD: "PDRouter",
    RoutingLogic.CONSISTENT_HASH: "ConsistentHashRouter",
    RoutingLogic.STATIC_HASH: "StaticHashRouter",
}

DefaultInitRoutingLogics = [
    RoutingLogic.ROUND_ROBIN,
    RoutingLogic.SESSION_BASED,
    RoutingLogic.PREFIXAWARE,
    RoutingLogic.DISAGGREGATED_PREFILL,
    RoutingLogic.PD,
    RoutingLogic.CONSISTENT_HASH,
    RoutingLogic.STATIC_HASH,
]

AllInitRoutingLogics = DefaultInitRoutingLogics + [
    RoutingLogic.KVAWARE,
]


class RoutingInterface(metaclass=SingletonABCMeta):

    def _qps_routing(
        self, endpoints: List[EndpointInfo], request_stats: Dict[str, RequestStats]
    ) -> str:
        """
        Route the request to the appropriate engine URL based on the QPS of
        each engine

        Args:
            endpoints (List[EndpointInfo]): The list of engine URLs
            request_stats (Dict[str, RequestStats]): The request stats
                indicating the request-level performance of each engine
        """
        lowest_qps = float("inf")
        ret = None
        for info in endpoints:
            url = info.url
            if url not in request_stats:
                return url  # This engine does not have any requests
            request_stat = request_stats[url]
            if request_stat.qps < lowest_qps:
                lowest_qps = request_stat.qps
                ret = url
        return ret

    def _update_hash_ring(self, endpoints: List["EndpointInfo"]):
        """
        Update the hash ring with the current list of endpoints.
        """
        # Extract endpoint URLs
        endpoint_urls = [endpoint.url for endpoint in endpoints]

        # Get the current nodes in the hash ring
        current_nodes = set(self.hash_ring.get_nodes())

        # Convert the new endpoint URLs to a set for easy comparison
        new_nodes = set(endpoint_urls)

        # Remove nodes that are no longer in the list
        for node in current_nodes - new_nodes:
            self.hash_ring.remove_node(node)

        # Add new nodes that are not already in the hash ring
        for node in new_nodes - current_nodes:
            self.hash_ring.add_node(node)

    def extract_session_id(self, request: Request, request_json: Dict) -> Optional[str]:
        """
        Extract the session id from the request headers or request body.
        """
        session_key = getattr(self, "session_key", None)
        if session_key is None:
            return None
        val = request.headers.get(session_key)
        return val if val is not None else request_json.get(session_key, None)

    @abc.abstractmethod
    def route_request(
        self,
        endpoints: List[EndpointInfo],
        engine_stats: Dict[str, EngineStats],
        request_stats: Dict[str, RequestStats],
        request: Request,
    ) -> str:
        """
        Route the request to the appropriate engine URL

        Args:
            endpoints (List[EndpointInfo]): The list of engine URLs
            engine_stats (Dict[str, EngineStats]): The engine stats indicating
                the 'physical' load of each engine
            request_stats (Dict[str, RequestStats]): The request stats
                indicating the request-level performance of each engine
            request (Request): The incoming request
        """
        raise NotImplementedError


class RoundRobinRouter(RoutingInterface):
    # TODO (ApostaC): when available engines in the endpoints changes, the
    # algorithm may not be "perfectly" round-robin.
    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self.req_id = 0
        self.sorted_endpoints = []
        self.last_endpoints_id = None
        self.last_endpoints_hash = None
        self._initialized = True

    def route_request(
        self,
        endpoints: List[EndpointInfo],
        engine_stats: Dict[str, EngineStats],
        request_stats: Dict[str, RequestStats],
        request: Request,
    ) -> str:
        """
        Route the request to the appropriate engine URL using a simple
        round-robin algorithm

        Args:
            endpoints (List[EndpointInfo]): The list of engine URLs
            engine_stats (Dict[str, EngineStats]): The engine stats indicating
                the 'physical' load of each engine
            request_stats (Dict[str, RequestStats]): The request stats
                indicating the request-level performance of each engine
            request (Request): The incoming request
        """
        endpoints_id = id(endpoints)
        if endpoints_id != self.last_endpoints_id:
            current_hash = hash(tuple(e.url for e in endpoints))
            if current_hash != self.last_endpoints_hash:
                self.sorted_endpoints = sorted(endpoints, key=lambda e: e.url)
                self.last_endpoints_hash = current_hash
            self.last_endpoints_id = endpoints_id
        chosen = self.sorted_endpoints[self.req_id % len(self.sorted_endpoints)]
        self.req_id += 1
        return chosen.url


class SessionRouter(RoutingInterface):
    """
    Route the request to the appropriate engine URL based on the session key
    in the request headers
    """

    def __init__(self, session_key: str = None):
        if hasattr(self, "_initialized"):
            return
        if session_key is None:
            raise ValueError("SessionRouter must be initialized with a session_key")
        self.session_key = session_key
        self.hash_ring = HashRing()
        self._initialized = True

    async def route_request(
        self,
        endpoints: List[EndpointInfo],
        engine_stats: Dict[str, EngineStats],
        request_stats: Dict[str, RequestStats],
        request: Request,
        request_json: Dict,
    ) -> str:
        """
        Route the request to the appropriate engine URL by the 'session id' in
        the request headers or request body.
        If there is no session id in the request header or request body, it will pick a server
        with lowest qps

        Args:
            endpoints (List[EndpointInfo]): The list of engine URLs
            engine_stats (Dict[str, EngineStats]): The engine stats indicating
                the 'physical' load of each engine
            request_stats (Dict[str, RequestStats]): The request stats
                indicating the request-level performance of each engine
            request (Request): The incoming request
            request_json (Dict): The request body (needed for finding the session id)
        """
        session_id = self.extract_session_id(request, request_json)
        logger.debug(f"Got session id: {session_id}")

        # Update the hash ring with the current list of endpoints
        self._update_hash_ring(endpoints)

        if session_id is None:
            # Route based on QPS if no session ID is present
            url = self._qps_routing(endpoints, request_stats)
        else:
            # Use the hash ring to get the endpoint for the session ID
            url = self.hash_ring.get_node(session_id)

        return url


class KvawareRouter(RoutingInterface):
    """
    Route the request to the appropriate engine URL by where the KV cache
    of the longest prefix match is found.
    """

    def __init__(
        self,
        lmcache_controller_port: int,
        session_key: str,
        kv_aware_threshold: int = 2000,
    ):
        self.lmcache_controller_port = lmcache_controller_port
        logger.info(
            f"Initializing KvawareRouter with port: {self.lmcache_controller_port}"
        )
        self.kv_manager = controller_manager.LMCacheControllerManager(
            {
                "pull": f"0.0.0.0:{self.lmcache_controller_port}",
                "reply": None,
            }
        )
        self.req_id = 0
        self.instance_id_to_ip = {}
        self.session_key = session_key
        self.hash_ring = HashRing()
        self.tokenizer = None
        self.threshold = kv_aware_threshold

    def start_kv_manager(self):
        """
        Start the kv manager
        """
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self.lmcache_cluster_monitor_task = asyncio.run_coroutine_threadsafe(
            self.kv_manager.start_all(), self.loop
        )

    def query_manager(self, msg) -> str:
        """
        Get the instance id for the given message
        """
        instance_id = self.kv_manager.handle_orchestration_message(msg)
        return instance_id

    def close(self):
        """Gracefully shutdown the lmcache cluster monitor task."""
        if (
            hasattr(self, "lmcache_cluster_monitor_task")
            and self.lmcache_cluster_monitor_task
        ):
            logger.info("Shutting down lmcache cluster monitor task")
            self.lmcache_cluster_monitor_task.cancel()
            try:
                self.lmcache_cluster_monitor_task.result()
            except concurrent.futures.CancelledError:
                pass
            self.lmcache_cluster_monitor_task = None

    async def route_request(
        self,
        endpoints: List[EndpointInfo],
        engine_stats: Dict[str, EngineStats],
        request_stats: Dict[str, RequestStats],
        request: Request,
        request_json: Dict,
    ) -> str:
        """
        Route the request to the appropriate engine URL by where the KV cache
        of the longest prefix match is found.
        If there is no session id in the request header, it will pick a server
        with round robin.

        Args:
            endpoints (List[EndpointInfo]): The list of engine URLs
            engine_stats (Dict[str, EngineStats]): The engine stats indicating
               the 'physical' load of each engine
            request_stats (Dict[str, RequestStats]): The request stats
               indicating the request-level performance of each engine
            request (Request): The incoming request
            request_json (Dict): The request body (needed for finding the
            longest prefix match)
        """
        token_ids = None
        # Local-first tokenization, fall back to remote "/tokenize" API on failure
        # TODO (Yuhan): Handle chat completions
        try:
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    endpoints[0].model_names[0]
                )
            token_ids = self.tokenizer.encode(request_json.get("prompt", ""))
        except Exception:
            # Remote /tokenize fallback (let errors bubble up to keep behavior simple)
            remote_url = endpoints[0].url + "/tokenize"
            headers = {"Content-Type": "application/json"}
            data = {
                "model": endpoints[0].model_names[0],
                "prompt": request_json.get("prompt", ""),
            }
            body = requests.post(
                remote_url, headers=headers, json=data, timeout=10
            ).json()
            token_ids = body["tokens"]

        event_id = "Lookup" + str(uuid.uuid4())
        msg = LookupMsg(tokens=token_ids, event_id=event_id)
        instance_id = await self.query_manager(msg)
        matched_tokens = math.inf
        logger.debug(f"Lookup return message: {instance_id}")
        if len(list(instance_id.layout_info.keys())) > 0:
            matched_instance_id = list(instance_id.layout_info.keys())[
                0
            ]  # Get the first key
            matched_tokens = instance_id.layout_info[matched_instance_id][1]

        if (
            instance_id is None
            or len(instance_id.layout_info) == 0
            or matched_tokens < max(len(token_ids) - self.threshold, 0)
        ):
            session_id = self.extract_session_id(request, request_json)
            logger.debug(f"Fallback to using session id: {session_id}")
            # Update the hash ring with the current list of endpoints
            self._update_hash_ring(endpoints)
            if session_id is None:
                # Route based on QPS if no session ID is present
                url = self._qps_routing(endpoints, request_stats)
            else:
                # Use the hash ring to get the endpoint for the session ID
                url = self.hash_ring.get_node(session_id)
            return url
        else:
            queried_instance_ids = [info for info in instance_id.layout_info]
            if queried_instance_ids[0] not in self.instance_id_to_ip:
                for endpoint in endpoints:
                    event_id = "QueryInst" + str(uuid.uuid4())
                    query_ip = endpoint.url.split(f":{endpoint.url.split(':')[-1]}")[
                        0
                    ].split("//")[1]
                    query_message = QueryInstMsg(
                        ip=query_ip,
                        event_id=event_id,
                    )
                    endpoint_instance_id = await self.query_manager(query_message)
                    logger.debug(
                        f"Query ip: {query_ip}, return instance id: {endpoint_instance_id}"
                    )
                    self.instance_id_to_ip[endpoint_instance_id.instance_id] = (
                        endpoint.url
                    )
                logger.info(f"Instance id to ip mapping: {self.instance_id_to_ip}")
            logger.info(
                f"Routing request to {queried_instance_ids[0]} found by kvaware router"
            )
            return self.instance_id_to_ip[queried_instance_ids[0]]


class PrefixAwareRouter(RoutingInterface):
    """
    Route the request to the appropriate engine URL by where the longest
    prefix match is found.

    In this class, we assume that there is no eviction of prefix cache.
    """

    def __init__(self: int):
        if hasattr(self, "_initialized"):
            return
        from vllm_router.prefix.hashtrie import HashTrie

        self.hashtrie = HashTrie()
        self._initialized = True

    async def route_request(
        self,
        endpoints: List[EndpointInfo],
        engine_stats: Dict[str, EngineStats],
        request_stats: Dict[str, RequestStats],
        request: Request,
        request_json: Dict,
    ) -> str:
        """
        Route the request to the appropriate engine URL by where the longest
        prefix match is found.

        In this routing logic, we do not consider the eviction of prefix cache.

        Args:
            endpoints (List[EndpointInfo]): The list of engine URLs
            engine_stats (Dict[str, EngineStats]): The engine stats indicating
               the 'physical' load of each engine
            request_stats (Dict[str, RequestStats]): The request stats
               indicating the request-level performance of each engine
            request (Request): The incoming request
            request_json (Dict): The request body (needed for finding the
            longest prefix match)
        """

        # Handle chat completions
        if "messages" in request_json:
            # Get the last message from the messages array
            messages = request_json["messages"]
            if messages:
                # Concatenate all message content
                prompt_parts = []
                for message in messages:
                    content = message.get("content", "")
                    if isinstance(content, list):
                        # Handle multimodal messages
                        text_content = " ".join(
                            part.get("text", "")
                            for part in content
                            if part.get("type") == "text"
                        )
                        prompt_parts.append(text_content)
                    elif content is not None:
                        prompt_parts.append(content)
                prompt = "\n".join(prompt_parts)
            else:
                prompt = ""
        else:
            # Handle regular completions
            prompt = request_json["prompt"]

        available_endpoints = set(endpoint.url for endpoint in endpoints)
        _, matched_endpoint = await self.hashtrie.longest_prefix_match(
            prompt, available_endpoints
        )

        selected_endpoint = random.choice(list(matched_endpoint))

        await self.hashtrie.insert(prompt, selected_endpoint)

        return selected_endpoint


class DisaggregatedPrefillRouter(RoutingInterface):
    """
    Route the request to the appropriate engine URL by handling prefill and decode operations sequentially.
    First request goes to prefill endpoint, then second request goes to decode endpoint.
    """

    def __init__(self, prefill_model_labels: List[str], decode_model_labels: List[str]):
        self.prefill_model_labels = prefill_model_labels
        self.decode_model_labels = decode_model_labels
        self.request_cache = {}  # Cache to store prefill results

    def route_request(
        self,
        endpoints: List[EndpointInfo],
        engine_stats: Dict[str, EngineStats],
        request_stats: Dict[str, RequestStats],
        request: Request,
        request_json: Dict,
    ) -> str:
        """
        Route the request to appropriate endpoints for prefill and decode operations.
        First request goes to prefill endpoint, then second request goes to decode endpoint.
        """
        # Find prefill and decode endpoints
        is_prefill = request_json.get("max_tokens", 0) == 1
        if is_prefill:
            logger.info("Prefill request")
        else:
            logger.info("Decode request")

        # Find endpoints with matching model labels
        prefiller_endpoints = [
            e for e in endpoints if e.model_label in self.prefill_model_labels
        ]
        decoder_endpoints = [
            e for e in endpoints if e.model_label in self.decode_model_labels
        ]
        if is_prefill:
            return prefiller_endpoints[0].url
        else:
            return decoder_endpoints[0].url


@dataclass(frozen=True)
class RouteUnit:
    """Logical P/D route target inside a P/D domain."""

    domain: str
    role: str
    rank: int
    url: str
    endpoint_info: EndpointInfo

    @property
    def unit_id(self) -> str:
        return f"{self.domain}:{self.role}:{self.rank}:{self.url}"


@dataclass(frozen=True)
class PDRouteDecision:
    """Final P/D route decision for a request."""

    prefill: RouteUnit
    decode: RouteUnit

    @property
    def url(self) -> str:
        return self.decode.url

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "X-Neutree-PD-Role-Group": self.decode.domain,
            "X-Neutree-PD-Prefill-Index": str(self.prefill.rank),
            "X-Neutree-PD-Decode-Index": str(self.decode.rank),
        }

    @property
    def stats_metadata(self) -> Dict[str, str]:
        return {
            "pd_prefill_unit_id": self.prefill.unit_id,
            "pd_decode_unit_id": self.decode.unit_id,
        }


@dataclass
class PDRouterState:
    """Encapsulates P/D route units for a workspace+endpoint combination."""

    hash_to_decode_unit_id: Dict[int, str] = field(default_factory=dict)
    sorted_hashes: List[int] = field(default_factory=list)
    hash_to_prefill_unit_id_by_domain: Dict[str, Dict[int, str]] = field(
        default_factory=dict
    )
    prefill_sorted_hashes_by_domain: Dict[str, List[int]] = field(default_factory=dict)
    decode_units: Dict[str, RouteUnit] = field(default_factory=dict)
    prefill_units_by_domain: Dict[str, Dict[str, RouteUnit]] = field(
        default_factory=dict
    )
    last_sync_time: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


class PDRouter(RoutingInterface):
    """
    Route collocated P/D requests by selecting decode first, then prefill in the same domain.

    Discovery expands direct/group targets into one EndpointInfo per schedulable
    P/D unit. The router only consumes those unit endpoints and sends the
    selected unit ranks to the group entrypoint via X-Neutree-PD-* headers.
    """

    def __init__(
        self,
        virtual_nodes_per_replica: int = 100,
        load_factor: float = 1.25,
        max_user_messages_for_cache: int = 2,
    ):
        if hasattr(self, "_initialized"):
            return

        self._virtual_nodes = virtual_nodes_per_replica
        self._load_factor = load_factor
        self._max_user_messages_for_cache = max_user_messages_for_cache
        self._states: Dict[str, PDRouterState] = {}
        self._states_creation_lock = threading.Lock()
        self._service_discovery = None
        self._event_sync_enabled = False
        self._register_service_discovery_callback()
        self._initialized = True

        logger.info(
            "Initialized PDRouter with %s virtual nodes per P/D unit, "
            "load factor %s, max_user_messages_for_cache=%s",
            virtual_nodes_per_replica,
            load_factor,
            max_user_messages_for_cache,
        )

    def close(self) -> None:
        if self._service_discovery is None:
            return
        try:
            self._service_discovery.unregister_callback(
                self._on_service_discovery_event
            )
        except Exception as e:
            logger.warning("PDRouter: Could not unregister service callback: %s", e)
        self._service_discovery = None

    def _register_service_discovery_callback(self) -> None:
        try:
            sd = get_service_discovery()
            if hasattr(sd, "register_callback"):
                sd.register_callback(self._on_service_discovery_event)
                self._service_discovery = sd
                self._event_sync_enabled = True
                logger.info("Registered PDRouter callback with service discovery")
            else:
                logger.warning(
                    "Service discovery does not support callbacks. "
                    "P/D route units will be reconciled from request endpoints."
                )
        except Exception as e:
            logger.warning(
                "Could not register PDRouter service discovery callback: %s. "
                "P/D route units will be reconciled from request endpoints.",
                e,
            )

    def _get_routing_key(self, endpoint_info: EndpointInfo) -> str:
        workspace = endpoint_info.workspace
        endpoint = endpoint_info.endpoint
        return f"{workspace}:{endpoint}"

    def _get_or_create_state(self, routing_key: str) -> PDRouterState:
        state = self._states.get(routing_key)
        if state is not None:
            return state

        with self._states_creation_lock:
            state = self._states.get(routing_key)
            if state is None:
                state = PDRouterState()
                self._states[routing_key] = state
                logger.debug("Created new P/D router state for %s", routing_key)
            return state

    def _hash(self, key: str) -> int:
        hash_obj = hashlib.md5(key.encode())
        return int(hash_obj.hexdigest()[:16], 16)

    def _search(self, sorted_hashes: List[int], key_hash: int) -> Tuple[int, int]:
        if not sorted_hashes:
            raise ValueError("P/D hash ring is empty")

        idx = bisect.bisect_left(sorted_hashes, key_hash)
        if idx >= len(sorted_hashes):
            idx = 0

        return sorted_hashes[idx], idx

    def _endpoint_domain(self, endpoint: EndpointInfo) -> str:
        return endpoint.domain or endpoint.pod_name or endpoint.Id or endpoint.url

    def _endpoint_role(self, endpoint: EndpointInfo) -> Optional[str]:
        role = getattr(endpoint, "role", None)
        return role if role in {"prefill", "decode"} else None

    def _endpoint_rank(self, endpoint: EndpointInfo) -> Optional[int]:
        rank = getattr(endpoint, "rank", None)
        try:
            parsed = int(rank)
        except (TypeError, ValueError):
            return None
        return parsed if parsed >= 0 else None

    def _route_unit_from_endpoint(self, endpoint: EndpointInfo) -> Optional[RouteUnit]:
        role = self._endpoint_role(endpoint)
        rank = self._endpoint_rank(endpoint)
        if role is None or rank is None:
            logger.debug(
                "PDRouter: Skipping non-expanded P/D endpoint %s role=%s rank=%s",
                endpoint.url,
                role,
                rank,
            )
            return None

        return RouteUnit(
            domain=self._endpoint_domain(endpoint),
            role=role,
            rank=rank,
            url=endpoint.url,
            endpoint_info=endpoint,
        )

    def _state_endpoint_infos(self, state: PDRouterState) -> List[EndpointInfo]:
        endpoints = [unit.endpoint_info for unit in state.decode_units.values()]
        for units_by_id in state.prefill_units_by_domain.values():
            endpoints.extend(unit.endpoint_info for unit in units_by_id.values())
        return endpoints

    def _add_decode_unit_to_ring(self, state: PDRouterState, unit: RouteUnit) -> None:
        state.decode_units[unit.unit_id] = unit

        for i in range(self._virtual_nodes):
            virtual_node_key = f"{unit.unit_id}:{i}"
            hash_val = self._hash(virtual_node_key)
            state.hash_to_decode_unit_id[hash_val] = unit.unit_id
            bisect.insort(state.sorted_hashes, hash_val)

    def _add_prefill_unit_to_ring(self, state: PDRouterState, unit: RouteUnit) -> None:
        state.prefill_units_by_domain.setdefault(unit.domain, {})[unit.unit_id] = unit

        hash_to_unit_id = state.hash_to_prefill_unit_id_by_domain.setdefault(
            unit.domain, {}
        )
        sorted_hashes = state.prefill_sorted_hashes_by_domain.setdefault(
            unit.domain, []
        )
        for i in range(self._virtual_nodes):
            virtual_node_key = f"{unit.unit_id}:{i}"
            hash_val = self._hash(virtual_node_key)
            hash_to_unit_id[hash_val] = unit.unit_id
            bisect.insort(sorted_hashes, hash_val)

    def _sync_route_units(
        self, routing_key: str, state: PDRouterState, endpoints: List[EndpointInfo]
    ) -> None:
        state.hash_to_decode_unit_id.clear()
        state.sorted_hashes.clear()
        state.hash_to_prefill_unit_id_by_domain.clear()
        state.prefill_sorted_hashes_by_domain.clear()
        state.decode_units.clear()
        state.prefill_units_by_domain.clear()

        for endpoint in sorted(endpoints, key=lambda e: e.url):
            unit = self._route_unit_from_endpoint(endpoint)
            if unit is None:
                continue

            if unit.role == "prefill":
                self._add_prefill_unit_to_ring(state, unit)
            elif unit.role == "decode":
                self._add_decode_unit_to_ring(
                    state,
                    unit,
                )

        state.last_sync_time = time.time()
        logger.debug(
            "PDRouter: Synced %s decode units and %s domains for %s",
            len(state.decode_units),
            len(state.prefill_units_by_domain),
            routing_key,
        )

    def _extract_cache_key(self, request_json: Dict, request_id: str) -> str:
        if not request_json:
            return request_id

        try:
            cache_components = []

            if "messages" in request_json:
                messages = request_json.get("messages", [])
                system_prompt = None
                user_messages = []
                for msg in messages:
                    if not isinstance(msg, dict):
                        continue
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "system":
                        system_prompt = content
                    elif role == "user":
                        user_messages.append(content)
                        if len(user_messages) >= self._max_user_messages_for_cache:
                            break

                if system_prompt:
                    cache_components.append(
                        f"system:{json.dumps(system_prompt, sort_keys=True)}"
                    )
                for i, user_message in enumerate(user_messages):
                    cache_components.append(
                        f"user_{i}:{json.dumps(user_message, sort_keys=True)}"
                    )

            elif "prompt" in request_json:
                cache_components.append(
                    f"prompt:{json.dumps(request_json.get('prompt', ''), sort_keys=True)}"
                )

            return "|".join(cache_components) if cache_components else request_id
        except Exception as e:
            logger.warning("PDRouter: Error extracting cache key: %s", e)
            return request_id

    def _get_unit_load(self, state: PDRouterState, unit: RouteUnit) -> Optional[float]:
        try:
            from vllm_router.stats.request_stats import get_request_stats_monitor

            monitor = get_request_stats_monitor()
            active = monitor.get_active_pd_unit_request_count(unit.unit_id)
        except Exception as e:
            logger.warning("PDRouter: Could not get load for %s: %s", unit.unit_id, e)
            return None
        return active

    def _get_total_unit_load(
        self, state: PDRouterState, units: List[RouteUnit]
    ) -> float:
        total = 0.0
        for unit in units:
            load = self._get_unit_load(state, unit)
            if load is not None:
                total += load
        return total

    def _check_load(
        self, state: PDRouterState, unit: RouteUnit, candidate_units: List[RouteUnit]
    ) -> bool:
        load = self._get_unit_load(state, unit)
        if load is None:
            return True

        num_units = len(candidate_units)
        if num_units == 0:
            return True

        avg_load = (self._get_total_unit_load(state, candidate_units) + 1) / num_units
        threshold = avg_load * self._load_factor
        return (load + 1) <= threshold

    def _select_unit_from_ring(
        self,
        state: PDRouterState,
        payload_hash: int,
        units_by_id: Dict[str, RouteUnit],
        hash_to_unit_id: Dict[int, str],
        sorted_hashes: List[int],
    ) -> Optional[RouteUnit]:
        if not units_by_id or not sorted_hashes:
            return None

        candidate_units = list(units_by_id.values())
        _, initial_idx = self._search(sorted_hashes, payload_hash)
        checked_unit_ids: Set[str] = set()
        default_unit = None
        current_idx = initial_idx

        while len(checked_unit_ids) < len(units_by_id):
            current_hash = sorted_hashes[current_idx]
            current_unit_id = hash_to_unit_id[current_hash]
            current_unit = units_by_id[current_unit_id]

            if current_unit_id in checked_unit_ids:
                current_idx = (current_idx + 1) % len(sorted_hashes)
                continue

            checked_unit_ids.add(current_unit_id)
            if default_unit is None:
                default_unit = current_unit

            if self._check_load(state, current_unit, candidate_units):
                return current_unit

            current_idx = (current_idx + 1) % len(sorted_hashes)

        return default_unit

    def _select_decode_unit(
        self, state: PDRouterState, payload_hash: int
    ) -> Optional[RouteUnit]:
        return self._select_unit_from_ring(
            state,
            payload_hash,
            state.decode_units,
            state.hash_to_decode_unit_id,
            state.sorted_hashes,
        )

    def _select_prefill_unit(
        self, state: PDRouterState, decode_unit: RouteUnit, payload_hash: int
    ) -> Optional[RouteUnit]:
        prefill_units = state.prefill_units_by_domain.get(decode_unit.domain, {})
        return self._select_unit_from_ring(
            state,
            payload_hash,
            prefill_units,
            state.hash_to_prefill_unit_id_by_domain.get(decode_unit.domain, {}),
            state.prefill_sorted_hashes_by_domain.get(decode_unit.domain, []),
        )

    def _on_service_discovery_event(
        self,
        event_type: ServiceDiscoveryEventType,
        engine_name: str,
        endpoint_info: Optional[EndpointInfo],
    ) -> None:
        self._event_sync_enabled = True
        if endpoint_info is None:
            return
        if endpoint_info.routing_logic and endpoint_info.routing_logic != "pd":
            return

        event_unit = self._route_unit_from_endpoint(endpoint_info)
        if event_unit is None:
            return

        routing_key = self._get_routing_key(endpoint_info)
        state = self._get_or_create_state(routing_key)
        with state.lock:
            endpoints = self._state_endpoint_infos(state)
            endpoints = [
                endpoint
                for endpoint in endpoints
                if (unit := self._route_unit_from_endpoint(endpoint)) is not None
                and unit.unit_id != event_unit.unit_id
            ]
            if event_type == ServiceDiscoveryEventType.ENGINE_ADDED:
                endpoints.append(endpoint_info)
            elif event_type != ServiceDiscoveryEventType.ENGINE_DELETED:
                return

            self._sync_route_units(routing_key, state, endpoints)
            logger.debug(
                "PDRouter: Applied %s for %s on %s",
                event_type.value,
                engine_name,
                event_unit.unit_id,
            )

    async def route_request(
        self,
        endpoints: List[EndpointInfo],
        engine_stats: Dict[str, EngineStats],
        request_stats: Dict[str, RequestStats],
        request: Request,
        request_json: Optional[Dict] = None,
    ) -> Optional[PDRouteDecision]:
        if not endpoints:
            logger.error("PDRouter: No endpoints available for routing")
            return None

        routing_key = self._get_routing_key(endpoints[0])
        state = self._get_or_create_state(routing_key)

        with state.lock:
            if (
                not self._event_sync_enabled
                or not state.decode_units
                or not state.prefill_units_by_domain
            ):
                self._sync_route_units(routing_key, state, endpoints)

            if request_json is None:
                try:
                    request_json = await request.json()
                except Exception as e:
                    logger.warning("PDRouter: Could not parse request JSON: %s", e)
                    request_json = {}

            request_id = str(uuid.uuid4())
            cache_key = self._extract_cache_key(request_json, request_id)
            payload_hash = self._hash(cache_key)

            decode_unit = self._select_decode_unit(state, payload_hash)
            if decode_unit is None:
                logger.error("PDRouter: No ready decode unit for %s", routing_key)
                return None

            prefill_unit = self._select_prefill_unit(state, decode_unit, payload_hash)
            if prefill_unit is None:
                logger.error(
                    "PDRouter: No ready prefill unit in domain %s for %s",
                    decode_unit.domain,
                    routing_key,
                )
                return None

            logger.info(
                "PDRouter: Selected domain=%s prefill=%s decode=%s url=%s",
                decode_unit.domain,
                prefill_unit.rank,
                decode_unit.rank,
                decode_unit.url,
            )
            return PDRouteDecision(prefill=prefill_unit, decode=decode_unit)


@dataclass
class HashRingState:
    """Encapsulates hash ring state for a specific workspace+endpoint combination."""

    hash_to_endpoint_url: Dict[int, str] = field(default_factory=dict)
    sorted_hashes: List[int] = field(default_factory=list)
    available_replicas: Dict[str, EndpointInfo] = field(default_factory=dict)
    last_sync_time: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


class ConsistentHashRouter(RoutingInterface):
    """
    Route requests using Consistent Hashing with Bounded Loads (CHWBL).

    This router ensures that similar payloads are routed to the same replica
    while maintaining load balance and minimizing disruption when replicas are
    added or removed.

    Features:
    - Consistent hashing with virtual nodes for better load distribution
    - Bounded load checking to prevent overloading any single replica
    - Cache key extraction from chat completions (system prompt + user messages)
    - Integration with K8sPodIPServiceDiscovery via callbacks
    - Load tracking via RequestStatsMonitor
    - Multi-tenant support with separate hash rings per workspace+endpoint
    """

    def __init__(
        self,
        virtual_nodes_per_replica: int = 100,
        load_factor: float = 1.25,
        max_user_messages_for_cache: int = 2,
    ):
        """
        Initialize the Consistent Hash Router.

        Args:
            virtual_nodes_per_replica: Number of virtual nodes per replica on the hash ring
            load_factor: Maximum load factor (e.g., 1.25 means 125% of average load)
            max_user_messages_for_cache: Max number of user messages to include in cache key
        """
        if hasattr(self, "_initialized"):
            return

        # Consistent hashing settings
        self._virtual_nodes = virtual_nodes_per_replica
        self._load_factor = load_factor
        self._max_user_messages_for_cache = max_user_messages_for_cache

        # Hash ring data structures - separate ring per workspace+endpoint
        self._hash_rings: Dict[str, HashRingState] = {}
        # Global lock only for creating new routing_key entries
        self._hash_rings_creation_lock = threading.Lock()

        # Register callback with service discovery
        self._register_service_discovery_callback()

        logger.info(
            f"Initialized ConsistentHashRouter with "
            f"{virtual_nodes_per_replica} virtual nodes per replica, "
            f"load factor of {load_factor}, "
            f"max_user_messages_for_cache={max_user_messages_for_cache}"
        )

        self._initialized = True

    def _register_service_discovery_callback(self):
        """Register callback with K8sPodIPServiceDiscovery to track replica changes."""

        try:
            sd = get_service_discovery()

            # Check if service discovery supports callbacks (K8sPodIPServiceDiscovery)
            if hasattr(sd, "register_callback"):
                sd.register_callback(self._on_service_discovery_event)
                logger.info(
                    "Registered ConsistentHashRouter callback with service discovery"
                )
            else:
                logger.warning(
                    "Service discovery does not support callbacks. "
                    "Replica tracking will be done via route_request updates."
                )
        except Exception as e:
            logger.warning(
                f"Could not register service discovery callback: {e}. "
                f"Replica tracking will be done via route_request updates."
            )

    def _on_service_discovery_event(
        self,
        event_type: ServiceDiscoveryEventType,
        engine_name: str,
        endpoint_info: Optional["EndpointInfo"],
    ):
        """
        Callback for service discovery events.

        Args:
            event_type: Type of event (ENGINE_ADDED, ENGINE_DELETED, etc.)
            engine_name: Name of the engine/pod
            endpoint_info: Endpoint information (may be None for some events)
        """
        if endpoint_info:
            # Filter: only process endpoints matching this router's strategy
            if (
                endpoint_info.routing_logic
                and endpoint_info.routing_logic != "consistent_hash"
            ):
                logger.debug(
                    f"ConsistentHashRouter: Skipping endpoint {endpoint_info.url} "
                    f"with routing_logic={endpoint_info.routing_logic}"
                )
                return
        if event_type == ServiceDiscoveryEventType.ENGINE_ADDED:
            if endpoint_info:
                routing_key = self._get_routing_key(endpoint_info)
                ring = self._get_or_create_ring(routing_key)
                with ring.lock:
                    was_added = endpoint_info.url not in ring.available_replicas
                    self._add_replica_to_ring(ring, endpoint_info)
                    if was_added:
                        logger.info(
                            f"ConsistentHashRouter: Added replica {engine_name} "
                            f"at {endpoint_info.url} to ring {routing_key}"
                        )

        elif event_type == ServiceDiscoveryEventType.ENGINE_DELETED:
            if endpoint_info:
                routing_key = self._get_routing_key(endpoint_info)
                # Use creation lock to safely check existence
                with self._hash_rings_creation_lock:
                    ring = self._hash_rings.get(routing_key)
                if ring:
                    with ring.lock:
                        if endpoint_info.url in ring.available_replicas:
                            self._remove_replica_from_ring(ring, endpoint_info.url)
                            logger.info(
                                f"ConsistentHashRouter: Removed replica {engine_name} "
                                f"at {endpoint_info.url} from ring {routing_key}"
                            )

    def _get_routing_key(self, endpoint_info: EndpointInfo) -> str:
        """
        Extract routing key from endpoint info.

        Args:
            endpoint_info: Endpoint information

        Returns:
            Routing key in format "workspace:endpoint"
        """
        workspace = endpoint_info.workspace
        endpoint = endpoint_info.endpoint
        return f"{workspace}:{endpoint}"

    def _get_or_create_ring(self, routing_key: str) -> HashRingState:
        """
        Get or create a hash ring for the given routing key.
        Thread-safe with double-checked locking pattern.

        Args:
            routing_key: The routing key (workspace:endpoint)

        Returns:
            HashRingState for this routing key
        """
        # Fast path: check without lock
        ring = self._hash_rings.get(routing_key)
        if ring is not None:
            return ring

        # Slow path: create with lock
        with self._hash_rings_creation_lock:
            # Double-check after acquiring lock
            ring = self._hash_rings.get(routing_key)
            if ring is None:
                ring = HashRingState()
                self._hash_rings[routing_key] = ring
                logger.debug(f"Created new hash ring for routing key: {routing_key}")
            return ring

    def _hash(self, key: str) -> int:
        """Hash a key to an integer value using MD5."""
        hash_obj = hashlib.md5(key.encode())
        # Use first 16 hex characters (8 bytes) as an integer
        return int(hash_obj.hexdigest()[:16], 16)

    def _search(self, ring: HashRingState, key_hash: int) -> Tuple[int, int]:
        """
        Find the hash point and its index on the ring for a given key hash.

        Args:
            ring: The hash ring state to search
            key_hash: The hash value to search for

        Returns:
            Tuple of (hash_value, index) on the ring
        """
        if not ring.sorted_hashes:
            raise ValueError("Hash ring is empty")

        # Binary search for the first hash >= key_hash
        idx = bisect.bisect_left(ring.sorted_hashes, key_hash)

        # If we're past the end, wrap around to the first hash
        if idx >= len(ring.sorted_hashes):
            idx = 0

        return ring.sorted_hashes[idx], idx

    def _add_replica_to_ring(self, ring: HashRingState, endpoint_info: EndpointInfo):
        """
        Add a replica to the hash ring with virtual nodes.

        Args:
            ring: The hash ring state to update
            endpoint_info: Information about the endpoint to add
        """
        url = endpoint_info.url

        # Check if replica is already in the ring (idempotency check)
        if url in ring.available_replicas:
            logger.debug(f"Replica {url} already exists in hash ring, skipping")
            return

        # Store the endpoint info
        ring.available_replicas[url] = endpoint_info

        # Add virtual nodes to the hash ring
        for i in range(self._virtual_nodes):
            # Create a unique hash for each virtual node
            virtual_node_key = f"{url}:{i}"
            hash_val = self._hash(virtual_node_key)

            # Add to hash ring
            ring.hash_to_endpoint_url[hash_val] = url
            bisect.insort(ring.sorted_hashes, hash_val)

        logger.debug(
            f"Added replica {url} to hash ring with {self._virtual_nodes} virtual nodes"
        )

    def _remove_replica_from_ring(self, ring: HashRingState, url: str):
        """
        Remove a replica from the hash ring.

        Args:
            ring: The hash ring state to update
            url: The endpoint URL to remove
        """
        # Check if replica exists
        if url not in ring.available_replicas:
            logger.debug(f"Replica {url} not found in hash ring, skipping removal")
            return

        # Remove from available replicas
        del ring.available_replicas[url]

        # Find all hash points for this replica
        hash_points_to_remove = []
        for hash_val, endpoint_url in ring.hash_to_endpoint_url.items():
            if endpoint_url == url:
                hash_points_to_remove.append(hash_val)

        # Remove from data structures
        for hash_val in hash_points_to_remove:
            del ring.hash_to_endpoint_url[hash_val]
            # Remove ALL occurrences of this hash (in case of duplicates)
            while hash_val in ring.sorted_hashes:
                idx = bisect.bisect_left(ring.sorted_hashes, hash_val)
                if (
                    idx < len(ring.sorted_hashes)
                    and ring.sorted_hashes[idx] == hash_val
                ):
                    ring.sorted_hashes.pop(idx)
                else:
                    break

        logger.debug(
            f"Removed replica {url} from hash ring "
            f"({len(hash_points_to_remove)} virtual nodes removed)"
        )

    def _extract_cache_key(self, request_json: Dict, request_id: str) -> str:
        """
        Extract cache key from OpenAI-compatible chat completions payload.

        For chat completions, we hash based on:
        1. System prompt (if present)
        2. First N user messages (configurable)

        This ensures that similar conversation contexts are routed to the same replica.

        Args:
            request_json: The request payload
            request_id: Fallback request ID if cache key cannot be extracted

        Returns:
            Cache key string
        """
        if not request_json:
            return str(request_id)

        try:
            cache_components = []

            # Handle chat completions format
            if "messages" in request_json:
                messages = request_json.get("messages", [])
                system_prompt = None
                user_messages = []

                for msg in messages:
                    if isinstance(msg, dict):
                        role = msg.get("role", "")
                        content = msg.get("content", "")

                        if role == "system":
                            system_prompt = content
                        elif role == "user":
                            user_messages.append(content)
                            # Early exit when we have enough user messages
                            if len(user_messages) >= self._max_user_messages_for_cache:
                                break

                # Add system prompt to cache key
                if system_prompt:
                    cache_components.append(f"system:{system_prompt}")

                # Add first N user messages
                for i, user_msg in enumerate(user_messages):
                    cache_components.append(f"user_{i}:{user_msg}")

            # Handle regular completions format
            elif "prompt" in request_json:
                prompt = request_json.get("prompt", "")
                cache_components.append(f"prompt:{prompt}")

            # Join components
            if cache_components:
                cache_key = "|".join(cache_components)
                logger.debug(f"Extracted cache key: {cache_key[:100]}...")
                return cache_key
            else:
                # No recognizable format, fallback
                logger.debug("No recognizable format, using request_id")
                return str(request_id)

        except Exception as e:
            logger.warning(f"Error extracting cache key: {e}, using request_id")
            return str(request_id)

    def _get_replica_load(self, url: str) -> Optional[int]:
        """
        Get the current load (active requests) for a replica.

        Args:
            url: The endpoint URL

        Returns:
            Number of active requests, or None if unknown
        """
        try:
            from vllm_router.stats.request_stats import get_request_stats_monitor

            monitor = get_request_stats_monitor()
            return monitor.get_active_request_count(url)
        except Exception as e:
            logger.warning(f"Could not get replica load for {url}: {e}")
            return None

    def _get_total_load(self, ring: HashRingState) -> int:
        """Get the total load across all replicas in a ring."""
        total = 0
        for url in ring.available_replicas:
            load = self._get_replica_load(url)
            if load is not None:
                total += load
        return total

    def _check_load(self, ring: HashRingState, url: str) -> bool:
        """
        Check if the replica meets the load constraints.

        Args:
            ring: The hash ring state
            url: The endpoint URL to check

        Returns:
            True if the replica is within load bounds, False otherwise
        """
        # Get the current load
        load = self._get_replica_load(url)
        if load is None:
            # If we can't determine load, assume it's OK
            return True

        # Calculate average load across all replicas in this ring
        num_replicas = len(ring.available_replicas)
        if num_replicas == 0:
            return True

        total_load = self._get_total_load(ring)
        avg_load = (total_load + 1) / num_replicas  # +1 for the current request

        # Apply load factor threshold
        threshold = avg_load * self._load_factor

        # Check if this replica is under the threshold (including the current request)
        return (load + 1) <= threshold

    def _sync_replicas(
        self, routing_key: str, ring: HashRingState, endpoints: List[EndpointInfo]
    ):
        """
        Synchronize a specific hash ring with the current list of endpoints.
        This is used as a fallback when service discovery callbacks are not available.

        Args:
            routing_key: The routing key for this ring
            ring: The hash ring state to synchronize
            endpoints: Current list of available endpoints for this routing key
        """
        # Build set of current endpoint URLs
        current_urls = {e.url for e in endpoints}
        available_urls = set(ring.available_replicas.keys())

        # Remove endpoints that are no longer available
        for url in available_urls - current_urls:
            self._remove_replica_from_ring(ring, url)

        # Add new endpoints
        for endpoint in endpoints:
            if endpoint.url not in available_urls:
                self._add_replica_to_ring(ring, endpoint)

        # Update sync time
        ring.last_sync_time = time.time()

    async def route_request(
        self,
        endpoints: List[EndpointInfo],
        engine_stats: Dict[str, EngineStats],
        request_stats: Dict[str, RequestStats],
        request: Request,
        request_json: Optional[Dict] = None,
    ) -> str:
        """
        Route request using consistent hashing with bounded load.

        Args:
            endpoints: List of available endpoints
            engine_stats: Engine statistics (not used in this router)
            request_stats: Request statistics (not used directly, uses RequestStatsMonitor)
            request: The incoming FastAPI request
            request_json: Parsed request JSON body

        Returns:
            The selected endpoint URL
        """
        if not endpoints:
            logger.error("No endpoints available for routing")
            return None

        # Extract routing key from the first endpoint
        routing_key = self._get_routing_key(endpoints[0])

        # Get or create hash ring for this routing key
        ring = self._get_or_create_ring(routing_key)

        # Use fine-grained lock for this specific routing key
        with ring.lock:
            # Sync replicas if needed
            if not ring.available_replicas or len(ring.available_replicas) != len(
                endpoints
            ):
                self._sync_replicas(routing_key, ring, endpoints)

            if not ring.available_replicas or len(ring.sorted_hashes) == 0:
                logger.warning(f"No replicas available for routing key {routing_key}")
                return endpoints[0].url

            # Parse request body if not provided
            if request_json is None:
                try:
                    request_json = await request.json()
                except Exception as e:
                    logger.warning(f"Could not parse request JSON: {e}")
                    request_json = {}

            # Extract cache key
            request_id = str(uuid.uuid4())
            cache_key = self._extract_cache_key(request_json, request_id)

            # Calculate hash of the cache key
            payload_hash = self._hash(cache_key)

            # Find initial replica using consistent hashing
            replica_hash, replica_idx = self._search(ring, payload_hash)
            initial_url = ring.hash_to_endpoint_url[replica_hash]

            logger.debug(
                f"CHWBL: Initial lookup for routing key {routing_key}, "
                f"payload hash {payload_hash} -> {initial_url}"
            )

            # Track replicas we've checked to avoid infinite loops
            checked_urls: Set[str] = set()
            default_url = None

            # Start from the initial replica and check load constraints
            current_idx = replica_idx
            while len(checked_urls) < len(ring.available_replicas):
                current_hash = ring.sorted_hashes[current_idx]
                current_url = ring.hash_to_endpoint_url[current_hash]

                # Skip if we've already checked this URL
                if current_url in checked_urls:
                    current_idx = (current_idx + 1) % len(ring.sorted_hashes)
                    continue

                checked_urls.add(current_url)

                # Save first valid URL as default
                if default_url is None:
                    default_url = current_url

                # Check if this replica meets the load constraints
                if self._check_load(ring, current_url):
                    logger.info(
                        f"CHWBL: Selected replica {current_url} for routing key {routing_key} "
                        f"after checking {len(checked_urls)} replicas (payload hash {payload_hash})"
                    )
                    return current_url

                # Move to next replica
                current_idx = (current_idx + 1) % len(ring.sorted_hashes)

            # If no replica satisfies the load factor, use the default
            if default_url:
                logger.info(
                    f"CHWBL: Using default replica {default_url} for routing key {routing_key} "
                    f"as no replica met load factor (payload hash {payload_hash})"
                )
                return default_url

            # No replicas available at all (shouldn't happen)
            logger.error(f"CHWBL: No replicas available for routing key {routing_key}")
            return endpoints[0].url


@dataclass
class StaticHashRouterState:
    """Encapsulates replica list state for a specific workspace+endpoint combination."""

    replica_list: List[EndpointInfo] = field(default_factory=list)
    last_sync_time: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


class StaticHashRouter(RoutingInterface):
    """
    Route requests using simple static hash-based routing (similar to neutree's StaticHashReplicaScheduler).

    This router ensures that identical payloads are always routed to the same replica.
    The scheduling is deterministic based on the hash of the request payload.

    Unlike ConsistentHashRouter:
    - No virtual nodes (simpler implementation)
    - No load factor checking (purely hash-based)
    - Replicas are stored in a list and selected by hash % replica_count
    - Multi-tenant support with separate replica lists per workspace+endpoint

    Use cases:
    - When you want deterministic routing based on payload
    - When load balancing is handled by other mechanisms
    - When simplicity is preferred over sophisticated load distribution
    """

    def __init__(self):
        """Initialize the Static Hash Router."""
        if hasattr(self, "_initialized"):
            return

        # Replica lists - separate list per workspace+endpoint
        self._replica_states: Dict[str, StaticHashRouterState] = {}
        # Global lock only for creating new routing_key entries
        self._states_creation_lock = threading.Lock()

        # Register callback with service discovery
        self._register_service_discovery_callback()

        logger.info("Initialized StaticHashRouter with multi-tenant support")
        self._initialized = True

    def _register_service_discovery_callback(self):
        """Register callback with K8sPodIPServiceDiscovery to track replica changes."""

        try:
            sd = get_service_discovery()

            # Check if service discovery supports callbacks (K8sPodIPServiceDiscovery)
            if hasattr(sd, "register_callback"):
                sd.register_callback(self._on_service_discovery_event)
                logger.info(
                    "Registered StaticHashRouter callback with service discovery"
                )
            else:
                logger.warning(
                    "Service discovery does not support callbacks. "
                    "Replica tracking will be done via route_request updates."
                )
        except Exception as e:
            logger.warning(
                f"Could not register service discovery callback: {e}. "
                f"Replica tracking will be done via route_request updates."
            )

    def _on_service_discovery_event(
        self,
        event_type: ServiceDiscoveryEventType,
        engine_name: str,
        endpoint_info: Optional["EndpointInfo"],
    ):
        """
        Callback for service discovery events.

        Args:
            event_type: Type of event (ENGINE_ADDED or ENGINE_DELETED)
            engine_name: Name of the engine/pod
            endpoint_info: Endpoint information (may be None for DELETED events)
        """

        if endpoint_info:
            # Filter: only process endpoints matching this router's strategy
            if (
                endpoint_info.routing_logic
                and endpoint_info.routing_logic != "static_hash"
            ):
                logger.debug(
                    f"StaticHashRouter: Skipping endpoint {endpoint_info.url} "
                    f"with routing_logic={endpoint_info.routing_logic}"
                )
                return
        if event_type == ServiceDiscoveryEventType.ENGINE_ADDED:
            if endpoint_info:
                routing_key = self._get_routing_key(endpoint_info)
                state = self._get_or_create_state(routing_key)
                with state.lock:
                    # Add replica to list if not already present
                    if endpoint_info not in state.replica_list:
                        state.replica_list.append(endpoint_info)
                        logger.info(
                            f"StaticHashRouter: Added replica {engine_name} at {endpoint_info.url} "
                            f"to routing key {routing_key}. Total replicas: {len(state.replica_list)}"
                        )

        elif event_type == ServiceDiscoveryEventType.ENGINE_DELETED:
            if endpoint_info:
                routing_key = self._get_routing_key(endpoint_info)
                # Use creation lock to safely check existence
                with self._states_creation_lock:
                    state = self._replica_states.get(routing_key)
                if state:
                    with state.lock:
                        # Remove replica from list
                        try:
                            state.replica_list.remove(endpoint_info)
                            logger.info(
                                f"StaticHashRouter: Removed replica {engine_name} at {endpoint_info.url} "
                                f"from routing key {routing_key}. Remaining replicas: {len(state.replica_list)}"
                            )
                        except ValueError:
                            logger.warning(
                                f"StaticHashRouter: Tried to remove non-existent replica {endpoint_info.url}"
                            )

    def _get_routing_key(self, endpoint_info: EndpointInfo) -> str:
        """
        Extract routing key from endpoint info.

        Args:
            endpoint_info: Endpoint information

        Returns:
            Routing key in format "workspace:endpoint"
        """
        workspace = endpoint_info.workspace
        endpoint = endpoint_info.endpoint
        return f"{workspace}:{endpoint}"

    def _get_or_create_state(self, routing_key: str) -> StaticHashRouterState:
        """
        Get or create a replica state for the given routing key.
        Thread-safe with double-checked locking pattern.

        Args:
            routing_key: The routing key (workspace:endpoint)

        Returns:
            StaticHashRouterState for this routing key
        """
        # Fast path: check without lock
        state = self._replica_states.get(routing_key)
        if state is not None:
            return state

        # Slow path: create with lock
        with self._states_creation_lock:
            # Double-check after acquiring lock
            state = self._replica_states.get(routing_key)
            if state is None:
                state = StaticHashRouterState()
                self._replica_states[routing_key] = state
                logger.debug(
                    f"Created new replica state for routing key: {routing_key}"
                )
            return state

    def _hash(self, key: str) -> int:
        """Hash a key to an integer value using MD5."""
        hash_obj = hashlib.md5(key.encode())
        # Use first 16 hex characters (8 bytes) as an integer
        return int(hash_obj.hexdigest()[:16], 16)

    def _extract_payload_key(self, request_json: Dict, request_id: str) -> str:
        """
        Extract payload key from request for hashing.

        Similar to ConsistentHashRouter, but we extract the entire message payload
        for more deterministic routing.

        Args:
            request_json: The request payload
            request_id: Fallback request ID if payload key cannot be extracted

        Returns:
            Payload key string
        """
        if not request_json:
            return str(request_id)

        try:
            # For chat completions, hash all messages
            if "messages" in request_json:
                messages = request_json.get("messages", [])
                # Serialize messages to a string
                message_str = str(messages)
                return f"messages:{message_str}"

            # For regular completions, hash the prompt
            elif "prompt" in request_json:
                prompt = request_json.get("prompt", "")
                return f"prompt:{prompt}"

            # Fallback to request ID
            return str(request_id)

        except Exception as e:
            logger.warning(f"Error extracting payload key: {e}, using request_id")
            return str(request_id)

    def _sync_replicas(
        self,
        routing_key: str,
        state: StaticHashRouterState,
        endpoints: List[EndpointInfo],
    ):
        """
        Synchronize the replica list with the current list of endpoints for a specific routing key.
        This is used as a fallback when service discovery callbacks are not available.

        Args:
            routing_key: The routing key for this state
            state: The replica state to synchronize
            endpoints: Current list of available endpoints for this routing key
        """
        # Simple replacement: update the list
        state.replica_list = list(endpoints)
        state.last_sync_time = time.time()
        logger.debug(
            f"StaticHashRouter: Synced {len(state.replica_list)} replicas for routing key {routing_key}"
        )

    async def route_request(
        self,
        endpoints: List[EndpointInfo],
        engine_stats: Dict[str, EngineStats],
        request_stats: Dict[str, RequestStats],
        request: Request,
        request_json: Optional[Dict] = None,
    ) -> str:
        """
        Route request using simple static hash.

        Args:
            endpoints: List of available endpoints
            engine_stats: Engine statistics (not used in this router)
            request_stats: Request statistics (not used in this router)
            request: The incoming FastAPI request
            request_json: Parsed request JSON body

        Returns:
            The selected endpoint URL
        """
        if not endpoints:
            logger.error("No endpoints available for routing")
            return None

        # Extract routing key from the first endpoint
        routing_key = self._get_routing_key(endpoints[0])

        # Get or create replica state for this routing key
        state = self._get_or_create_state(routing_key)

        # Use fine-grained lock for this specific routing key
        with state.lock:
            # Sync replicas if needed
            if not state.replica_list or len(state.replica_list) != len(endpoints):
                self._sync_replicas(routing_key, state, endpoints)

            if not state.replica_list:
                logger.warning(f"No replicas available for routing key {routing_key}")
                return endpoints[0].url

            # Parse request body if not provided
            if request_json is None:
                try:
                    request_json = await request.json()
                except Exception as e:
                    logger.warning(f"Could not parse request JSON: {e}")
                    request_json = {}

            # Extract payload key
            request_id = str(uuid.uuid4())
            payload_key = self._extract_payload_key(request_json, request_id)

            # Calculate hash of the payload
            payload_hash = self._hash(payload_key)

            # Select replica by hash % replica_count
            replica_idx = payload_hash % len(state.replica_list)
            selected_replica = state.replica_list[replica_idx]

            logger.debug(
                f"StaticHashRouter: Payload hash {payload_hash} for routing key {routing_key} -> "
                f"replica {replica_idx}/{len(state.replica_list)} ({selected_replica.url})"
            )

            return selected_replica.url


# Instead of managing a global _global_router, we can define the initialization functions as:
def initialize_routing_logic(
    routing_logic: RoutingLogic, *args, **kwargs
) -> RoutingInterface:
    if routing_logic == RoutingLogic.ROUND_ROBIN:
        logger.info("Initializing round-robin routing logic")
        return RoundRobinRouter()
    elif routing_logic == RoutingLogic.SESSION_BASED:
        logger.info(f"Initializing session-based routing logic with kwargs: {kwargs}")
        return SessionRouter(kwargs.get("session_key"))
    elif routing_logic == RoutingLogic.KVAWARE:
        logger.info("Initializing kvaware routing logic")
        router = KvawareRouter(
            kwargs.get("lmcache_controller_port"),
            kwargs.get("session_key"),
            kwargs.get("kv_aware_threshold"),
        )
        router.start_kv_manager()
        return router
    elif routing_logic == RoutingLogic.PREFIXAWARE:
        logger.info("Initializing prefix-aware routing logic")
        return PrefixAwareRouter()
    elif routing_logic == RoutingLogic.DISAGGREGATED_PREFILL:
        logger.info("Initializing disaggregated prefill routing logic")
        return DisaggregatedPrefillRouter(
            kwargs.get("prefill_model_labels"), kwargs.get("decode_model_labels")
        )
    elif routing_logic == RoutingLogic.PD:
        logger.info("Initializing P/D same-host routing logic")
        return PDRouter(
            virtual_nodes_per_replica=kwargs.get("virtual_nodes_per_replica", 100),
            load_factor=kwargs.get("load_factor", 1.25),
            max_user_messages_for_cache=kwargs.get("max_user_messages_for_cache", 2),
        )
    elif routing_logic == RoutingLogic.CONSISTENT_HASH:
        logger.info("Initializing consistent hash routing logic")
        return ConsistentHashRouter(
            virtual_nodes_per_replica=kwargs.get("virtual_nodes_per_replica", 100),
            load_factor=kwargs.get("load_factor", 1.25),
            max_user_messages_for_cache=kwargs.get("max_user_messages_for_cache", 2),
        )
    elif routing_logic == RoutingLogic.STATIC_HASH:
        logger.info("Initializing static hash routing logic")
        return StaticHashRouter()
    else:
        raise ValueError(f"Invalid routing logic {routing_logic}")


def reconfigure_routing_logic(
    routing_logic: RoutingLogic, *args, **kwargs
) -> RoutingInterface:
    # Remove the existing routers from the singleton registry
    cleanup_routing_logic()
    return initialize_routing_logic(routing_logic, *args, **kwargs)


def get_routing_logic() -> RoutingInterface:
    # Look up in our singleton registry which router (if any) has been created.
    for cls in (
        SessionRouter,
        RoundRobinRouter,
        KvawareRouter,
        PrefixAwareRouter,
        DisaggregatedPrefillRouter,
        PDRouter,
        ConsistentHashRouter,
        StaticHashRouter,
    ):
        if cls in SingletonABCMeta._instances:
            return cls()
    raise ValueError("The global router has not been initialized")


def get_routing_logic_by_type(routing_logic: RoutingLogic) -> RoutingInterface:
    """Gets the initialized routing logic instance of a specific type."""
    target_cls_name = ROUTING_LOGIC_TO_CLASS.get(routing_logic)
    if not target_cls_name:
        raise ValueError(f"The router of type {routing_logic.value} does not exist.")

    for cls in (
        SessionRouter,
        RoundRobinRouter,
        KvawareRouter,
        PrefixAwareRouter,
        DisaggregatedPrefillRouter,
        PDRouter,
        ConsistentHashRouter,
        StaticHashRouter,
    ):
        if cls.__name__ == target_cls_name and cls in SingletonABCMeta._instances:
            return cls()

    raise ValueError(
        f"The router of type {routing_logic.value} has not been initialized"
    )


def cleanup_routing_logic():
    """Clean up all routing logic instances."""
    for cls in (
        SessionRouter,
        RoundRobinRouter,
        KvawareRouter,
        PrefixAwareRouter,
        DisaggregatedPrefillRouter,
        PDRouter,
        ConsistentHashRouter,
        StaticHashRouter,
    ):
        if cls in SingletonABCMeta._instances:
            instance = cls()
            if hasattr(instance, "close"):
                instance.close()
            del SingletonABCMeta._instances[cls]
