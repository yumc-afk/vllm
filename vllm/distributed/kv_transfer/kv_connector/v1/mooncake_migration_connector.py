# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import torch
from safetensors.torch import load as safetensors_load
from safetensors.torch import save as safetensors_save

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)


_KIND = "mooncake_kv_migration_v1"


@dataclass
class _SaveReq:
    request_id: str
    snapshot_id: str
    commit_id: int
    num_cached_tokens: int
    block_ids: list[int]
    model_hash: str
    block_size: int
    tp_world_size: int
    chunk_blocks: int


@dataclass
class _LoadReq:
    request_id: str
    snapshot_id: str
    expected_head_commit: int
    num_cached_tokens: int
    block_ids: list[int]
    model_hash: str
    block_size: int
    tp_world_size: int
    chunk_blocks: int


@dataclass
class MooncakeMigrationConnectorMetadata(KVConnectorMetadata):
    save_reqs: list[_SaveReq] = field(default_factory=list)
    load_reqs: list[_LoadReq] = field(default_factory=list)


def _align_down(num_tokens: int, block_size: int) -> int:
    if num_tokens <= 0:
        return 0
    return (num_tokens // block_size) * block_size


def _is_mla_kv_format(kv_layer: torch.Tensor) -> bool:
    # non-MLA paged KV is typically [2, num_pages, page_size, ...]
    return not (kv_layer.dim() >= 3 and kv_layer.shape[0] == 2)


def _gather_blocks(kv_layer: torch.Tensor, block_ids: list[int]) -> torch.Tensor:
    if _is_mla_kv_format(kv_layer):
        return kv_layer[block_ids, ...]
    return kv_layer[:, block_ids, ...]


def _scatter_blocks(kv_layer: torch.Tensor, block_ids: list[int],
                    blocks: torch.Tensor) -> None:
    if _is_mla_kv_format(kv_layer):
        kv_layer[block_ids, ...] = blocks
    else:
        kv_layer[:, block_ids, ...] = blocks


class _MooncakeBytesStore:

    def __init__(self, vllm_config: VllmConfig):
        try:
            from mooncake.store import MooncakeDistributedStore  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Missing dependency 'mooncake'. Please install mooncake to use "
                "MooncakeMigrationConnector."
            ) from e

        from vllm.distributed.kv_transfer.kv_lookup_buffer.mooncake_store import (
            MooncakeStoreConfig,
        )

        self._store = MooncakeDistributedStore()
        self._cfg = MooncakeStoreConfig.load_from_env()
        self._store.setup(
            self._cfg.local_hostname,
            self._cfg.metadata_server,
            self._cfg.global_segment_size,
            self._cfg.local_buffer_size,
            self._cfg.protocol,
            self._cfg.device_name,
            self._cfg.master_server_address,
        )

        extra = (vllm_config.kv_transfer_config.kv_connector_extra_config
                 if vllm_config.kv_transfer_config else {})
        self._prefix = str(extra.get("mooncake_kv_mig_prefix", "vllm_kv_mig"))

    def _key_head(self, snapshot_id: str) -> str:
        return f"{self._prefix}/v1/{snapshot_id}/head"

    def _key_manifest(self, snapshot_id: str, commit_id: int) -> str:
        return f"{self._prefix}/v1/{snapshot_id}/commits/{commit_id}/manifest.json"

    def _key_layer_part(self, snapshot_id: str, commit_id: int, tp_rank: int,
                        layer_name: str, part_idx: int) -> str:
        safe_layer = layer_name.replace("/", "__")
        return (
            f"{self._prefix}/v1/{snapshot_id}/commits/{commit_id}/layers/"
            f"{tp_rank}/{safe_layer}/part_{part_idx:06d}"
        )

    def put_bytes(self, key: str, value: bytes) -> None:
        self._store.put(key, value)

    def get_bytes(self, key: str) -> Optional[bytes]:
        data = self._store.get(key)
        return data if data else None

    def put_tensor(self, key: str, tensor: torch.Tensor) -> None:
        payload = safetensors_save({"tensor": tensor})
        self.put_bytes(key, payload)

    def get_tensor(self, key: str) -> Optional[torch.Tensor]:
        data = self.get_bytes(key)
        if data is None:
            return None
        return safetensors_load(data)["tensor"]

    def put_manifest(self, snapshot_id: str, commit_id: int,
                     manifest: dict[str, Any]) -> None:
        self.put_bytes(
            self._key_manifest(snapshot_id, commit_id),
            json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode(),
        )

    def get_manifest(self, snapshot_id: str,
                     commit_id: int) -> Optional[dict[str, Any]]:
        data = self.get_bytes(self._key_manifest(snapshot_id, commit_id))
        if data is None:
            return None
        return json.loads(data.decode())

    def put_head(self, snapshot_id: str, commit_id: int) -> None:
        self.put_bytes(self._key_head(snapshot_id), str(commit_id).encode())

    def get_head(self, snapshot_id: str) -> Optional[int]:
        data = self.get_bytes(self._key_head(snapshot_id))
        if data is None:
            return None
        try:
            return int(data.decode())
        except Exception:
            return None


class _MooncakeMigrationConnectorWorker:

    def __init__(self, vllm_config: VllmConfig):
        self._vllm_config = vllm_config
        self._kv_caches: dict[str, torch.Tensor] = {}
        self._store: Optional[_MooncakeBytesStore] = None

        self._saved_req_ids: set[str] = set()
        self._loaded_req_ids: set[str] = set()
        self._finished_sending: set[str] = set()
        self._finished_recving: set[str] = set()

    def _get_store(self) -> _MooncakeBytesStore:
        if self._store is None:
            self._store = _MooncakeBytesStore(self._vllm_config)
        return self._store

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self._kv_caches = kv_caches

    def start_load_kv(self, meta: MooncakeMigrationConnectorMetadata) -> None:
        if not meta.load_reqs:
            return

        store = self._get_store()
        tp_rank = get_tp_group().rank
        for req in meta.load_reqs:
            if req.request_id in self._loaded_req_ids:
                continue

            head_commit = store.get_head(req.snapshot_id)
            if head_commit is None:
                continue
            if head_commit != req.expected_head_commit:
                raise ValueError(
                    f"snapshot head_commit mismatch: {req.snapshot_id} "
                    f"head={head_commit} expected={req.expected_head_commit}")

            manifest = store.get_manifest(req.snapshot_id, head_commit)
            if manifest is None:
                continue
            if int(manifest.get("block_size", -1)) != req.block_size:
                raise ValueError(
                    f"block_size mismatch: {manifest.get('block_size')} vs "
                    f"{req.block_size}")
            if str(manifest.get("model_hash", "")) != req.model_hash:
                raise ValueError("model_hash mismatch")
            if int(manifest.get("tp_world_size", -1)) != req.tp_world_size:
                raise ValueError("tp_world_size mismatch")

            num_cached_blocks = len(req.block_ids)
            num_parts = int(manifest.get("num_parts", 0))
            if num_parts <= 0:
                raise ValueError("invalid manifest: num_parts")

            for layer_name, kv_layer in self._kv_caches.items():
                parts: list[torch.Tensor] = []
                for part_idx in range(num_parts):
                    key = store._key_layer_part(req.snapshot_id, head_commit,
                                                tp_rank, layer_name, part_idx)
                    part = store.get_tensor(key)
                    if part is None:
                        raise ValueError(
                            f"missing layer part: {key} (snapshot not complete?)"
                        )
                    parts.append(part)
                blocks = torch.cat(
                    parts,
                    dim=0 if _is_mla_kv_format(kv_layer) else 1,
                )
                blocks = blocks.to(kv_layer.device)

                if (_is_mla_kv_format(kv_layer)
                        and blocks.shape[0] != num_cached_blocks):
                    raise ValueError("loaded blocks mismatch (MLA)")
                if (not _is_mla_kv_format(kv_layer)
                        and blocks.shape[1] != num_cached_blocks):
                    raise ValueError("loaded blocks mismatch")

                _scatter_blocks(kv_layer, req.block_ids, blocks)

            self._loaded_req_ids.add(req.request_id)
            self._finished_recving.add(req.request_id)

    def get_finished(
        self,
        meta: MooncakeMigrationConnectorMetadata,
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        active_save_ids = {r.request_id for r in meta.save_reqs}
        active_load_ids = {r.request_id for r in meta.load_reqs}
        if self._saved_req_ids:
            self._saved_req_ids.intersection_update(active_save_ids)
        if self._loaded_req_ids:
            self._loaded_req_ids.intersection_update(active_load_ids)

        if meta.save_reqs:
            self._save(meta.save_reqs)
        finished_sending = set(self._finished_sending)
        finished_recving = set(self._finished_recving)
        self._finished_sending.clear()
        self._finished_recving.clear()
        return (finished_sending or None), (finished_recving or None)

    def _save(self, save_reqs: list[_SaveReq]) -> None:
        store = self._get_store()
        tp = get_tp_group()
        tp_rank = tp.rank
        for req in save_reqs:
            if req.request_id in self._saved_req_ids:
                continue

            num_cached_blocks = len(req.block_ids)
            if num_cached_blocks == 0:
                self._saved_req_ids.add(req.request_id)
                self._finished_sending.add(req.request_id)
                continue

            chunk_blocks = max(int(req.chunk_blocks), 0)
            num_parts = 1
            if chunk_blocks > 0:
                num_parts = (num_cached_blocks + chunk_blocks -
                             1) // chunk_blocks

            for layer_name, kv_layer in self._kv_caches.items():
                blocks = _gather_blocks(kv_layer, req.block_ids).detach().cpu()
                for part_idx in range(num_parts):
                    if chunk_blocks <= 0:
                        part = blocks
                    else:
                        start = part_idx * chunk_blocks
                        end = min(start + chunk_blocks, num_cached_blocks)
                        if _is_mla_kv_format(kv_layer):
                            part = blocks[start:end, ...]
                        else:
                            part = blocks[:, start:end, ...]
                    key = store._key_layer_part(req.snapshot_id, req.commit_id,
                                                tp_rank, layer_name, part_idx)
                    store.put_tensor(key, part)

            tp.barrier()
            if tp_rank == 0:
                manifest = {
                    "format_version": 1,
                    "kind": _KIND,
                    "model_hash": req.model_hash,
                    "tp_world_size": req.tp_world_size,
                    "block_size": req.block_size,
                    "num_cached_tokens": req.num_cached_tokens,
                    "num_cached_blocks": num_cached_blocks,
                    "chunk_blocks": req.chunk_blocks,
                    "num_parts": num_parts,
                    "created_at_ms": int(time.time() * 1000),
                }
                store.put_manifest(req.snapshot_id, req.commit_id, manifest)
                store.put_head(req.snapshot_id, req.commit_id)
            tp.barrier()

            self._saved_req_ids.add(req.request_id)
            self._finished_sending.add(req.request_id)


class _MooncakeMigrationConnectorScheduler:

    def __init__(self, vllm_config: VllmConfig):
        self._vllm_config = vllm_config
        self._block_size = vllm_config.cache_config.block_size
        self._tp_world_size = vllm_config.parallel_config.tensor_parallel_size
        self._model_hash = vllm_config.compute_hash()

        extra = (vllm_config.kv_transfer_config.kv_connector_extra_config
                 if vllm_config.kv_transfer_config else {})
        self._chunk_blocks = int(extra.get("mooncake_kv_mig_chunk_blocks", 0))

        self._export_snapshot_by_req_id: dict[str, str] = {}
        self._inflight_saves: dict[str, _SaveReq] = {}
        self._inflight_loads: dict[str, _LoadReq] = {}

    def schedule_export(self, request_id: str, snapshot_id: str) -> None:
        self._export_snapshot_by_req_id[request_id] = snapshot_id

    def on_finished_sending(self, request_id: str) -> None:
        self._inflight_saves.pop(request_id, None)

    def on_finished_recving(self, request_id: str) -> None:
        self._inflight_loads.pop(request_id, None)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        params = request.kv_transfer_params
        if not params or params.get("kind") != _KIND:
            return 0, False

        num_cached_tokens = int(params.get("num_cached_tokens", 0))
        block_size = int(params.get("block_size", self._block_size))
        model_hash = str(params.get("model_hash", ""))
        tp_world_size = int(params.get("tp_world_size", self._tp_world_size))
        if block_size != self._block_size:
            raise ValueError("block_size mismatch")
        if model_hash and model_hash != self._model_hash:
            raise ValueError("model_hash mismatch")
        if tp_world_size != self._tp_world_size:
            raise ValueError("tp_world_size mismatch")

        max_cacheable = max(request.num_tokens - 1, 0)
        num_cached_tokens = min(num_cached_tokens, max_cacheable)
        num_cached_tokens = _align_down(num_cached_tokens, self._block_size)
        if num_cached_tokens <= num_computed_tokens:
            return 0, False

        # Always async-load for robustness (snapshot may not be committed yet).
        return num_cached_tokens - num_computed_tokens, True

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        if num_external_tokens <= 0:
            return

        params = request.kv_transfer_params
        if not params or params.get("kind") != _KIND:
            return

        snapshot_id = str(params["snapshot_id"])
        expected_head_commit = int(params.get("head_commit", 1))
        block_ids = list(blocks.get_block_ids()[0])

        self._inflight_loads[request.request_id] = _LoadReq(
            request_id=request.request_id,
            snapshot_id=snapshot_id,
            expected_head_commit=expected_head_commit,
            num_cached_tokens=int(params.get("num_cached_tokens",
                                             num_external_tokens)),
            block_ids=block_ids,
            model_hash=str(params.get("model_hash", self._model_hash)),
            block_size=int(params.get("block_size", self._block_size)),
            tp_world_size=int(params.get("tp_world_size", self._tp_world_size)),
            chunk_blocks=int(params.get("chunk_blocks", self._chunk_blocks)),
        )

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        snapshot_id = self._export_snapshot_by_req_id.pop(request.request_id,
                                                         "")
        if not snapshot_id:
            return False, None

        commit_id = 1
        num_cached_tokens = min(
            int(request.num_computed_tokens),
            max(int(request.num_tokens) - 1, 0),
        )
        num_cached_tokens = _align_down(num_cached_tokens, self._block_size)
        num_cached_blocks = num_cached_tokens // self._block_size
        export_block_ids = list(block_ids[:num_cached_blocks])

        self._inflight_saves[request.request_id] = _SaveReq(
            request_id=request.request_id,
            snapshot_id=snapshot_id,
            commit_id=commit_id,
            num_cached_tokens=num_cached_tokens,
            block_ids=export_block_ids,
            model_hash=self._model_hash,
            block_size=self._block_size,
            tp_world_size=self._tp_world_size,
            chunk_blocks=self._chunk_blocks,
        )

        kv_transfer_params = {
            "kind": _KIND,
            "snapshot_id": snapshot_id,
            "head_commit": commit_id,
            "num_cached_tokens": num_cached_tokens,
            "block_size": self._block_size,
            "tp_world_size": self._tp_world_size,
            "model_hash": self._model_hash,
            "chunk_blocks": self._chunk_blocks,
        }
        return True, kv_transfer_params

    def build_connector_meta(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> KVConnectorMetadata:
        return MooncakeMigrationConnectorMetadata(
            save_reqs=list(self._inflight_saves.values()),
            load_reqs=list(self._inflight_loads.values()),
        )


class MooncakeMigrationConnector(KVConnectorBase_V1):

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self._scheduler: Optional[_MooncakeMigrationConnectorScheduler] = None
        self._worker: Optional[_MooncakeMigrationConnectorWorker] = None
        if role == KVConnectorRole.SCHEDULER:
            self._scheduler = _MooncakeMigrationConnectorScheduler(vllm_config)
        else:
            self._worker = _MooncakeMigrationConnectorWorker(vllm_config)

    # ---------- scheduler-side extensions ----------

    def schedule_export(self, request_id: str, snapshot_id: str) -> None:
        assert self._scheduler is not None
        self._scheduler.schedule_export(request_id, snapshot_id)

    def on_finished_sending(self, request_id: str) -> None:
        assert self._scheduler is not None
        self._scheduler.on_finished_sending(request_id)

    def on_finished_recving(self, request_id: str) -> None:
        assert self._scheduler is not None
        self._scheduler.on_finished_recving(request_id)

    # ---------- scheduler-side required ----------

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        assert self._scheduler is not None
        return self._scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        assert self._scheduler is not None
        self._scheduler.update_state_after_alloc(request, blocks,
                                                 num_external_tokens)

    def build_connector_meta(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> KVConnectorMetadata:
        assert self._scheduler is not None
        return self._scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        assert self._scheduler is not None
        return self._scheduler.request_finished(request, block_ids)

    # ---------- worker-side required ----------

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self._worker is not None
        self._worker.register_kv_caches(kv_caches)

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        assert self._worker is not None
        assert isinstance(self._connector_metadata,
                          MooncakeMigrationConnectorMetadata)
        self._worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self,
                      layer_name: str,
                      kv_layer: torch.Tensor,
                      attn_metadata: Any,
                      **kwargs) -> None:
        return

    def wait_for_save(self):
        return

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        assert self._worker is not None
        assert isinstance(self._connector_metadata,
                          MooncakeMigrationConnectorMetadata)
        return self._worker.get_finished(self._connector_metadata)

