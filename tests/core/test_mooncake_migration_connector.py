# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import torch

from vllm.distributed.kv_transfer.kv_connector.v1 import (
    mooncake_migration_connector as mmc,
)


class _DummyTPGroup:
    rank = 0

    def barrier(self):
        return


class _InMemStore:

    def __init__(self, prefix: str = "test"):
        self._prefix = prefix
        self._kv: dict[str, bytes] = {}

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
        self._kv[key] = value

    def get_bytes(self, key: str) -> bytes | None:
        return self._kv.get(key)

    def put_tensor(self, key: str, tensor: torch.Tensor) -> None:
        payload = mmc.safetensors_save({"tensor": tensor})
        self.put_bytes(key, payload)

    def get_tensor(self, key: str) -> torch.Tensor | None:
        data = self.get_bytes(key)
        if data is None:
            return None
        return mmc.safetensors_load(data)["tensor"]

    def put_manifest(self, snapshot_id: str, commit_id: int, manifest: dict):
        self.put_bytes(self._key_manifest(snapshot_id, commit_id),
                       mmc.json.dumps(manifest).encode())

    def get_manifest(self, snapshot_id: str, commit_id: int) -> dict | None:
        data = self.get_bytes(self._key_manifest(snapshot_id, commit_id))
        if data is None:
            return None
        return mmc.json.loads(data.decode())

    def put_head(self, snapshot_id: str, commit_id: int) -> None:
        self.put_bytes(self._key_head(snapshot_id), str(commit_id).encode())

    def get_head(self, snapshot_id: str) -> int | None:
        data = self.get_bytes(self._key_head(snapshot_id))
        if data is None:
            return None
        return int(data.decode())


def test_mooncake_migration_save_and_load_cpu():
    mmc.get_tp_group = lambda: _DummyTPGroup()  # type: ignore[assignment]

    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=4),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={}),
        compute_hash=lambda: "hash0",
    )

    store = _InMemStore(prefix="p")
    layer_name = "model.layers.0.attn"

    src_kv = torch.randn(2, 16, 4, 3)
    src_block_ids = [2, 5, 7]
    # Keep only chosen blocks non-zero for easy comparison.
    mask = torch.ones_like(src_kv)
    mask[:, src_block_ids, :, :] = 0
    src_kv = src_kv * (1 - mask)

    w_save = mmc._MooncakeMigrationConnectorWorker(vllm_config)  # type: ignore[arg-type]
    w_save._store = store  # type: ignore[assignment]
    w_save.register_kv_caches({layer_name: src_kv})

    save_req = mmc._SaveReq(
        request_id="r0",
        snapshot_id="s0",
        commit_id=1,
        num_cached_tokens=12,
        block_ids=src_block_ids,
        model_hash="hash0",
        block_size=4,
        tp_world_size=1,
        chunk_blocks=2,
    )
    meta = mmc.MooncakeMigrationConnectorMetadata(save_reqs=[save_req],
                                                 load_reqs=[])
    finished_sending, _ = w_save.get_finished(meta)
    assert finished_sending == {"r0"}
    assert store.get_head("s0") == 1
    manifest = store.get_manifest("s0", 1)
    assert manifest is not None
    assert manifest["num_parts"] == 2

    dst_kv = torch.zeros_like(src_kv)
    dst_block_ids = [1, 3, 4]
    w_load = mmc._MooncakeMigrationConnectorWorker(vllm_config)  # type: ignore[arg-type]
    w_load._store = store  # type: ignore[assignment]
    w_load.register_kv_caches({layer_name: dst_kv})

    load_req = mmc._LoadReq(
        request_id="r1",
        snapshot_id="s0",
        expected_head_commit=1,
        num_cached_tokens=12,
        block_ids=dst_block_ids,
        model_hash="hash0",
        block_size=4,
        tp_world_size=1,
        chunk_blocks=2,
    )
    meta2 = mmc.MooncakeMigrationConnectorMetadata(save_reqs=[],
                                                  load_reqs=[load_req])
    w_load.start_load_kv(meta2)
    _, finished_recving = w_load.get_finished(meta2)
    assert finished_recving == {"r1"}
    assert torch.allclose(dst_kv[:, dst_block_ids, :, :],
                          src_kv[:, src_block_ids, :, :])

