---
title: 使用 Engine 参数配置数据并行建联
---

在使用 `torchrun` 等 **外部启动器** 启动 vLLM 时，可直接在 Engine 参数中传入所有数据并行相关字段，而无需设置 `VLLM_DP_*` 环境变量。`data_parallel_size` 与是否使用外部启动器并无直接关系，下文仅以其大于 `1` 的场景为例说明握手流程。此方法既适用于离线推理也适用于 API 服务。

## 两级数据并行

在 vLLM 中，`data_parallel_size` 表示全局的数据并行度，可跨多机或多进程使用；`data_parallel_size_local` 则决定每个节点同时启动多少个 Engine 实例，即 "external DP"。两者互不影响。

## 原理概览

初始化阶段，`ParallelConfig.__post_init__` 会检查是否在参数中显式指定了数据并行配置：

```python
if self.data_parallel_size > 1 or self.data_parallel_size_local == 0:
    # 在 Engine 参数中指定了数据并行信息。
    self.data_parallel_master_port = get_open_port()
else:
    # 否则退回到环境变量（例如离线单机多进程的情况）。
    self.data_parallel_size = envs.VLLM_DP_SIZE
    self.data_parallel_rank = envs.VLLM_DP_RANK
    self.data_parallel_rank_local = envs.VLLM_DP_RANK_LOCAL
    self.data_parallel_master_ip = envs.VLLM_DP_MASTER_IP
    self.data_parallel_master_port = envs.VLLM_DP_MASTER_PORT
```

随后，主进程在启动握手阶段会通过 `EngineHandshakeMetadata` 将这些设置发送给各个引擎：

```python
init_message = msgspec.msgpack.encode(
    EngineHandshakeMetadata(
        addresses=addresses,
        parallel_config={
            "data_parallel_master_ip": parallel_config.data_parallel_master_ip,
            "data_parallel_master_port": parallel_config.data_parallel_master_port,
            "data_parallel_size": parallel_config.data_parallel_size,
        }))
```

每个引擎收到消息后都会更新自身的 `ParallelConfig`：

```python
init_message: EngineHandshakeMetadata = msgspec.msgpack.decode(init_bytes, type=EngineHandshakeMetadata)
received_parallel_config = init_message.parallel_config
for key, value in received_parallel_config.items():
    setattr(parallel_config, key, value)
```

## 示例用法

使用 `torchrun` 启动引擎，并直接传入数据并行参数：

```bash
# 在同一台机器上启动两个 rank
torchrun --nproc-per-node=2 run.py
```

`run.py` 中创建引擎无需依赖环境变量：

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.1-8B",
    tensor_parallel_size=2,
    distributed_executor_backend="external_launcher",
    data_parallel_size=2,
    data_parallel_master_ip="10.0.0.1",
    data_parallel_master_port=29500,
    data_parallel_rpc_port=29501,
)
```

上述参数同样适用于 `vllm serve`：

```bash
torchrun --nproc-per-node=2 vllm serve meta-llama/Llama-3.1-8B \
    --distributed-executor-backend external_launcher \
    --data-parallel-size 2 \
    --data-parallel-master-ip 10.0.0.1 \
    --data-parallel-master-port 29500 \
    --data-parallel-rpc-port 29501
```

通过在 Engine 参数中提供这些字段，启动时的握手信息会自动完成传递，因此无需设置 `VLLM_DP_MASTER_IP` 等环境变量。

## `data_parallel_rpc_port` 的作用

当引擎位于不同主机时，无法依赖本地 IPC 套接字进行启动和协同。`data_parallel_rpc_port` 用于指定跨节点 ZMQ 连接的 TCP 端口，既在最初的握手阶段使用，也被 `DPCoordinator` 用来在各数据并行 rank 之间同步信息。若所有引擎都在同一台机器上，可保持默认值，届时会使用 IPC 路径而非 TCP。

## 外部启动器下的建联流程

下面以外部启动器配合 `data_parallel_size` 大于 1 的情形为例，建联过程如下（size 为 1 时会跳过步骤 2 和 4）：

1. **初始化 World 进程组**：`ExecutorWithExternalLauncher` 通过 `env://` 创建全局进程组，依赖启动器设置的 `RANK`、`LOCAL_RANK`、`MASTER_ADDR`、`MASTER_PORT` 等环境变量。
2. **创建数据并行组**：若 `data_parallel_size_local` > 1，
   同一节点的多个进程会以不同的 `LOCAL_RANK` 参与数据并行。主节点调用 `ParallelConfig.stateless_init_dp_group()` 形成跨节点的 gloo 组，此步骤在模型加载前完成。
3. **引擎握手**：每个进程向前端发送 `HELLO`，`wait_for_engine_startup` 会以 `EngineHandshakeMetadata` 回复 DP master IP、端口、size 等信息，并告知其 `LOCAL_RANK`。引擎据此更新 `ParallelConfig` 并在完成初始化后回复 `READY`。
4. **启动协调器**：若数据并行 rank 多于 1，则会启动 `DPCoordinator` 进程。它使用 `data_parallel_rpc_port` 在各 rank 之间同步请求波次，并把引擎状态回传给前端。

按上述流程，即使所有进程完全由外部启动器拉起，head 节点仍能协调分布在多机上的数据并行引擎完成联机运行。
