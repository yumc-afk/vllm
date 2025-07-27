---
title: TP下的 WQKV 偏置加载流程
---
[](){ #wqkv-bias-tp }

本文档简要说明了在使用 Tensor Parallelism (TP) 时，`wqkv`的 bias 参数如何被加载。

## 1. WQKV 层的建立

`wqkv` 层通过 [`QKVParallelLinear`](../contributing/model/basic.md)类实现，继承自 `ColumnParallelLinear`。当模型定义该层时，如 [`InternLM2Attention`](gh-file:vllm/model_executor/models/internlm2.py#L110-L130)中所示，可指定是否包含 bias 。

## 2. 初始化与 bias 层

当初始化 `QKVParallelLinear`时，若使用 `bias=True`，会为每个 Tensor Parallel rank 初始化归属于本分片的 bias 数组，代码如下：

```python
   if bias:
       self.bias = Parameter(
           torch.empty(self.output_size_per_partition, dtype=self.params_dtype))
       set_weight_attrs(self.bias, {
           "output_dim": 0,
           "weight_loader": self.weight_loader,
       })
```

【文件位置】[vllm/model_executor/layers/linear.py](gh-file:vllm/model_executor/layers/linear.py#L290-L303)

这里的 `output_dim=0`使得 bias 参数与权量一样，在加载时会按照 TP 分片进行分片读取。

## 3. 加载逻辑

`ColumnParallelLinear`实现了加载逻辑，直接选取对应的片段存到本地变量中：

```python
   def load_column_parallel_weight(self, loaded_weight: torch.Tensor):
       tp_rank = get_tensor_model_parallel_rank()
       shard_size = self.data.shape[self.output_dim]
       loaded_weight = loaded_weight.narrow(self.output_dim,
                                            tp_rank * shard_size, shard_size)
       assert self.data.shape == loaded_weight.shape
       self.data.copy_(loaded_weight)
```

【文件位置】[vllm/model_executor/parameter.py](gh-file:vllm/model_executor/parameter.py#L96-L110)

当 bias 被加载时，变量的 `output_dim` 为 0，每个 TP rank 只读入自己当前的片段，这样可以避免多余的内存占用和网络传输。

## 4. WQKV 的权量加载

在 TP 环境中，`wqkv`的主要权量也通过
`load_qkv_weight`加载，代码位置如下：

```python
   param.load_qkv_weight(loaded_weight=loaded_weight,
                         num_heads=self.num_kv_head_replicas,
                         shard_id=loaded_shard_id,
                         shard_offset=shard_offset,
                         shard_size=shard_size)
```

【文件位置】[vllm/model_executor/layers/linear.py](gh-file:vllm/model_executor/layers/linear.py#L935-L947)

这个加载过程也对应分片读取、仅加载当前 rank 的部分权量或 bias 。

## 5. 与 weight loader v2 的交互

在某些量化方法下，`QKVParallelLinear` 会使用 *weight loader v2*。此时
`self.weight_loader` 被自动替换为 `weight_loader_v2`，因此 bias 的加载也
会走新版流程。代码片段如下：

```python
    self.quant_method.create_weights(
        layer=self,
        input_size_per_partition=self.input_size_per_partition,
        output_partition_sizes=self.output_partition_sizes,
        input_size=self.input_size,
        output_size=self.output_size,
        params_dtype=self.params_dtype,
        weight_loader=(
            self.weight_loader_v2 if self.quant_method.__class__.__name__
            in WEIGHT_LOADER_V2_SUPPORTED else self.weight_loader))
    ...
    set_weight_attrs(self.bias, {
        "output_dim": 0,
        "weight_loader": self.weight_loader,
    })
```

`weight_loader_v2` 本质上调用 `param.load_column_parallel_weight`
来截取当前 TP 分片的数据：

```python
def weight_loader_v2(self, param: Parameter, loaded_weight: torch.Tensor):
    if len(loaded_weight.shape) == 0:
        assert loaded_weight.numel() == 1
        loaded_weight = loaded_weight.reshape(1)
    param.load_column_parallel_weight(loaded_weight=loaded_weight)
```

【文件位置】[vllm/model_executor/layers/linear.py](gh-file:vllm/model_executor/layers/linear.py#L412-L479)

这样，当启用 weight loader v2 时，bias 也会由 `load_column_parallel_weight`
负责切片加载，与权重量保持一致。

## 6. 总结

简言之，当使用 TP 时，`wqkv`的 bias 参数会在初始化时指定 `output_dim=0`，
而加载时通过 `load_column_parallel_weight`根据当前分片 id
截取抽取对应的权量分位，以此完成在 TP 环境中的 bias 缓存和加载。
