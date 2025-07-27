---
title: Loading wqkv weight_scale_inv for FP8 TP
---

This guide explains how the `weight_scale_inv` parameter of the fused
qkv projection (named `qkv_proj` in the Qwen2 model) is loaded when
using FP8 quantization with tensor parallelism (TP).  The same loading
flow applies to models that expose `wqkv` parameters.

## Overview

1. **Model initialization** – Qwen2 creates a `QKVParallelLinear`
   layer for `qkv_proj` with the provided `quant_config`.
2. **Parameter creation** – `Fp8LinearMethod.create_weights` registers
   the `weight` tensor and, when block-wise quantization is enabled,
   a `weight_scale_inv` parameter.
3. **Weight loading** – during `AutoWeightsLoader.load_weights`, each
   parameter receives a state‑dict tensor. For TP layers the loader
   splits the tensor according to the shard id before copying it into
   the parameter.
4. **Post‑processing** – after all weights are loaded,
   `process_weights_after_loading` is called. For FP8 this converts the
   loaded scales and ensures the tensors are in the correct layout for
   inference.

The following sections walk through these steps with code references.

## Qwen2 attention

The attention block constructs the fused QKV projection:

```python
self.qkv_proj = QKVParallelLinear(
    hidden_size,
    self.head_dim,
    self.total_num_heads,
    self.total_num_kv_heads,
    bias=True,
    quant_config=quant_config,
    prefix=f"{prefix}.qkv_proj",
)
```

{cite:bcd2e8L136-L144}

## Parameter registration

During initialization of `QKVParallelLinear`, the FP8 quantization
method creates the scale parameter:

```python
scale = BlockQuantScaleParameter(
    data=torch.empty(
        (output_size_per_partition + block_n - 1) // block_n,
        (input_size_per_partition + block_k - 1) // block_k,
        dtype=torch.float32,
    ),
    input_dim=1,
    output_dim=0,
    weight_loader=weight_loader,
)
scale[:] = torch.finfo(torch.float32).min
set_weight_attrs(scale, {"scale_type": "weight_scale"})
# The weight_scale_inv name is intentional for deepseekv3
layer.register_parameter("weight_scale_inv", scale)
```

{cite:00e5c0L280-L299}

## Loading shards

`QKVParallelLinear.weight_loader_v2` handles loading each tensor
shard according to the TP rank. For scale parameters it computes the
correct offset using the block size:

```python
if isinstance(param, BlockQuantScaleParameter):
    weight_block_size = self.quant_method.quant_config.weight_block_size
    block_n, _ = weight_block_size[0], weight_block_size[1]
    shard_offset = ((sum(self.output_sizes[:loaded_shard_id]) + block_n - 1)
                    // block_n) // tp_size
    shard_size = ((self.output_sizes[loaded_shard_id] + block_n - 1)
                  // block_n // tp_size)
else:
    shard_offset = sum(self.output_sizes[:loaded_shard_id]) // tp_size
    shard_size = self.output_sizes[loaded_shard_id] // tp_size
param.load_merged_column_weight(
    loaded_weight=loaded_weight,
    shard_id=loaded_shard_id,
    shard_offset=shard_offset,
    shard_size=shard_size,
)
```

{cite:c4604eL730-L782}

`load_merged_column_weight` on `BlockQuantScaleParameter` copies the
correct slice into the parameter tensor while respecting TP partitioning.

## Post‑processing after loading

Once all state‑dict tensors are loaded, weight post‑processing runs:

```python
for _, module in model.named_modules():
    quant_method = getattr(module, "quant_method", None)
    if isinstance(quant_method, QuantizeMethodBase):
        with device_loading_context(module, target_device):
            quant_method.process_weights_after_loading(module)
```

{cite:c6be5cL96-L113}

For FP8 and block quantization this method converts the weight and scale
buffers and reassigns them as standard `Parameter` objects:

```python
if self.block_quant:
    weight, weight_scale_inv, _ = normalize_e4m3fn_to_e4m3fnuz(
        weight=layer.weight,
        weight_scale=layer.weight_scale_inv)
    weight = self._maybe_pad_weight(weight)
    layer.weight = Parameter(weight, requires_grad=False)
    layer.weight_scale_inv = Parameter(weight_scale_inv,
                                       requires_grad=False)
```

{cite:00e5c0L311-L331}

At this point the `weight_scale_inv` tensor for each TP shard is ready
for inference.

## Summary

1. `Qwen2Attention` builds `qkv_proj` with `QKVParallelLinear`.
2. `Fp8LinearMethod.create_weights` registers `weight_scale_inv` when
   block quantization is enabled.
3. `AutoWeightsLoader` calls `weight_loader_v2` which partitions the
   scale tensor by TP rank and copies it into the parameter.
4. `process_weights_after_loading` finalizes the FP8 tensors,
   converting the scales if necessary.

Understanding this flow helps debug issues with FP8 checkpoint loading
and ensures the `weight_scale_inv` parameters are correctly initialized
for tensor parallel inference.
