---
title: Understanding FP8 Blockwise Weight Loading
---
[](){ #fp8-blockwise-weight-loading }

This document summarizes how vLLM loads FP8 blockwise weights and their corresponding scale tensors when using fused layers like `qkv_proj` and `gate_up_proj`. The explanation is based on a discussion about the `load_weights` implementation and the role of `stacked_params_mapping`.

## Background

When using FP8 quantization with blockwise weight scales, vLLM represents fused linear layers such as the self-attention projection (`QKVParallelLinear`) or the MLP projection (`MergedColumnParallelLinear`). Each fused layer expects its weights and scale tensors to be provided as shards from the HuggingFace checkpoint.

## Mapping Checkpoint Names

During weight loading, vLLM maintains a mapping list `stacked_params_mapping`. The mapping pairs substrings of the original HuggingFace parameter names with the fused parameter names and indicates the shard id:

```python
stacked_params_mapping = [
    (".qkv_proj", ".q_proj", "q"),
    (".qkv_proj", ".k_proj", "k"),
    (".qkv_proj", ".v_proj", "v"),
    (".gate_up_proj", ".gate_proj", 0),
    (".gate_up_proj", ".up_proj", 1),
]
```

The replacement applies to all parameters whose names contain these substrings, including FP8 scale tensors such as `q_proj.weight_scale_inv`. Therefore `q_proj.weight_scale_inv` is remapped to `qkv_proj.weight_scale_inv` and is loaded into the fused layer just like the weight tensor itself.

Special handling in `maybe_remap_kv_scale_name` further covers names like `k_scale` and `v_scale` when present.

## Loading Blockwise Scales

Fused layers allocate `BlockQuantScaleParameter` objects to hold FP8 scales. When the loader encounters a tensor of this type, it calculates the shard offset and size using the configured block sizes and the tensor-parallel world size:

```python
weight_block_size = self.quant_method.quant_config.weight_block_size
block_n, _ = weight_block_size
shard_offset = ((sum(self.output_sizes[:loaded_shard_id]) + block_n - 1) // block_n) // tp_size
shard_size = ((self.output_sizes[loaded_shard_id] + block_n - 1) // block_n // tp_size)
```

The loaded slice is then copied into the shard of the fused layer.

## Required Checkpoint Format

For each unfused linear layer in the HuggingFace checkpoint, an additional tensor named `weight_scale_inv` must be provided. The tensor shape is:

```
[ceil(output_size / block_n), ceil(input_size / block_k)]
```

where `block_n` and `block_k` are the block dimensions used by FP8 quantization. This ensures that when vLLM remaps the parameter names, both the weights and their corresponding scales are transferred correctly into the fused representation.

## Summary

By rewriting parameter names through `stacked_params_mapping`, vLLM seamlessly redirects both weights and FP8 blockwise scale tensors from HuggingFace checkpoints to the corresponding fused layers. The `BlockQuantScaleParameter` objects within these layers manage the per-block scale slices, enabling correct loading of gate-up or qkv weights under FP8 blockwise quantization.

