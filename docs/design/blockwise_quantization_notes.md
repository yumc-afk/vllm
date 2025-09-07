---
title: Blockwise Quantization Notes
---

This document summarizes how vLLM handles blockwise FP8 quantization and why certain layers such as DeepSeek V3's `kv_a_proj_with_mqa` work even when their dimensions are not multiples of the 128x128 block size.

## Blocksize checks in `Fp8LinearMethod.create_weights`

`Fp8LinearMethod.create_weights` validates tensor-parallel partitions when block quantization is enabled:

```python
if self.block_quant:
    tp_size = get_tensor_model_parallel_world_size()
    assert self.quant_config.weight_block_size is not None
    layer.weight_block_size = self.quant_config.weight_block_size
    block_n, block_k = (
        self.quant_config.weight_block_size[0],
        self.quant_config.weight_block_size[1],
    )
    # Required by row parallel
    if (tp_size > 1
            and input_size // input_size_per_partition == tp_size
            and input_size_per_partition % block_k != 0):
        raise ValueError(
            f"Weight input_size_per_partition = "
            f"{input_size_per_partition} is not divisible by "
            f"weight quantization block_k = {block_k}.")
    # Required by column parallel or enabling merged weights
    if (tp_size > 1 and output_size // output_size_per_partition
            == tp_size) or len(output_partition_sizes) > 1:
        for output_partition_size in output_partition_sizes:
            if output_partition_size % block_n != 0:
                raise ValueError(
                    f"Weight output_partition_size = "
                    f"{output_partition_size} is not divisible by "
                    f"weight quantization block_n = {block_n}.")
```

For row-parallel layers, the input dimension of each partition must be divisible by `block_k`. For column-parallel layers, every output partition must be divisible by `block_n`. However, these checks do **not** run for `ReplicatedLinear` because its weights are not partitioned; `input_size // input_size_per_partition` and `output_size // output_size_per_partition` both equal `1`.

## Matmul with non-divisible sizes

During execution, block‑quantized matmuls rely on `prepare_block_fp8_matmul_inputs`, which uses ceil‑division to align scale tensors with the weight shapes:

```python
def prepare_block_fp8_matmul_inputs(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.float16,
) -> tuple[int, int, int, torch.Tensor]:
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]

    assert A.shape[-1] == B.shape[-1]
    assert A.shape[:-1] == As.shape[:-1]
    assert A.is_contiguous()
    assert triton.cdiv(A.shape[-1], block_k) == As.shape[-1]

    M = A.numel() // A.shape[-1]

    assert B.ndim == 2
    assert B.is_contiguous()
    assert Bs.ndim == 2
    N, K = B.shape
    assert triton.cdiv(N, block_n) == Bs.shape[0]
    assert triton.cdiv(K, block_k) == Bs.shape[1]

    C_shape = A.shape[:-1] + (N, )
    C = A.new_empty(C_shape, dtype=output_dtype)

    return M, N, K, C
```

`triton.cdiv` ensures that scale tensors allocate enough blocks even when a weight dimension is not a multiple of the block size.

`apply_w8a8_block_fp8_linear` selects optimized kernels only when the weight shape matches the block size. Otherwise it falls back to a generic implementation:

```python
if current_platform.is_cuda():
    if current_platform.has_device_capability(100):
        use_cutlass = cutlass_block_fp8_supported and (
            cdiv(weight.shape[0], 128) == weight_scale.shape[0]
            and cdiv(weight.shape[1], 128) == weight_scale.shape[1])
    else:
        # TODO: update this after switching to public sm90 block scale gemm
        # as it also supports weight.shape % 128 != 0
        use_cutlass = cutlass_block_fp8_supported and (
            weight.shape[0] % 128 == 0 and weight.shape[1] % 128 == 0)
else:
    use_cutlass = False
```

## Example: DeepSeek V3

DeepSeek V3 introduces a 64‑dimensional RoPE component. The `kv_a_proj_with_mqa` layer combines this with a lora rank (512 + 64 → 576) and is implemented using `ReplicatedLinear`:

```python
self.kv_a_proj_with_mqa = ReplicatedLinear(
    self.hidden_size,
    self.kv_lora_rank + self.qk_rope_head_dim,
    bias=False,
    quant_config=quant_config,
    prefix=f"{prefix}.kv_a_proj_with_mqa")
```

Because `ReplicatedLinear` keeps the full weight on every GPU, the divisibility checks above do not trigger. Quantization proceeds with an extra block worth of scales, and the kernel falls back when necessary. Hence layers with output size 576 load and run without errors.

In contrast, row/column parallel layers and MoE blocks enforce divisibility. For example, MoE gating uses:

```python
# NOTE: To ensure proper alignment of the block-wise quantization
# scales, the output_size of the weights for both the gate and up
# layers must be divisible by block_n.
# Required by column parallel or enabling merged weights
if intermediate_size_per_partition % block_n != 0:
    raise ValueError(
        f"The output_size of gate's and up's weight = "
        f"{intermediate_size_per_partition} is not divisible by "
        f"weight quantization block_n = {block_n}.")
if (tp_size > 1
        and intermediate_size_per_partition % block_k != 0):
    # Required by row parallel
    raise ValueError(
        f"The input_size of down's weight = "
        f"{intermediate_size_per_partition} is not divisible by "
        f"weight quantization block_k = {block_k}.")
```

If the intermediate size per partition is not divisible by the block dimensions, initialization raises `ValueError`.

## Summary

- Block quantization checks only apply to partitioned (row or column parallel) weights.
- `ReplicatedLinear` layers skip these checks and can have dimensions that are not multiples of `[block_n, block_k]`.
- Matmul utilities allocate scales using ceil‑div, so kernels still work with non-divisible sizes, though they may fall back to a slower implementation.
- MoE and other partitioned layers must strictly satisfy the blocksize requirements.

## Codemap

- `vllm/model_executor/layers/quantization/fp8.py` defines `Fp8LinearMethod.create_weights` and the kernel selection logic for `apply_w8a8_block_fp8_linear`.
- `vllm/model_executor/layers/quantization/deepgemm.py` implements `prepare_block_fp8_matmul_inputs`.
- `vllm/model_executor/layers/quantization/utils/fp8_utils.py` wraps the CUDA/Cutlass operators for blockwise GEMM.
- `vllm/lora/models.py` instantiates `ReplicatedLinear` in DeepSeek V3.
