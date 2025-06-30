---
title: Block FP8
---
[](){ #block-fp8 }

Blockwise FP8 quantization stores model weights in 8-bit floating point format
while grouping them into fixed-size blocks. Each block shares a scaling factor
so that weights and activations can be efficiently dequantized during matrix
multiplication. This format is used by models such as DeepSeek V3.

## Usage

To run a checkpoint that was serialized in blockwise FP8 format you need to
provide a `Fp8Config` when constructing the model. The configuration enables the
weight block size and dynamic activation scaling:

```python
from vllm import LLM
from vllm.model_executor.layers.quantization.fp8 import Fp8Config

quant_config = Fp8Config(
    is_checkpoint_fp8_serialized=True,
    activation_scheme="dynamic",
    weight_block_size=[128, 128],
)
llm = LLM(model="deepseek-ai/DeepSeek-V3", quant_config=quant_config)
```

## Implementation Overview

- **Configuration** – Blockwise options are parsed in
  `Fp8Config` which validates the `weight_block_size` argument and enforces
  dynamic activation scaling【F:vllm/model_executor/layers/quantization/fp8.py†L55-L87】.
- **Linear Execution** – During inference
  `apply_w8a8_block_fp8_linear` performs the block FP8 matrix multiply and
  chooses between CUTLASS, Triton or DeepGEMM kernels depending on hardware
  support【F:vllm/model_executor/layers/quantization/utils/fp8_utils.py†L119-L211】.
- **Operator Bindings** – Python invokes custom ops such as
  `torch.ops._C.cutlass_scaled_mm` and
  `torch.ops.vllm.w8a8_block_fp8_matmul_deepgemm` through wrappers in
  `_custom_ops.py`【F:vllm/_custom_ops.py†L660-L736】.
- **Kernel Entry** – CUDA kernels are compiled in
  `scaled_mm_entry.cu`, which exposes capability queries like
  `cutlass_scaled_mm_supports_block_fp8`【F:csrc/quantization/cutlass_w8a8/scaled_mm_entry.cu†L120-L132】.
- **Blockwise Kernels** – The actual GEMM implementations live in
  `scaled_mm_blockwise_sm90_fp8.cu` for Hopper GPUs and
  `scaled_mm_blockwise_sm100_fp8.cu` for Blackwell GPUs【F:csrc/quantization/cutlass_w8a8/c3x/scaled_mm_blockwise_sm90_fp8.cu†L1-L24】【F:csrc/quantization/cutlass_w8a8/c3x/scaled_mm_blockwise_sm100_fp8.cu†L8-L23】.
- **Binding Registration** – Custom ops are registered in
  `torch_bindings.cpp` including the block FP8 capability check【F:csrc/torch_bindings.cpp†L474-L479】.

## Code Map

```
vllm/model_executor/layers/quantization/fp8.py        # Fp8Config and layer logic
vllm/model_executor/layers/quantization/utils/fp8_utils.py  # block_fp8 kernels
vllm/_custom_ops.py                                     # Python wrappers for C++ ops
csrc/quantization/cutlass_w8a8/scaled_mm_entry.cu       # capability checks
csrc/quantization/cutlass_w8a8/c3x/*blockwise_sm*_fp8.cu     # CUTLASS kernels
csrc/torch_bindings.cpp                                 # PyTorch custom op bindings
```

These components collectively enable blockwise FP8 execution in vLLM.
