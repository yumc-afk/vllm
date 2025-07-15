---
title: FP8 Blockwise Guide
---
[](){ #fp8-blockwise-guide }

本指南介绍在启用 FP8 blockwise 量化时如何禁用 FP8 attention 与 KV cache 量化。

## 启用 FP8 blockwise 量化

在构建 `LLM` 时，设置 `quantization="fp8"` 并在 `quantization_config` 中提供 `weight_block_size` 参数即可启用块级 FP8 权重量化。例如：

```python
quant_config = {"weight_block_size": [64, 256]}
llm = LLM(model="your-model", quantization="fp8", quantization_config=quant_config)
```

## 关闭 FP8 attention 与 KV cache 量化

FP8 attention 与 KV cache 量化会在 `kv_cache_dtype` 为 `"fp8"` 时自动开启。若要保持权重量化为 FP8，但关闭这两项功能，可在创建 `LLM` 时将 `kv_cache_dtype` 设为 `"auto"` (或其他非 FP8 类型)：

```python
llm = LLM(model="your-model",
          quantization="fp8",
          quantization_config=quant_config,
          kv_cache_dtype="auto")
```

`kv_cache_dtype` 不是 FP8 时，attention 操作也不会进行 FP8 量化，从而实现仅使用 blockwise 权重量化的效果。
