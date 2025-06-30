# FP8 Scale Loading Discussion

## 背景问题
- 在 vLLM 中使用 FP8 块量化时，scale 的名称以及其与权重的映射方式如何？
- 共用同一 `weight_loader` 时，scale 的加载流程是什么？
- 是否可以在 `LLMEngine.load_weights` 中传入生成器以加载 blockwise FP8 scale？

## 核心发现
1. `Fp8LinearMethod.create_weights()` 在 block-wise 量化下会创建 `BlockQuantScaleParameter`，并以 `weight_scale_inv` 名称注册到层中。
2. 每个参数都会保存 `weight_loader`，在加载时由 `AutoWeightsLoader` 根据参数名调用对应的 `weight_loader`。
3. `ColumnParallelLinear.weight_loader_v2` 识别 `BlockQuantScaleParameter` 后，依据量化块大小计算 `shard_offset` 与 `shard_size`，然后调用 `load_merged_column_weight` 将分片复制到参数 tensor。
4. `_ColumnvLLMParameter.load_merged_column_weight` 根据偏移和大小截取张量，完成拷贝，使得块量化 scale 在显存中正确就位。
5. 通过上述逻辑，`LLMEngine.load_weights` 支持从生成器读取数据并同时加载权重与其 block-wise scale。

## 结论
- FP8 block-wise 量化时 scale 的参数名为 **`weight_scale_inv`**，与权重共用同一加载器，但加载时会按块尺寸计算偏移和分片大小。

