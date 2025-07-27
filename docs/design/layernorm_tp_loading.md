# TP加载LayerNorm参数指南

在 vLLM 中使用张量并行（Tensor Parallelism，TP）时，线性层或嵌入层等大尺寸权重会
按照并行度进行切分，并分布到各个 GPU 上。相比之下，`nn.LayerNorm` 和 `RMSNorm`
的 `weight` 与 `bias` 参数非常小，并没有额外的 `weight_loader` 属性来指定特殊加载逻
辑，因此它们不会被切分，而是在所有 TP 进程中保持完整复制。

## 加载流程概述
1. `AutoWeightsLoader` 在遍历模型权重时，会为每个参数寻找 `weight_loader` 属性；若
   未指定，则使用 `default_weight_loader`。
2. 对线性层等需要切分的权重，`weight_loader` 会被设置为 `row_parallel_weight_loader`、
   `column_parallel_weight_loader` 等函数，用于将权重按并行度切片后再加载。
3. `LayerNorm` 参数未声明 `weight_loader`，因此会落入 `default_weight_loader`。它不会
   对权重做任何分片处理，而是直接拷贝完整张量到各个进程，使得所有 TP rank 加载同一
   份 LayerNorm 权重。

### 关键代码片段
来自`vllm/model_executor/models/utils.py`的`_load_param`实现：

```python
weight_loader = getattr(param, "weight_loader", default_weight_loader)
weight_loader(param, weight_data)
```

【F:vllm/model_executor/models/utils.py†L166-L168】

`default_weight_loader`位于`vllm/model_executor/model_loader/weight_utils.py`：

```python
def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    if param.numel() == 1 and loaded_weight.numel() == 1:
        param.data.fill_(loaded_weight.item())
    else:
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight)
```

【F:vllm/model_executor/model_loader/weight_utils.py†L604-L618】

因此LayerNorm权重会在每个TP rank上完整加载并保持一致。

## 结果影响
由于LayerNorm参数在TP下并未切分，各进程计算出的归一化结果保持一致，符合Transformer等模型的标准实现，也避免了由于切分带来的同步开销。
