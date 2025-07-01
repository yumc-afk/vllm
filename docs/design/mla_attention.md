# Memory Latent Attention 指南

本文档总结了我们在对话中讨论的 MLA(Memory Latent Attention) 实现细节，包括 *compute friendly* 与 *data-movement friendly* 两种计算路径的差异，以及在 TP/SP 环境下的行为。

## 背景与符号

MLA 在 `vllm/attention/backends/mla/common.py` 中实现，核心理念是用单个 latent 向量表示每个 token 的 KV cache，并根据阶段选择不同的计算方式。常用符号如下：

- `Sq`: 当前查询序列长度
- `Skv`: 总的 KV 序列长度
- `N`: 注意力头数
- `Lkv`: KV latent 维度 (DeepSeek V3 中为 512)
- `P`: 不经 RoPE 的维度 (128)
- `R`: 经 RoPE 的维度 (64)
- `V`: V 的 head dim (128)

## Compute Friendly（Prefill）

prefill 阶段 `Sq/Skv` 比较小，先将 Q/K/V 完全上投影后再执行標準 MHA。代码示例：

```python
q_c      = h_t @ W_DQ
q_nope   = (q_c @ W_UQ).view(Sq, N, P)
q_pe     = RoPE(q_c @ W_QR).view(Sq, N, R)
new_kv_c = h_t @ W_DKV
new_k_pe = RoPE(h_t @ W_KR)
kv_c     = torch.cat([new_kv_c, cache_kv_c], dim=0)
k_pe     = torch.cat([new_k_pe, cache_k_pe], dim=0)
k_nope   = (kv_c @ W_UK.view(Lkv, N * P)).view(Skv, N, P)
v        = (kv_c @ W_UV.view(Lkv, N * V)).view(Skv, N, V)

# MHA with QK headdim = P + R
spda_o = scaled_dot_product_attention(
    torch.cat([q_nope, q_pe], dim=-1),
    torch.cat([k_nope, k_pe.unsqueeze(1).expand(-1, N, -1)], dim=-1),
    v,
)
return spda_o @ W_O
```

上述代码位于 `vllm/attention/backends/mla/common.py`【F:vllm/attention/backends/mla/common.py†L60-L90】。

prefill 路径计算量较大，但对 KV cache 只需一次上投影，数据移动较少，适合序列长度相近的阶段。

## Data-Movement Friendly（Decode）

decode 阶段 `Sq/Skv` 比较大，为减少跨 GPU 数据移动，只保存 latent KV，在更宽的维度 `Lkv + R` 上计算注意力。示例代码：

```python
# MQA with QK headdim = Lkv + R
#           V headdim = Lkv
#      spda_o shape [Sq, N, Lkv]
# NOTE: this is less compute-friendly since Lkv > P
#       but is more data-movement friendly since its MQA vs MHA
spda_o = scaled_dot_product_attention(
    torch.cat([ql_nope, q_pe], dim=-1),
    torch.cat([kv_c, k_pe], dim=-1),
    kv_c,
)
```

摘自 `vllm/attention/backends/mla/common.py`【F:vllm/attention/backends/mla/common.py†L98-L118】。
由于 `Lkv` 大于 `P`，此路径在更大的头维上执行 softmax 和乘法，单次计算量更高，但省去了对历史 KV 的持续上投影，也减少了显存占用和 GPU 间通信，在长序列解码或分布式场景中更具优势。

## TP/SP 环境下的行为

- MLA 解码时等同于 MQA，只需要一个 KV 头。`get_num_kv_heads` 会在使用 MLA 时忽略 tensor parallel 的头复制，降低通信量。
- 若启用 `enable_async_tp`，系统会自动开启 sequence parallelism，在编译阶段通过 `ReduceScatter`/`AllGather` 等操作协调计算与通信。
- `platforms/cuda.py` 中若启用 MLA，则默认选择 FlashMLA 后端，并将 KV cache block size 固定为 64，以适配底层实现。

## 与一般 GQA 的区别

普通的 Grouped‑Query Attention 在解码阶段需要维护多组 KV 头，并始终在较小的 head dim 上做注意力；MLA 在 decode 时采用单一 latent 表示，通过更大的 `Lkv+R` 维度计算，减少显存与数据搬运。prefill 阶段则回到标准 MHA，充分利用算力。

## 代码位置（Codemap）

```
vllm/
├── attention/
│   ├── backends/
│   │   ├── mla/common.py      # MLA 主要实现和注释
│   │   └── flashmla.py        # FlashMLA 后端封装
│   ├── ops/flashmla.py        # 低层算子接口
│   └── selector.py            # 根据配置选择 backend
├── compilation/sequence_parallelism.py  # SP/TP 编译 pass
└── platforms/cuda.py                    # 自动选择 FlashMLA
```

以上内容帮助理解 MLA 的两种计算路径及其在分布式环境下的优势。
