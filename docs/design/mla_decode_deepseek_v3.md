---
title: DeepSeek V3 MLA Decode Notes
---
[](){ #mla-decode-deepseek }

This document records key details discussed in a prior conversation about the
Masked Latent Attention (MLA) implementation used by DeepSeek V3 within vLLM.
It focuses on the decode path and how low‑rank queries and values are handled.

## Shape Symbols

In this note we use the following symbols to describe tensor shapes:

| Symbol | Meaning |
| ------ | ------- |
| `B`    | batch size (decode tokens) |
| `N`    | number of attention heads |
| `P`    | query/key head dim without RoPE |
| `R`    | query/key head dim with RoPE |
| `L`    | latent rank for key/value projections |
| `V`    | value head dimension |

## Weight Layout

The weight matrices involved in MLA are packed as follows:

- `q_b_proj` contains `[W_UQ; W_QR]` per head. It projects latent queries
  `q_c` to the *no‑PE* part `q_nope` and the RoPE part `q_pe`.
- `kv_b_proj` contains `[W_UK; W_UV]` per head. After loading, the code splits
  it into the low‑rank key weight `W_UK` and value up‑projection weight `W_UV`.
  The relevant processing is in `MLACommonImpl.process_weights_after_loading`:

```python
kv_b_proj_weight = get_and_maybe_dequant_weights(self.kv_b_proj).T
kv_b_proj_weight = kv_b_proj_weight.view(
    self.kv_lora_rank,
    self.num_heads,
    self.qk_nope_head_dim + self.v_head_dim,
)
W_UK, W_UV = kv_b_proj_weight.split(
    [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
self.W_UV = W_UV.transpose(0, 1)     # (N, L, V)
self.W_UK_T = W_UK.permute(1, 2, 0)  # (N, P, L)
```

【F:vllm/v1/attention/backends/mla/common.py†L730-L764】

## Query Processing in Decode

During decode the query tensor `decode_q` has shape `(B, N, P + R)` where
`B` is the batch size, `N` is the head count, `P` is the no‑RoPE dimension
and `R` is the RoPE dimension. The code splits and multiplies the no‑PE part
with `W_UK_T` to obtain a low‑rank query:

```python
decode_q_nope, decode_q_pe = decode_q.split(
    [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
# (B, N, P) -> (N, B, P)
decode_q_nope = decode_q_nope.transpose(0, 1)
# (N, B, P) x (N, P, L) -> (N, B, L)
decode_ql_nope = torch.bmm(decode_q_nope, self.W_UK_T)
# (N, B, L) -> (B, N, L)
decode_ql_nope = decode_ql_nope.transpose(0, 1)
```

【F:vllm/v1/attention/backends/mla/common.py†L952-L966】

The low‑rank query `decode_ql_nope` (shape `(B, N, L)`) and the RoPE portion
`decode_q_pe` are then passed to `_forward_decode`.

## FlashMLA Decode Path

For the FlashMLA backend, `_forward_decode` concatenates the low‑rank query with
its RoPE part, adds a sequence length dimension of `1`, and calls
`flash_mla_with_kvcache`. The kernel returns an output in the latent value
rank. Afterwards `_v_up_proj` applies `W_UV` to map the result to the final
value dimension:

```python
q = torch.cat([q_nope, q_pe], dim=-1).unsqueeze(1)
o, _ = flash_mla_with_kvcache(
    q=q,
    k_cache=kv_c_and_k_pe_cache.unsqueeze(-2),
    block_table=attn_metadata.decode.block_table,
    cache_seqlens=attn_metadata.decode.seq_lens,
    head_dim_v=self.kv_lora_rank,
    tile_scheduler_metadata=attn_metadata.decode.tile_scheduler_metadata,
    num_splits=attn_metadata.decode.num_splits,
    softmax_scale=self.scale,
    causal=True,
)
return self._v_up_proj(o)
```

【F:vllm/v1/attention/backends/mla/flashmla.py†L164-L180】

The helper `_v_up_proj` performs the batched matrix multiplication with `W_UV`:

```python
# (B, N, L) -> (N, B, L)
x = x.view(-1, self.num_heads, self.kv_lora_rank).transpose(0, 1)
# (N, B, L) x (N, L, V) -> (N, B, V)
x = torch.bmm(x, self.W_UV)
# (N, B, V) -> (B, N * V)
return x.transpose(0, 1).reshape(-1, self.num_heads * self.v_head_dim)
```

【F:vllm/v1/attention/backends/mla/common.py†L692-L703】

## Key Takeaways

- `W_UQ` is used once to expand latent queries into `q_nope` and `q_pe`; the
  low‑rank query derived from `q_nope` is **not** up‑projected back.
- `kv_b_proj` stores both `W_UK` and `W_UV`; after loading, `W_UK` is used to
  produce low‑rank queries and `W_UV` is applied **after** the FlashMLA kernel.
- `flash_mla_with_kvcache` itself operates in the latent space and outputs
  tensors of size `kv_lora_rank`; the value up‑projection happens later in
  `_v_up_proj`.
- Keeping queries and keys in low rank during decode reduces memory bandwidth
  while the final value up‑projection yields full‑dimensional results.

These details clarify how MLA handles query and value projections in the
DeepSeek V3 decode workflow.
