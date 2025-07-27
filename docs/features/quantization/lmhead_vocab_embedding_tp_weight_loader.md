---
title: Loading LMHead and vocab embeddings with TP
---

This guide describes how the tensor-parallel (TP) versions of
`VocabParallelEmbedding` and `ParallelLMHead` load their weights using the
built‑in weight loader. The Qwen2 model is used as the example.

## Qwen2 model construction

Both the token embedding layer and the LM head are sharded across TP
ranks. They are constructed using `VocabParallelEmbedding` and
`ParallelLMHead`:

```python
self.embed_tokens = VocabParallelEmbedding(
    config.vocab_size,
    config.hidden_size,
    quant_config=quant_config,
    prefix=f"{prefix}.embed_tokens",
)
...
self.lm_head = ParallelLMHead(
    config.vocab_size,
    config.hidden_size,
    quant_config=quant_config,
    prefix=maybe_prefix(prefix, "lm_head"),
)
```

{cite:3cb583L303-L310}
{cite:071594L454-L458}

## Registering the weight loader

`VocabParallelEmbedding` registers a custom `weight_loader` when creating
its parameters. This loader slices the incoming tensor according to the
TP shard and copies it into the parameter:

```python
self.quant_method.create_weights(
    self,
    self.embedding_dim,
    [self.num_embeddings_per_partition],
    self.embedding_dim,
    self.num_embeddings_padded,
    params_dtype=params_dtype,
    weight_loader=self.weight_loader,
)
```

{cite:37b2f0L266-L272}

## Loading weights

During `AutoWeightsLoader.load_weights`, each parameter's `weight_loader`
is invoked instead of the default loader:

```python
weight_loader = getattr(param, "weight_loader", default_weight_loader)
weight_loader(param, weight_data)
```

{cite:b911eeL166-L168}

For embeddings, this method ultimately calls
`VocabParallelEmbedding.weight_loader`:

```python
def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
    output_dim = getattr(param, "output_dim", None)
    ...
    start_idx = self.shard_indices.org_vocab_start_index
    shard_size = self.shard_indices.org_vocab_end_index - start_idx
    loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
    param[:loaded_weight.shape[0]].data.copy_(loaded_weight)
    param[loaded_weight.shape[0]:].data.fill_(0)
```

{cite:fe0c0eL351-L404}

This function computes the correct shard indices for the current TP rank
and loads only that portion of the checkpoint tensor. The LM head reuses
the same loader because it inherits from `VocabParallelEmbedding`.

## Summary

1. `VocabParallelEmbedding` and `ParallelLMHead` are used for TP
   embeddings and LM logits.
2. `create_weights` attaches `weight_loader` so each parameter knows how
   to slice its checkpoint tensor.
3. `AutoWeightsLoader` retrieves this loader and calls it when processing
   the state dict.
4. `VocabParallelEmbedding.weight_loader` copies only the relevant shard,
   ensuring embeddings and LM head weights are correctly initialized on
   each rank.
