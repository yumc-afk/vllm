# Online Weight Switching

This document describes the online weight switching functionality in vLLM's AsyncEngine, which enables dynamically updating model weights during inference without engine restarts.

## Overview

The online weight switching feature allows you to update model weights in a running vLLM AsyncEngine instance without having to restart the engine. This is particularly useful for integration with training pipelines like Megatron for PPO (Proximal Policy Optimization), where model weights need to be updated frequently during training.

## Key Features

- **Dynamic Weight Updates**: Update model weights during inference without restarting the engine
- **Thread Safety**: Thread-safe implementation with proper synchronization between weight updates and ongoing inference requests
- **Memory Efficiency**: In-place parameter updates with proper CUDA memory management
- **Multi-GPU Support**: Coordinated weight updates across distributed workers
- **Two Update Interfaces**:
  - File-based updates via model weights path
  - Direct GPU memory updates via state dictionary

## API Reference

### AsyncLLMEngine

```python
async def update_model_weights(self, model_weights_path: str) -> None:
    """Update model weights without restarting the engine.
    
    This method allows dynamically updating model weights during inference,
    enabling integration with training pipelines like Megatron for PPO.
    
    Args:
        model_weights_path: Path to the new model weights.
    """
```

```python
async def update_model_weights_from_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
    """Update model weights directly from a state dict without restarting the engine.
    
    This method allows for more efficient weight updates when the weights are already
    available in memory, such as during training loops.
    
    Args:
        state_dict: Dictionary mapping parameter names to tensor values.
    """
```

## Usage Examples

### File-Based Weight Updates

```python
import asyncio
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

async def update_weights_example():
    # Initialize the engine
    engine_args = AsyncEngineArgs(model="path/to/initial/model")
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    # Run inference with initial weights
    sampling_params = SamplingParams(temperature=0.8, max_tokens=20)
    request_id = "request1"
    generator = engine.generate("Once upon a time", sampling_params, request_id)
    async for result in generator:
        if result.finished:
            print(f"Initial output: {result.outputs[0].text}")
            break
    
    # Update model weights
    await engine.update_model_weights("path/to/new/model")
    
    # Run inference with updated weights
    request_id = "request2"
    generator = engine.generate("Once upon a time", sampling_params, request_id)
    async for result in generator:
        if result.finished:
            print(f"Updated output: {result.outputs[0].text}")
            break

if __name__ == "__main__":
    asyncio.run(update_weights_example())
```

### State Dict-Based Weight Updates

```python
import asyncio
import torch
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

async def update_weights_from_state_dict_example():
    # Initialize the engine
    engine_args = AsyncEngineArgs(model="path/to/initial/model")
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    # Create a modified state dict (for demonstration purposes)
    # In a real scenario, this would come from your training framework (e.g., Megatron)
    state_dict = {
        "decoder.layers.0.self_attention.query_key_value.weight": torch.randn(768, 768) * 1.5,
        "decoder.layers.0.mlp.fc1.weight": torch.randn(3072, 768) * 1.5
    }
    
    # Update model weights using state dict
    await engine.update_model_weights_from_state_dict(state_dict)
    
    # Run inference with updated weights
    sampling_params = SamplingParams(temperature=0.8, max_tokens=20)
    request_id = "request1"
    generator = engine.generate("Once upon a time", sampling_params, request_id)
    async for result in generator:
        if result.finished:
            print(f"Output: {result.outputs[0].text}")
            break

if __name__ == "__main__":
    asyncio.run(update_weights_from_state_dict_example())
```

## Integration with Megatron for PPO Training

The online weight switching functionality is designed to integrate with Megatron for PPO training. Here's a high-level overview of how to use this feature in a PPO training loop:

1. Initialize the vLLM AsyncEngine with your initial model weights
2. Run inference with the current model to generate responses
3. Compute rewards and update model weights using your PPO training framework
4. Use `update_model_weights_from_state_dict()` to update the model weights in the AsyncEngine
5. Repeat steps 2-4 for each training iteration

## Implementation Details

### Weight Switching Mechanism

The weight switching mechanism works by:

1. Temporarily blocking new inference requests during weight updates
2. Using `torch.no_grad()` to ensure efficient in-place parameter updates
3. Copying new parameter values to existing parameters using `param.copy_()`
4. Clearing CUDA cache after updates to free any temporary memory
5. Resuming inference requests with the updated weights

### Synchronization Strategy

The implementation ensures thread safety through:

1. Request blocking during weight updates
2. Health checks before and after weight updates
3. Proper error handling and recovery
4. Coordinated updates across distributed workers using collective RPC

### Memory Management

Memory efficiency is achieved by:

1. In-place parameter updates to avoid creating new tensors
2. Proper CUDA memory management with `torch.cuda.empty_cache()`
3. Reusing existing model structures rather than creating new ones
4. Minimizing temporary memory allocations during weight updates

## Testing

A comprehensive test script is available at `examples/deepseek_v3_weight_switching_test.py` that demonstrates how to use the online weight switching functionality with DeepSeek V3 mini models.

## Limitations and Future Work

- Currently supports same-process weight updates
- Future work will add support for cross-process updates via CUDA IPC
- Performance optimizations for very large models
- Support for partial weight updates (e.g., updating only specific layers)
