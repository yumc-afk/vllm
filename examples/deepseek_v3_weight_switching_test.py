"""
Test script for vLLM online weight switching with DeepSeek V3 mini model.

This script demonstrates how to use the online weight switching functionality
in vLLM's AsyncEngine to dynamically update model weights during inference
without engine restarts.

Usage:
    1. First generate two mini DeepSeek V3 models using the Pai-Megatron-Patch script
    2. Convert them to HuggingFace format
    3. Run this script to test weight switching between the two models
"""

import os
import sys
import time
import torch
import asyncio
from typing import Dict, List, Optional, AsyncIterator

from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.outputs import RequestOutput


async def test_weight_switching():
    """Test online weight switching with DeepSeek V3 mini models."""
    
    model_path_1 = os.environ.get("MODEL_PATH_1", "./mini_model_1")
    
    model_path_2 = os.environ.get("MODEL_PATH_2", "./mini_model_2")
    
    print(f"Initializing AsyncLLMEngine with model: {model_path_1}")
    engine_args = AsyncEngineArgs(model=model_path_1)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    test_prompts = [
        "Once upon a time",
        "The quick brown fox",
        "In a galaxy far far away",
    ]
    
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=20,
    )
    
    print("\n=== Running inference with the first model ===")
    first_model_outputs = []
    for i, prompt in enumerate(test_prompts):
        request_id = f"request1_{i}"
        print(f"Processing prompt {i+1}: {prompt}")
        generator = engine.generate(prompt, sampling_params, request_id)
        async for result in generator:
            if result.finished:
                output_text = result.outputs[0].text
                first_model_outputs.append(output_text)
                print(f"Output: {output_text}\n")
                break
    
    print("\n=== Updating model weights from file ===")
    start_time = time.time()
    await engine.update_model_weights(model_path_2)
    end_time = time.time()
    print(f"Weight update took {end_time - start_time:.4f} seconds")
    
    print("\n=== Running inference with the updated model ===")
    second_model_outputs = []
    for i, prompt in enumerate(test_prompts):
        request_id = f"request2_{i}"
        print(f"Processing prompt {i+1}: {prompt}")
        generator = engine.generate(prompt, sampling_params, request_id)
        async for result in generator:
            if result.finished:
                output_text = result.outputs[0].text
                second_model_outputs.append(output_text)
                print(f"Output: {output_text}\n")
                break
    
    print("\n=== Comparing outputs ===")
    for i, (first, second) in enumerate(zip(first_model_outputs, second_model_outputs)):
        print(f"Prompt {i+1}:")
        print(f"  First model: {first}")
        print(f"  Second model: {second}")
        if first != second:
            print("  Outputs differ ✓")
        else:
            print("  Outputs are identical ✗")
    
    print("\n=== Testing direct state dict update ===")
    
    print("Creating modified state dict...")
    
    original_state_dict = {
        "decoder.layers.0.self_attention.query_key_value.weight": torch.randn(768, 768) * 1.5,
        "decoder.layers.0.mlp.fc1.weight": torch.randn(3072, 768) * 1.5
    }
    
    print("Updating weights using state dict...")
    start_time = time.time()
    await engine.update_model_weights_from_state_dict(original_state_dict)
    end_time = time.time()
    print(f"State dict update took {end_time - start_time:.4f} seconds")
    
    print("\n=== Running inference with state dict updated model ===")
    third_model_outputs = []
    for i, prompt in enumerate(test_prompts):
        request_id = f"request3_{i}"
        print(f"Processing prompt {i+1}: {prompt}")
        generator = engine.generate(prompt, sampling_params, request_id)
        async for result in generator:
            if result.finished:
                output_text = result.outputs[0].text
                third_model_outputs.append(output_text)
                print(f"Output: {output_text}\n")
                break
    
    print("\n=== Comparing outputs with state dict updated model ===")
    for i, (second, third) in enumerate(zip(second_model_outputs, third_model_outputs)):
        print(f"Prompt {i+1}:")
        print(f"  Before update: {second}")
        print(f"  After update: {third}")
        if second != third:
            print("  Outputs differ ✓")
        else:
            print("  Outputs are identical ✗")
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_weight_switching())
