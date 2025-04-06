# Testing Guide for vLLM Online Weight Switching

This guide provides instructions for testing the online weight switching functionality in vLLM. It includes both CPU-based validation tests and GPU-based functional tests.

## Prerequisites

- Python 3.8 or higher
- Git
- For GPU testing: CUDA 11.8 or higher and compatible GPU

## Installation

We provide an installation script that sets up the necessary dependencies for testing. The script creates a Python virtual environment and installs the required packages.

```bash
# Clone the repository if you haven't already
git clone https://github.com/yumc-afk/vllm.git
cd vllm

# Make the installation script executable
chmod +x scripts/install_test_dependencies.sh

# For CPU-only testing
./scripts/install_test_dependencies.sh --cpu

# For GPU testing
./scripts/install_test_dependencies.sh --gpu

# Activate the virtual environment
source venv/bin/activate
```

## CPU-Based Validation Tests

These tests validate the implementation logic without requiring GPU hardware. They use mock objects and static analysis to verify the correctness of the implementation.

### Running the Simplified Mock Tests

```bash
# Run the simplified mock tests
python simplified_mock_tests.py
```

This script tests the control flow of the weight switching implementation using mock objects. It verifies:
- Method calls propagate correctly through the class hierarchy
- Parameters are correctly passed through the call chain
- Requests are blocked during weight updates
- Error handling works correctly

### Running the Code Path Analysis

```bash
# Run the code path analysis
python code_path_analysis.py
```

This script analyzes all possible execution paths in the weight switching implementation to ensure correctness and identify potential issues.

### Running the Static Validation

```bash
# Run the static validation
python simplified_validation.py
```

This script performs static validation of the implementation, checking API consistency, type correctness, thread safety, and memory management.

## GPU-Based Functional Tests

These tests require GPU hardware and validate the actual functionality of the implementation.

### Running the Unit Tests

```bash
# Run the weight switching unit tests
pytest tests/async_engine/test_weight_switching.py -v
```

This runs the unit tests for the weight switching functionality, verifying that it works correctly with actual GPU operations.

### Running the DeepSeek V3 Example

```bash
# Run the DeepSeek V3 weight switching test
python examples/deepseek_v3_weight_switching_test.py
```

This example demonstrates the weight switching functionality with a mini DeepSeek V3 model. It shows how to:
1. Initialize the AsyncEngine with a model
2. Generate text with the initial model
3. Update the model weights
4. Generate text with the updated model

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'vllm'**
   - Make sure you've installed vLLM with `pip install -e .`
   - Ensure you've activated the virtual environment with `source venv/bin/activate`

2. **CUDA out of memory**
   - Reduce the model size or batch size
   - Free up GPU memory by closing other applications

3. **Test failures**
   - Check the error message for details
   - Ensure you're using the correct branch with the weight switching implementation
   - Verify that all dependencies are installed correctly

### Getting Help

If you encounter issues not covered in this guide, please:
1. Check the vLLM documentation
2. Open an issue on the GitHub repository
3. Contact the maintainers for assistance

## Conclusion

By following this guide, you should be able to test the online weight switching functionality in vLLM using both CPU-based validation tests and GPU-based functional tests. The CPU-based tests provide a way to validate the implementation logic without requiring GPU hardware, while the GPU-based tests validate the actual functionality with real hardware.
