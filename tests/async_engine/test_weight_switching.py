
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs


class SimpleModel(nn.Module):
    """A simple model for testing weight switching."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        
    def forward(self, x):
        return self.linear(x)


class TestWeightSwitching(unittest.TestCase):
    """Test cases for online weight switching functionality."""
    
    @patch("vllm.worker.model_runner.get_model_loader")
    @patch("vllm.engine.async_llm_engine.make_async")
    async def test_update_model_weights(self, mock_make_async, mock_get_model_loader):
        """Test that update_model_weights correctly updates model weights."""
        mock_async_func = MagicMock()
        mock_make_async.return_value = mock_async_func
        
        engine = MagicMock()
        async_engine = AsyncLLMEngine(engine)
        
        model_weights_path = "/path/to/model/weights"
        await async_engine.update_model_weights(model_weights_path)
        
        mock_async_func.assert_called_once_with(model_weights_path)
    
    @patch("vllm.model_executor.model_loader.get_model_loader")
    def test_model_runner_update_weights(self, mock_get_model_loader):
        """Test that GPUModelRunnerBase.update_model_weights updates model weights correctly."""
        from vllm.worker.model_runner import ModelRunner
        
        model = SimpleModel()
        original_weight = model.linear.weight.clone()
        
        mock_loader = MagicMock()
        mock_get_model_loader.return_value = mock_loader
        
        new_weight = original_weight + 1.0
        mock_loader.load_state_dict.return_value = {"linear.weight": new_weight}
        
        mock_vllm_config = MagicMock()
        
        with patch.object(ModelRunner, "__init__", return_value=None):
            runner = MagicMock(spec=ModelRunner)
            runner.model = model
            runner.vllm_config = mock_vllm_config
            
            from vllm.worker.model_runner import GPUModelRunnerBase
            runner.update_model_weights = GPUModelRunnerBase.update_model_weights.__get__(runner)
            
            runner.update_model_weights("/path/to/model/weights")
            
            torch.testing.assert_close(model.linear.weight, new_weight)
            
            mock_get_model_loader.assert_called_once()
            mock_loader.load_state_dict.assert_called_once()
    
    @patch("vllm.executor.executor_base.make_async")
    async def test_executor_update_weights(self, mock_make_async):
        """Test that ExecutorBase.update_model_weights coordinates updates across workers."""
        from vllm.executor.executor_base import ExecutorBase
        
        mock_async_func = MagicMock()
        mock_make_async.return_value = mock_async_func
        
        executor = MagicMock(spec=ExecutorBase)
        executor.collective_rpc = MagicMock()
        executor._accepting_requests = True
        
        executor.update_model_weights = ExecutorBase.update_model_weights.__get__(executor)
        
        model_weights_path = "/path/to/model/weights"
        executor.update_model_weights(model_weights_path)
        
        executor.collective_rpc.assert_called_once_with(
            "update_model_weights", args=(model_weights_path,))
        
        self.assertTrue(executor._accepting_requests)
    
    def test_worker_update_weights(self):
        """Test that Worker.update_model_weights calls model_runner.update_model_weights."""
        from vllm.worker.worker import Worker
        
        worker = MagicMock(spec=Worker)
        worker.model_runner = MagicMock()
        
        worker.update_model_weights = Worker.update_model_weights.__get__(worker)
        
        model_weights_path = "/path/to/model/weights"
        worker.update_model_weights(model_weights_path)
        
        worker.model_runner.update_model_weights.assert_called_once_with(model_weights_path)


if __name__ == "__main__":
    unittest.main()
