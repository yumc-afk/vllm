"""
Code path analysis for vLLM online weight switching functionality.

This script analyzes all possible execution paths in the weight switching implementation
to ensure correctness and identify potential issues.
"""

import os
import sys
import inspect
import ast
from typing import Dict, List, Set, Tuple, Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class FunctionCallVisitor(ast.NodeVisitor):
    """AST visitor to find function calls within a function."""
    
    def __init__(self):
        self.calls = set()
        
    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            self.calls.add(node.func.attr)
        elif isinstance(node.func, ast.Name):
            self.calls.add(node.func.id)
        self.generic_visit(node)

class ControlFlowVisitor(ast.NodeVisitor):
    """AST visitor to analyze control flow paths."""
    
    def __init__(self):
        self.branches = 0
        self.loops = 0
        self.try_blocks = 0
        self.exception_handlers = 0
        
    def visit_If(self, node):
        self.branches += 1
        self.generic_visit(node)
        
    def visit_For(self, node):
        self.loops += 1
        self.generic_visit(node)
        
    def visit_While(self, node):
        self.loops += 1
        self.generic_visit(node)
        
    def visit_Try(self, node):
        self.try_blocks += 1
        self.exception_handlers += len(node.handlers)
        self.generic_visit(node)

def analyze_function(func):
    """Analyze a function's code paths and function calls."""
    source = inspect.getsource(func)
    tree = ast.parse(source)
    
    call_visitor = FunctionCallVisitor()
    call_visitor.visit(tree)
    
    flow_visitor = ControlFlowVisitor()
    flow_visitor.visit(tree)
    
    return {
        "function_name": func.__name__,
        "calls": call_visitor.calls,
        "branches": flow_visitor.branches,
        "loops": flow_visitor.loops,
        "try_blocks": flow_visitor.try_blocks,
        "exception_handlers": flow_visitor.exception_handlers
    }

def analyze_execution_path(start_func, max_depth=3):
    """Analyze execution paths starting from a function."""
    visited = set()
    path = []
    
    def dfs(func, depth=0):
        if depth > max_depth or func.__name__ in visited:
            return
        
        visited.add(func.__name__)
        path.append(func.__name__)
        
        analysis = analyze_function(func)
        print(f"{'  ' * depth}Function: {func.__name__}")
        print(f"{'  ' * depth}  Branches: {analysis['branches']}")
        print(f"{'  ' * depth}  Loops: {analysis['loops']}")
        print(f"{'  ' * depth}  Try blocks: {analysis['try_blocks']}")
        print(f"{'  ' * depth}  Exception handlers: {analysis['exception_handlers']}")
        print(f"{'  ' * depth}  Calls: {', '.join(analysis['calls'])}")
        
        for call_name in analysis['calls']:
            if hasattr(func.__self__, call_name):
                called_func = getattr(func.__self__, call_name)
                if callable(called_func):
                    dfs(called_func, depth + 1)
        
        path.pop()
    
    dfs(start_func)
    return visited

def analyze_weight_switching_paths():
    """Analyze all execution paths in the weight switching implementation."""
    print("=== Code Path Analysis for vLLM Online Weight Switching ===\n")
    
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.engine.llm_engine import LLMEngine
    from vllm.executor.executor_base import ExecutorBase
    from vllm.worker.worker import Worker
    from vllm.worker.model_runner import GPUModelRunnerBase
    
    print("\n=== Analyzing AsyncLLMEngine.update_model_weights ===")
    analyze_function(AsyncLLMEngine.update_model_weights)
    
    print("\n=== Analyzing AsyncLLMEngine.update_model_weights_from_state_dict ===")
    analyze_function(AsyncLLMEngine.update_model_weights_from_state_dict)
    
    print("\n=== Analyzing LLMEngine.update_model_weights ===")
    analyze_function(LLMEngine.update_model_weights)
    
    print("\n=== Analyzing LLMEngine.update_model_weights_from_state_dict ===")
    analyze_function(LLMEngine.update_model_weights_from_state_dict)
    
    print("\n=== Analyzing ExecutorBase.update_model_weights ===")
    analyze_function(ExecutorBase.update_model_weights)
    
    print("\n=== Analyzing ExecutorBase.update_model_weights_from_state_dict ===")
    analyze_function(ExecutorBase.update_model_weights_from_state_dict)
    
    print("\n=== Analyzing Worker.update_model_weights ===")
    analyze_function(Worker.update_model_weights)
    
    print("\n=== Analyzing Worker.update_model_weights_from_state_dict ===")
    analyze_function(Worker.update_model_weights_from_state_dict)
    
    print("\n=== Analyzing GPUModelRunnerBase.update_model_weights ===")
    analyze_function(GPUModelRunnerBase.update_model_weights)
    
    print("\n=== Analyzing GPUModelRunnerBase.update_model_weights_from_state_dict ===")
    analyze_function(GPUModelRunnerBase.update_model_weights_from_state_dict)
    
    print("\n=== Code Path Analysis Complete ===")

def main():
    """Run code path analysis."""
    try:
        analyze_weight_switching_paths()
    except Exception as e:
        print(f"Error during code path analysis: {e}")

if __name__ == "__main__":
    main()
