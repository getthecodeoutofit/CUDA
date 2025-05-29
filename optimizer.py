"""
Python Code Optimizer using CUDA for GPU acceleration.

This module contains the main Optimizer class that orchestrates the optimization process.
"""

import ast
import inspect
import time
from typing import Dict, List, Optional, Set, Tuple, Union, Callable

import numpy as np
from numba import cuda

from ast_utils import ASTVisitor, ASTTransformer
from cuda_kernels import initialize_cuda, shutdown_cuda
from optimization_techniques import (
    ConstantFolder,
    DeadCodeEliminator,
    LoopUnroller,
    CommonSubexpressionEliminator,
    FunctionInliner,
    StrengthReducer,
)


class Optimizer:

    def __init__(self, 
                 enable_constant_folding: bool = True,
                 enable_dead_code_elimination: bool = True,
                 enable_loop_unrolling: bool = True,
                 enable_common_subexpression_elimination: bool = True,
                 enable_function_inlining: bool = True,
                 enable_strength_reduction: bool = True,
                 cuda_device: int = 0,
                 verbose: bool = False):
        self.enable_constant_folding = enable_constant_folding
        self.enable_dead_code_elimination = enable_dead_code_elimination
        self.enable_loop_unrolling = enable_loop_unrolling
        self.enable_common_subexpression_elimination = enable_common_subexpression_elimination
        self.enable_function_inlining = enable_function_inlining
        self.enable_strength_reduction = enable_strength_reduction
        
        self.cuda_device = cuda_device
        self.verbose = verbose
        
        # Initialize CUDA context
        initialize_cuda(cuda_device)
        
        # Initialize optimization techniques
        self.constant_folder = ConstantFolder() if enable_constant_folding else None
        self.dead_code_eliminator = DeadCodeEliminator() if enable_dead_code_elimination else None
        self.loop_unroller = LoopUnroller() if enable_loop_unrolling else None
        self.common_subexpression_eliminator = CommonSubexpressionEliminator() if enable_common_subexpression_elimination else None
        self.function_inliner = FunctionInliner() if enable_function_inlining else None
        self.strength_reducer = StrengthReducer() if enable_strength_reduction else None
        
        self.optimization_stats = {}
    
    def __del__(self):
        """Clean up CUDA resources when the optimizer is destroyed."""
        shutdown_cuda()
    
    def optimize(self, code: str) -> Tuple[str, Dict]:

        start_time = time.time()
        
        # Parse the code into an AST
        tree = ast.parse(code)
        
        # Apply optimizations
        if self.verbose:
            print("Starting optimization process...")
        
        # Apply constant folding
        if self.enable_constant_folding:
            tree = self.constant_folder.optimize(tree)
            if self.verbose:
                print("Applied constant folding")
        
        # Apply dead code elimination
        if self.enable_dead_code_elimination:
            tree = self.dead_code_eliminator.optimize(tree)
            if self.verbose:
                print("Applied dead code elimination")
        
        # Apply loop unrolling
        if self.enable_loop_unrolling:
            tree = self.loop_unroller.optimize(tree)
            if self.verbose:
                print("Applied loop unrolling")
        
        # Apply common subexpression elimination
        if self.enable_common_subexpression_elimination:
            tree = self.common_subexpression_eliminator.optimize(tree)
            if self.verbose:
                print("Applied common subexpression elimination")
        
        # Apply function inlining
        if self.enable_function_inlining:
            tree = self.function_inliner.optimize(tree)
            if self.verbose:
                print("Applied function inlining")
        
        # Apply strength reduction
        if self.enable_strength_reduction:
            tree = self.strength_reducer.optimize(tree)
            if self.verbose:
                print("Applied strength reduction")
        
        # Generate optimized code
        optimized_code = ast.unparse(tree)
        
        end_time = time.time()
        
        # Collect optimization statistics
        self.optimization_stats = {
            "execution_time": end_time - start_time,
            "original_size": len(code),
            "optimized_size": len(optimized_code),
            "size_reduction_percentage": (1 - len(optimized_code) / len(code)) * 100 if len(code) > 0 else 0,
        }
        
        if self.verbose:
            print(f"Optimization completed in {self.optimization_stats['execution_time']:.4f} seconds")
            print(f"Size reduction: {self.optimization_stats['size_reduction_percentage']:.2f}%")
        
        return optimized_code, self.optimization_stats
    
    
    def optimize_file(self, input_file: str, output_file: Optional[str] = None) -> Dict:
        if self.verbose:
            print(f"Optimizing file: {input_file}")
        
        with open(input_file, 'r') as f:
            code = f.read()
        
        optimized_code, stats = self.optimize(code)
        
        if output_file is None:
            output_file = input_file.rsplit('.', 1)[0] + '.optimized.py'
        
        with open(output_file, 'w') as f:
            f.write(optimized_code)
        
        if self.verbose:
            print(f"Optimized code written to: {output_file}")
        
        return stats
