import numpy as np
import math
import ast
import pickle
import warnings
from typing import Dict, List, Optional, Set, Tuple, Union, Callable

try:
    from numba import cuda, float32, int32, boolean, void, njit
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
except Exception as e:
    warnings.warn(f"CUDA initialization error: {str(e)}")
    CUDA_AVAILABLE = False

_cuda_context = None
_cuda_device = None
_cuda_enabled = False


def initialize_cuda(device_id: int = 0):
    global _cuda_context, _cuda_device, _cuda_enabled

    if not CUDA_AVAILABLE:
        print("CUDA is not available. Running in CPU-only mode.")
        _cuda_enabled = False
        return

    try:
        if _cuda_context is None:
            if not cuda.is_available():
                print("CUDA is not available on this system. Running in CPU-only mode.")
                _cuda_enabled = False
                return

            available_devices = cuda.list_devices()
            num_devices = len(available_devices)

            if device_id >= num_devices or device_id < 0:
                print(f"Invalid CUDA device ID: {device_id}")
                print(f"Available devices: {num_devices} (IDs: 0-{num_devices-1})")
                for i, device in enumerate(available_devices):
                    print(f"  Device {i}: {device}")
                print("Running in CPU-only mode.")
                _cuda_enabled = False
                return

            cuda.select_device(device_id)
            _cuda_device = cuda.get_current_device()
            _cuda_context = True 
            _cuda_enabled = True
            print(f"CUDA initialized on device {device_id}: {_cuda_device.name.decode()}")
    except Exception as e:
        print(f"Failed to initialize CUDA: {str(e)}")
        print("Running in CPU-only mode.")
        _cuda_enabled = False


def shutdown_cuda():
    global _cuda_context, _cuda_device, _cuda_enabled

    if _cuda_enabled:
        _cuda_context = None
        _cuda_device = None
        _cuda_enabled = False
        print("CUDA resources released")
    else:
        print("No CUDA resources to release (CPU-only mode)")


if CUDA_AVAILABLE:
    @cuda.jit
    def _parallel_pattern_match_kernel(patterns, code_tokens, results):
        pos = cuda.grid(1)
        if pos >= len(code_tokens):
            return

        for pattern_idx in range(len(patterns)):
            pattern_length = patterns[pattern_idx][0]

            if pos + pattern_length > len(code_tokens):
                continue

            match = True
            for i in range(pattern_length):
                if patterns[pattern_idx][i + 1] != code_tokens[pos + i]:
                    match = False
                    break

            if match:
                results[pos] = pattern_idx + 1 


def parallel_pattern_match(patterns: List[List[int]], code_tokens: List[int]) -> List[Tuple[int, int]]:
    max_pattern_length = max(len(pattern) for pattern in patterns)
    patterns_array = np.zeros((len(patterns), max_pattern_length + 1), dtype=np.int32)

    for i, pattern in enumerate(patterns):
        patterns_array[i, 0] = len(pattern)  
        patterns_array[i, 1:len(pattern) + 1] = pattern

    code_array = np.array(code_tokens, dtype=np.int32)
    results = np.zeros(len(code_array), dtype=np.int32)

    if CUDA_AVAILABLE and _cuda_enabled:
        threads_per_block = 256
        blocks_per_grid = (len(code_array) + threads_per_block - 1) // threads_per_block

        try:
            _parallel_pattern_match_kernel[blocks_per_grid, threads_per_block](
                patterns_array, code_array, results
            )
        except Exception as e:
            print(f"CUDA kernel execution failed: {str(e)}")
            print("Falling back to CPU implementation")
            _cpu_pattern_match(patterns_array, code_array, results)
    else:
        _cpu_pattern_match(patterns_array, code_array, results)

    matches = []
    for pos, pattern_idx in enumerate(results):
        if pattern_idx > 0: 
            matches.append((pos, pattern_idx - 1))

    return matches


def _cpu_pattern_match(patterns, code_tokens, results):

    for pos in range(len(code_tokens)):
        for pattern_idx in range(len(patterns)):
            pattern_length = patterns[pattern_idx][0]

            if pos + pattern_length > len(code_tokens):
                continue

            match = True
            for i in range(pattern_length):
                if patterns[pattern_idx][i + 1] != code_tokens[pos + i]:
                    match = False
                    break

            if match:
                results[pos] = pattern_idx + 1 


if CUDA_AVAILABLE:
    @cuda.jit
    def _parallel_constant_folding_kernel(expressions, results):

        idx = cuda.grid(1)
        if idx >= len(expressions):
            return
        expr = expressions[idx]
        op_type = expr[0]

        if op_type == 0:
            results[idx] = expr[1] + expr[2]
        elif op_type == 1: 
            results[idx] = expr[1] - expr[2]
        elif op_type == 2:
            results[idx] = expr[1] * expr[2]
        elif op_type == 3:
            if expr[2] != 0:
                results[idx] = expr[1] / expr[2]
        elif op_type == 4: 
            results[idx] = expr[1] ** expr[2]
        elif op_type == 5:
            if expr[2] != 0: 
                results[idx] = expr[1] % expr[2]


def parallel_constant_folding(expressions: List[Tuple]) -> List[float]:
    expr_array = np.array(expressions, dtype=np.float32)
    results = np.zeros(len(expressions), dtype=np.float32)

    if CUDA_AVAILABLE and _cuda_enabled:
        threads_per_block = 256
        blocks_per_grid = (len(expressions) + threads_per_block - 1) // threads_per_block

        try:
            _parallel_constant_folding_kernel[blocks_per_grid, threads_per_block](
                expr_array, results
            )
        except Exception as e:
            print(f"CUDA kernel execution failed: {str(e)}")
            print("Falling back to CPU implementation")
            _cpu_constant_folding(expr_array, results)
    else:
        _cpu_constant_folding(expr_array, results)

    return results.tolist()


def _cpu_constant_folding(expressions, results):
    for idx in range(len(expressions)):
        expr = expressions[idx]
        op_type = expr[0]

        if op_type == 0: 
            results[idx] = expr[1] + expr[2]
        elif op_type == 1: 
            results[idx] = expr[1] - expr[2]
        elif op_type == 2: 
            results[idx] = expr[1] * expr[2]
        elif op_type == 3:
            if expr[2] != 0: 
                results[idx] = expr[1] / expr[2]
        elif op_type == 4: 
            results[idx] = expr[1] ** expr[2]
        elif op_type == 5:
            if expr[2] != 0:
                results[idx] = expr[1] % expr[2]


if CUDA_AVAILABLE:
    @cuda.jit
    def _parallel_dead_code_analysis_kernel(var_uses, var_defs, results):
        var_idx = cuda.grid(1)
        if var_idx >= len(var_defs):
            return

        # A variable definition is dead if the variable is never used
        if var_uses[var_idx] == 0 and var_defs[var_idx] == 1:
            results[var_idx] = 1
        else:
            results[var_idx] = 0


def parallel_dead_code_analysis(var_uses: List[int], var_defs: List[int]) -> List[int]:

    var_uses_array = np.array(var_uses, dtype=np.int32)
    var_defs_array = np.array(var_defs, dtype=np.int32)
    results = np.zeros(len(var_defs), dtype=np.int32)

    if CUDA_AVAILABLE and _cuda_enabled:
        threads_per_block = 256
        blocks_per_grid = (len(var_defs) + threads_per_block - 1) // threads_per_block

        try:
            _parallel_dead_code_analysis_kernel[blocks_per_grid, threads_per_block](
                var_uses_array, var_defs_array, results
            )
        except Exception as e:
            print(f"CUDA kernel execution failed: {str(e)}")
            print("Falling back to CPU implementation")
            _cpu_dead_code_analysis(var_uses_array, var_defs_array, results)
    else:
        _cpu_dead_code_analysis(var_uses_array, var_defs_array, results)

    return results.tolist()


def _cpu_dead_code_analysis(var_uses, var_defs, results):
    for var_idx in range(len(var_defs)):
        if var_uses[var_idx] == 0 and var_defs[var_idx] == 1:
            results[var_idx] = 1 
        else:
            results[var_idx] = 0 
