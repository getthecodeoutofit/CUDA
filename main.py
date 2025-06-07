
import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Set, Tuple, Union, Callable

from optimizer import Optimizer


def parse_args():
    parser = argparse.ArgumentParser(description="Python Code Optimizer Tool")
    
    parser.add_argument("input_file", help="Input Python file to optimize")
    
    parser.add_argument("output_file", nargs="?", help="Output file for optimized code (default: input_file.optimized.py)")

    parser.add_argument("--no-constant-folding", action="store_true", help="Disable constant folding optimization")
    
    parser.add_argument("--no-dead-code-elimination", action="store_true", help="Disable dead code elimination")
    
    parser.add_argument("--no-loop-unrolling", action="store_true", help="Disable loop unrolling")
    
    parser.add_argument("--no-common-subexpr-elimination", action="store_true", help="Disable common subexpression elimination")
    
    parser.add_argument("--no-function-inlining", action="store_true", help="Disable function inlining")
    
    parser.add_argument("--no-strength-reduction", action="store_true", help="Disable strength reduction")
    
    parser.add_argument("--cuda-device", type=int, default=0, help="CUDA device ID to use (default: 0)")
    
    return parser.parse_args()


def print_optimization_stats(stats: Dict):
    """Print optimization statistics."""
    print("\nOptimization Statistics:")
    print(f"Execution time: {stats['execution_time']:.4f} seconds")
    print(f"Original code size: {stats['original_size']} bytes")
    print(f"Optimized code size: {stats['optimized_size']} bytes")
    print(f"Size reduction: {stats['size_reduction_percentage']:.2f}%")
    
    if 'folded_expressions' in stats:
        print(f"Constant expressions folded: {stats['folded_expressions']}")
    
    if 'eliminated_statements' in stats:
        print(f"Dead code statements eliminated: {stats['eliminated_statements']}")
    
    if 'unrolled_loops' in stats:
        print(f"Loops unrolled: {stats['unrolled_loops']}")


# def run_benchmark(optimizer: Optimizer, input_file: str):
#     print("\nRunning benchmark...")
    
#     with open(input_file, 'r') as f:
#         code = f.read()
#     optimizer.optimize(code)
#     num_runs = 5
#     total_time = 0
    
#     for i in range(num_runs):
#         start_time = time.time()
#         optimizer.optimize(code)
#         end_time = time.time()
        
#         run_time = end_time - start_time
#         total_time += run_time
        
#         print(f"Run {i+1}: {run_time:.4f} seconds")
    
#     avg_time = total_time / num_runs
#     print(f"\nAverage execution time over {num_runs} runs: {avg_time:.4f} seconds")


def main():
    """Main entry point."""
    args = parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist")
        sys.exit(1)
    
    optimizer = Optimizer(
        enable_constant_folding=not args.no_constant_folding,
        enable_dead_code_elimination=not args.no_dead_code_elimination,
        enable_loop_unrolling=not args.no_loop_unrolling,
        enable_common_subexpression_elimination=not args.no_common_subexpr_elimination,
        enable_function_inlining=not args.no_function_inlining,
        enable_strength_reduction=not args.no_strength_reduction,
        cuda_device=args.cuda_device,
        # verbose=args.verbose
    )
    
    # if args.benchmark:
    #     run_benchmark(optimizer, args.input_file)
    #     sys.exit(0)
    
    output_file = args.output_file
    stats = optimizer.optimize_file(args.input_file, output_file)
    print_optimization_stats(stats)
    print(f"\nOptimized code written to: {output_file or args.input_file + '.optimized.py'}")


if __name__ == "__main__":
    main()
