# Python Code Optimizer with CUDA GPU Acceleration

This project implements a Python code optimizer that mimics the optimization phases in traditional compilers. It takes Python code as input and generates optimized Python code as output, applying various optimization techniques. What makes this optimizer unique is that it leverages CUDA/NVCC GPU for the execution of the optimization process itself, making the optimization faster for large codebases.

## Features

- **GPU-Accelerated Optimization**: Uses CUDA for parallel processing of code analysis and transformation
- **Multiple Optimization Techniques**:
  - Constant Folding: Evaluates constant expressions at compile time
  - Dead Code Elimination: Removes code that doesn't affect the program output
  - Loop Unrolling: Reduces loop overhead by duplicating loop body
  - Common Subexpression Elimination: Avoids redundant computations
  - Function Inlining: Replaces function calls with function body
  - Strength Reduction: Replaces expensive operations with cheaper ones
- **Command-line Interface**: Easy to use from the command line
- **Benchmarking**: Built-in benchmarking to measure optimization performance
- **Detailed Statistics**: Provides detailed statistics about the optimization process

## Requirements

- Python 3.6+
- CUDA Toolkit 11.0+
- NumPy
- Numba

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/getthecodeoutofit/CUDA.git
   cd CUDA
   ```

2. Install the required packages:
   ```
   pip install numpy numba
   ```

3. Make sure you have CUDA Toolkit installed and configured.

## Usage

### Basic Usage

```
python main.py input_file.py [output_file.py]
```

If no output file is specified, the optimized code will be written to `input_file.optimized.py`.

### Options

```
python main.py [options] input_file [output_file]

Options:
  -h, --help                      Show this help message and exit
  -v, --verbose                   Enable verbose output
  --no-constant-folding           Disable constant folding optimization
  --no-dead-code-elimination      Disable dead code elimination
  --no-loop-unrolling             Disable loop unrolling
  --no-common-subexpr-elimination Disable common subexpression elimination
  --no-function-inlining          Disable function inlining
  --no-strength-reduction         Disable strength reduction
  --cuda-device DEVICE_ID         CUDA device ID to use (default: 0)
  --benchmark                     Run benchmark on the optimization process
```

### Examples

Optimize a Python file with all optimizations enabled:
```
python main.py examples/example1.py
```

Optimize a Python file with specific optimizations disabled:
```
python main.py --no-loop-unrolling --no-function-inlining examples/example2.py
```

Run a benchmark on the optimization process:
```
python main.py --benchmark examples/example1.py
```

## Project Structure

- `optimizer.py`: Main module with the Optimizer class
- `cuda_kernels.py`: CUDA kernels for optimization operations
- `optimization_techniques.py`: Implementation of different optimization techniques
- `ast_utils.py`: Utilities for working with Python's Abstract Syntax Tree
- `main.py`: Entry point for the application
- `examples/`: Directory with example Python code to optimize
- `tests/`: Directory with test cases

## How It Works

1. The optimizer parses the input Python code into an Abstract Syntax Tree (AST)
2. It applies various optimization techniques to the AST, using CUDA for parallel processing
3. It generates optimized Python code from the transformed AST

### GPU Acceleration

The optimizer uses CUDA for:
- Parallel pattern matching in the code
- Parallel evaluation of constant expressions
- Parallel analysis of variable uses and definitions for dead code elimination
- Other parallel operations in the optimization process

## Running Tests

To run the tests:
```
python -m unittest discover tests
```

## Limitations

- The optimizer works best on pure Python code without complex dependencies
- Some optimizations may change the behavior of the code if it relies on side effects
- The optimizer does not handle all Python language features


