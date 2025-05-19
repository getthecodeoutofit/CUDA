"""
Example 1: Simple Python code with optimization opportunities.

This example contains various patterns that can be optimized:
- Constant expressions
- Dead code
- Loops that can be unrolled
- Common subexpressions
- Functions that can be inlined
- Operations that can be strength-reduced
"""
import math
import time

def calculate_area(radius):
    """Calculate the area of a circle."""
    pi = 3.14159265358979
    r_squared = radius * radius
    area = pi * r_squared
    return area

def complex_calculation(x, y):
    """Perform a complex calculation with optimization opportunities."""
    result = x ** 2 + y ** 2
    expr = x * y + 10
    result += expr
    result += expr
    return result

def factorial(n):
    """Calculate factorial - can be inlined for small values."""
    if n <= 1:
        return 1
    else:
        return n * factorial(n - 1)

def main():
    """Main function with a loop that can be unrolled."""
    start_time = time.time()
    sum_val = 0
    for i in range(10):
        sum_val += i * i
    fact5 = factorial(5)
    magic_number = 89.0
    expr1 = x * y + math.sin(0.5)
    result1 = expr1 * 2
    result2 = expr1 * 3
    final_result = complex_calculation(x, y)
    print(f'Sum: {sum_val}')
    print(f'Factorial of 5: {fact5}')
    print(f'Magic number: {magic_number}')
    print(f'Results: {result1}, {result2}')
    print(f'Final result: {final_result}')
    end_time = time.time()
    print(f'Execution time: {end_time - start_time:.6f} seconds')
if __name__ == '__main__':
    main()