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
    # Constant expression that can be folded
    pi = 3.14159265358979
    
    # Common subexpression that can be eliminated
    r_squared = radius * radius
    area = pi * r_squared
    
    return area


def complex_calculation(x, y):
    """Perform a complex calculation with optimization opportunities."""
    # Dead code - this variable is never used
    unused_var = x + y + 100
    
    # Constant expressions that can be folded
    const_val = 10 * 20 + 5 / 2.5
    
    # Strength reduction opportunity (x ** 2 -> x * x)
    result = x ** 2 + y ** 2
    
    # Common subexpression
    expr = x * y + 10
    result += expr
    result += expr
    
    # More dead code - this condition is always false
    if False:
        print("This will never be executed")
        result = 0
    
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
    
    # Loop that can be unrolled
    sum_val = 0
    for i in range(10):
        sum_val += i * i
    
    # Function that can be inlined
    fact5 = factorial(5)
    
    # More constant expressions
    magic_number = 42 * 2 + 10 / 2
    
    # Common subexpressions
    x, y = 10, 20
    expr1 = x * y + math.sin(0.5)
    result1 = expr1 * 2
    result2 = expr1 * 3
    
    # Complex calculation with optimization opportunities
    final_result = complex_calculation(x, y)
    
    # More dead code - this is never used
    unused_result = x * y * 100
    
    # Print results
    print(f"Sum: {sum_val}")
    print(f"Factorial of 5: {fact5}")
    print(f"Magic number: {magic_number}")
    print(f"Results: {result1}, {result2}")
    print(f"Final result: {final_result}")
    
    # Calculate and print execution time
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.6f} seconds")


if __name__ == "__main__":
    main()
