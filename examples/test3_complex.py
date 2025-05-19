"""
A more complex test file with various optimization opportunities.
"""

import math
import time

# Constants that can be folded
PI = 3.14159
RADIUS = 5
AREA = PI * RADIUS * RADIUS  # This can be constant-folded

# Dead code
unused_variable = 100  # This is never used

def complex_calculation(x, y):
    # More constants that can be folded
    factor = 10 * 5 + 3  # This can be constant-folded to 53
    
    # Common subexpression
    expr = x * y + 10
    result = expr * 2
    result += expr  # Same expression used again
    
    # Dead code - this is never used
    unused_result = x + y + 100
    
    # Strength reduction opportunity (x ** 2 -> x * x)
    squared = x ** 2 + y ** 2
    
    # More dead code - this condition is always false
    if False:
        print("This will never be executed")
        result = 0
    
    return result + squared

def main():
    start_time = time.time()
    
    # Loop that can be unrolled
    sum_val = 0
    for i in range(5):
        sum_val += i * i
    
    # Call the function with some values
    result = complex_calculation(10, 20)
    
    # Print results
    print(f"Circle area: {AREA}")
    print(f"Sum: {sum_val}")
    print(f"Result: {result}")
    
    # Calculate and print execution time
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.6f} seconds")

if __name__ == "__main__":
    main()
