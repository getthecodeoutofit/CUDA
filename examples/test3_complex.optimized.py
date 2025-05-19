"""
A more complex test file with various optimization opportunities.
"""
import math
import time
PI = 3.14159
RADIUS = 5
AREA = PI * RADIUS * RADIUS

def complex_calculation(x, y):
    expr = x * y + 10
    result = expr * 2
    result += expr
    squared = x ** 2 + y ** 2
    return result + squared

def main():
    start_time = time.time()
    sum_val = 0
    for i in range(5):
        sum_val += i * i
    result = complex_calculation(10, 20)
    print(f'Circle area: {AREA}')
    print(f'Sum: {sum_val}')
    print(f'Result: {result}')
    end_time = time.time()
    print(f'Execution time: {end_time - start_time:.6f} seconds')
if __name__ == '__main__':
    main()