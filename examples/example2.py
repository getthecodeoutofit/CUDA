"""
Example 2: More complex Python code with optimization opportunities.

This example contains more complex patterns that can be optimized:
- Classes with methods that can be inlined
- More complex loops and conditionals
- Nested functions
- More opportunities for various optimizations
"""

import math
import random
import time
from typing import List, Tuple


class Vector:
    """A simple 2D vector class with optimization opportunities."""
    
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def magnitude(self) -> float:
        """Calculate the magnitude of the vector."""
        # Common subexpression and strength reduction opportunity
        return math.sqrt(self.x ** 2 + self.y ** 2)
    
    def normalize(self) -> 'Vector':
        """Return a normalized version of the vector."""
        mag = self.magnitude()
        # Dead code - this condition is almost always false for normal vectors
        if mag == 0:
            return Vector(0, 0)
        # Common subexpression - mag is used multiple times
        return Vector(self.x / mag, self.y / mag)
    
    def dot(self, other: 'Vector') -> float:
        """Calculate the dot product with another vector."""
        # Constant folding and common subexpression opportunities
        return self.x * other.x + self.y * other.y
    
    def __add__(self, other: 'Vector') -> 'Vector':
        """Add two vectors."""
        return Vector(self.x + other.x, self.y + other.y)
    
    def __mul__(self, scalar: float) -> 'Vector':
        """Multiply vector by a scalar."""
        return Vector(self.x * scalar, self.y * scalar)
    
    def __str__(self) -> str:
        """String representation of the vector."""
        # Constant folding opportunity
        return f"Vector({self.x:.2f}, {self.y:.2f})"


def process_vectors(vectors: List[Vector], factor: float) -> List[Vector]:
    """Process a list of vectors with optimization opportunities."""
    result = []
    
    # Loop that can be unrolled
    for i in range(len(vectors)):
        # Common subexpression
        v = vectors[i]
        
        # Dead code - this variable is calculated but never used
        unused_mag = v.magnitude() * 2
        
        # Function that can be inlined
        normalized = v.normalize()
        
        # Constant expression
        scaled = normalized * (factor * 1.5)
        
        result.append(scaled)
    
    return result


def calculate_statistics(values: List[float]) -> Tuple[float, float, float]:
    """Calculate statistics with optimization opportunities."""
    # Dead code - this is never used
    unused_values = values.copy()
    
    # Common subexpression - len(values) is used multiple times
    n = len(values)
    
    if n == 0:
        return 0, 0, 0
    
    # Common subexpression - sum(values) is used multiple times
    total = sum(values)
    mean = total / n
    
    # This loop can be optimized
    variance_sum = 0
    for value in values:
        # Common subexpression and strength reduction
        diff = value - mean
        variance_sum += diff ** 2
    
    variance = variance_sum / n
    std_dev = math.sqrt(variance)
    
    return mean, variance, std_dev


def fibonacci(n: int) -> int:
    """Calculate Fibonacci number - can be optimized for small values."""
    # Constant folding for base cases
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        # This recursive call can be optimized
        return fibonacci(n - 1) + fibonacci(n - 2)


def main():
    """Main function with various optimization opportunities."""
    start_time = time.time()
    
    # Create some random vectors
    vectors = []
    # Loop that can be unrolled
    for _ in range(10):
        x = random.uniform(-10, 10)
        y = random.uniform(-10, 10)
        vectors.append(Vector(x, y))
    
    # Process vectors
    factor = 2.5
    processed_vectors = process_vectors(vectors, factor)
    
    # Calculate magnitudes
    magnitudes = []
    # Another loop that can be unrolled
    for v in processed_vectors:
        # Function call that can be inlined
        mag = v.magnitude()
        magnitudes.append(mag)
    
    # Calculate statistics
    mean, variance, std_dev = calculate_statistics(magnitudes)
    
    # Calculate Fibonacci numbers
    fib_results = []
    # Loop with function calls that can be inlined
    for i in range(10):
        fib_results.append(fibonacci(i))
    
    # Print results
    print(f"Processed {len(vectors)} vectors with factor {factor}")
    print(f"Magnitudes statistics: mean={mean:.2f}, variance={variance:.2f}, std_dev={std_dev:.2f}")
    print(f"Fibonacci sequence: {fib_results}")
    
    # Calculate and print execution time
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.6f} seconds")


if __name__ == "__main__":
    main()
