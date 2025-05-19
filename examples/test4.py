"""
Test file with various optimization opportunities.
"""

# Constant expressions
PI = 3.14159
RADIUS = 5
AREA = PI * RADIUS * RADIUS  # This can be constant-folded

# Function with optimization opportunities
def calculate(x, y):
    # Constant expression
    factor = 10 * 2  # This can be constant-folded
    
    # Common subexpression
    expr = x * y
    result1 = expr + 10
    result2 = expr + 20  # Reuses the common subexpression
    
    # Dead code - this is never used
    unused = x + y + 100
    
    # Conditional with constant condition - dead code elimination opportunity
    if False:
        print("This will never be executed")
        result1 = 0
    
    return result1 + result2

# Main code
x = 10
y = 20

# Calculate and print results
result = calculate(x, y)
print(f"Circle area: {AREA}")
print(f"Calculation result: {result}")

# Loop that can be unrolled
total = 0
for i in range(3):
    total += i

print(f"Total: {total}")

# More dead code - this is never used
unused_result = x * y * 100
