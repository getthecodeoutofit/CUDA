"""
Test file with various optimization opportunities.
"""
PI = 3.14159
RADIUS = 5
AREA = PI * RADIUS * RADIUS

def calculate(x, y):
    expr = x * y
    result1 = expr + 10
    result2 = expr + 20
    return result1 + result2
x = 10
y = 20
result = calculate(x, y)
print(f'Circle area: {AREA}')
print(f'Calculation result: {result}')
total = 0
for i in range(3):
    total += i
print(f'Total: {total}')