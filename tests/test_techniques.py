"""
Tests for the optimization techniques.

This module contains tests for individual optimization techniques.
"""

import ast
import os
import sys
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optimization_techniques import (
    ConstantFolder,
    DeadCodeEliminator,
    LoopUnroller,
    CommonSubexpressionEliminator,
    FunctionInliner,
    StrengthReducer,
)


class TestConstantFolder(unittest.TestCase):
    """Test cases for the ConstantFolder class."""
    
    def setUp(self):
        """Set up test cases."""
        self.optimizer = ConstantFolder()
    
    def test_simple_constant_folding(self):
        """Test simple constant folding."""
        code = """
x = 10 * 20
y = 3.14 * 2.0
z = "hello" + " world"
"""
        tree = ast.parse(code)
        optimized_tree = self.optimizer.optimize(tree)
        optimized_code = ast.unparse(optimized_tree)
        
        # Parse the optimized code to check if constants were folded
        tree = ast.parse(optimized_code)
        assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)]
        
        # Check if numeric constants were folded
        x_folded = False
        y_folded = False
        z_folded = False
        
        for assign in assignments:
            if (isinstance(assign.targets[0], ast.Name) and 
                isinstance(assign.value, ast.Constant)):
                if assign.targets[0].id == 'x' and assign.value.value == 200:
                    x_folded = True
                elif assign.targets[0].id == 'y' and abs(assign.value.value - 6.28) < 0.01:
                    y_folded = True
                elif assign.targets[0].id == 'z' and assign.value.value == "hello world":
                    z_folded = True
        
        self.assertTrue(x_folded, "Integer constant folding failed")
        self.assertTrue(y_folded, "Float constant folding failed")
        self.assertTrue(z_folded, "String constant folding failed")
    
    def test_nested_constant_folding(self):
        """Test nested constant folding."""
        code = """
x = 10 * (20 + 5)
y = (3.0 + 1.0) * (2.0 + 3.0)
"""
        tree = ast.parse(code)
        optimized_tree = self.optimizer.optimize(tree)
        optimized_code = ast.unparse(optimized_tree)
        
        # Parse the optimized code to check if constants were folded
        tree = ast.parse(optimized_code)
        assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)]
        
        # Check if nested constants were folded
        x_folded = False
        y_folded = False
        
        for assign in assignments:
            if (isinstance(assign.targets[0], ast.Name) and 
                isinstance(assign.value, ast.Constant)):
                if assign.targets[0].id == 'x' and assign.value.value == 250:
                    x_folded = True
                elif assign.targets[0].id == 'y' and assign.value.value == 20.0:
                    y_folded = True
        
        self.assertTrue(x_folded, "Nested integer constant folding failed")
        self.assertTrue(y_folded, "Nested float constant folding failed")


class TestDeadCodeEliminator(unittest.TestCase):
    """Test cases for the DeadCodeEliminator class."""
    
    def setUp(self):
        """Set up test cases."""
        self.optimizer = DeadCodeEliminator()
    
    def test_unused_variable_elimination(self):
        """Test elimination of unused variables."""
        code = """
def test_function():
    x = 10
    y = 20  # This is dead code
    return x
"""
        tree = ast.parse(code)
        optimized_tree = self.optimizer.optimize(tree)
        optimized_code = ast.unparse(optimized_tree)
        
        # Parse the optimized code to check if dead code was eliminated
        tree = ast.parse(optimized_code)
        function_def = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
        
        # Count assignments in the function body
        assignments = [node for node in function_def.body if isinstance(node, ast.Assign)]
        
        # There should be only one assignment (x = 10) after optimization
        self.assertEqual(len(assignments), 1, "Unused variable elimination failed")
        self.assertEqual(assignments[0].targets[0].id, 'x', "Wrong variable was eliminated")
    
    def test_unreachable_code_elimination(self):
        """Test elimination of unreachable code."""
        code = """
def test_function():
    x = 10
    if False:
        x = 20  # This is unreachable
    return x
"""
        tree = ast.parse(code)
        optimized_tree = self.optimizer.optimize(tree)
        optimized_code = ast.unparse(optimized_tree)
        
        # Parse the optimized code to check if unreachable code was eliminated
        tree = ast.parse(optimized_code)
        function_def = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
        
        # Count if statements in the function body
        if_statements = [node for node in function_def.body if isinstance(node, ast.If)]
        
        # There should be no if statements after optimization
        self.assertEqual(len(if_statements), 0, "Unreachable code elimination failed")


# Add more test classes for other optimization techniques as needed


if __name__ == '__main__':
    unittest.main()
