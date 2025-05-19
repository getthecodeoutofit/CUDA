"""
Tests for the Python code optimizer.

This module contains tests for the main optimizer functionality.
"""

import ast
import os
import sys
import unittest
from unittest.mock import patch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optimizer import Optimizer


class TestOptimizer(unittest.TestCase):
    """Test cases for the Optimizer class."""
    
    def setUp(self):
        """Set up test cases."""
        self.optimizer = Optimizer(verbose=False)
    
    def tearDown(self):
        """Clean up after tests."""
        del self.optimizer
    
    def test_constant_folding(self):
        """Test constant folding optimization."""
        code = """
def test_function():
    x = 10 * 20 + 5
    y = 3.14 * 2.0
    return x + y
"""
        optimized_code, stats = self.optimizer.optimize(code)
        
        # Parse the optimized code to check if constants were folded
        tree = ast.parse(optimized_code)
        assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)]
        
        # Check if constants were folded
        constants_folded = False
        for assign in assignments:
            if (isinstance(assign.targets[0], ast.Name) and 
                isinstance(assign.value, ast.Constant)):
                if assign.targets[0].id == 'x' and assign.value.value == 205:
                    constants_folded = True
        
        self.assertTrue(constants_folded, "Constant folding did not work as expected")
    
    def test_dead_code_elimination(self):
        """Test dead code elimination optimization."""
        code = """
def test_function():
    x = 10
    y = 20  # This is dead code
    return x
"""
        optimized_code, stats = self.optimizer.optimize(code)
        
        # Parse the optimized code to check if dead code was eliminated
        tree = ast.parse(optimized_code)
        function_def = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
        
        # Count assignments in the function body
        assignments = [node for node in function_def.body if isinstance(node, ast.Assign)]
        
        # There should be only one assignment (x = 10) after optimization
        self.assertEqual(len(assignments), 1, "Dead code elimination did not work as expected")
    
    def test_optimize_file(self):
        """Test optimizing a file."""
        # Create a temporary test file
        test_file = "temp_test_file.py"
        with open(test_file, 'w') as f:
            f.write("""
def test_function():
    x = 10 * 20  # This should be folded
    y = 30  # This is dead code
    return x
""")
        
        try:
            # Optimize the file
            output_file = "temp_test_file.optimized.py"
            stats = self.optimizer.optimize_file(test_file, output_file)
            
            # Check if the output file exists
            self.assertTrue(os.path.exists(output_file), "Output file was not created")
            
            # Read the optimized code
            with open(output_file, 'r') as f:
                optimized_code = f.read()
            
            # Parse the optimized code to check optimizations
            tree = ast.parse(optimized_code)
            function_def = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
            
            # Count assignments in the function body
            assignments = [node for node in function_def.body if isinstance(node, ast.Assign)]
            
            # There should be only one assignment after optimization
            self.assertEqual(len(assignments), 1, "File optimization did not work as expected")
            
            # Check if the constant was folded
            if assignments:
                self.assertTrue(
                    isinstance(assignments[0].value, ast.Constant) and assignments[0].value.value == 200,
                    "Constant folding in file optimization did not work as expected"
                )
        
        finally:
            # Clean up temporary files
            if os.path.exists(test_file):
                os.remove(test_file)
            if os.path.exists(output_file):
                os.remove(output_file)
    
    def test_all_optimizations_disabled(self):
        """Test optimizer with all optimizations disabled."""
        optimizer = Optimizer(
            enable_constant_folding=False,
            enable_dead_code_elimination=False,
            enable_loop_unrolling=False,
            enable_common_subexpression_elimination=False,
            enable_function_inlining=False,
            enable_strength_reduction=False,
            verbose=False
        )
        
        code = """
def test_function():
    x = 10 * 20
    y = 30
    return x
"""
        optimized_code, stats = optimizer.optimize(code)
        
        # The code should be unchanged
        self.assertEqual(
            ast.unparse(ast.parse(code)).strip(),
            ast.unparse(ast.parse(optimized_code)).strip(),
            "Disabled optimizations still modified the code"
        )


if __name__ == '__main__':
    unittest.main()
