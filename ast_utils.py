"""
Utilities for working with Python's Abstract Syntax Tree (AST).

This module provides helper classes for visiting and transforming AST nodes.
"""

import ast
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable


class ASTVisitor(ast.NodeVisitor):
    """
    Extended AST visitor with additional utilities for code analysis.
    
    This class provides methods for collecting information about the AST
    that can be used for optimization decisions.
    """
    
    def __init__(self):
        self.variables = set()
        self.functions = set()
        self.function_calls = set()
        self.constants = {}
        self.loops = []
        self.conditionals = []
        self.assignments = []
        self.imports = []
    
    def visit_Name(self, node):
        """Record variable names."""
        self.variables.add(node.id)
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        """Record function definitions."""
        self.functions.add(node.name)
        self.generic_visit(node)
    
    def visit_Call(self, node):
        """Record function calls."""
        if isinstance(node.func, ast.Name):
            self.function_calls.add(node.func.id)
        self.generic_visit(node)
    
    def visit_Constant(self, node):
        """Record constant values."""
        # Use string representation of the node as a key
        node_str = ast.dump(node)
        self.constants[node_str] = node.value
        self.generic_visit(node)
    
    def visit_For(self, node):
        """Record for loops."""
        self.loops.append(('for', node))
        self.generic_visit(node)
    
    def visit_While(self, node):
        """Record while loops."""
        self.loops.append(('while', node))
        self.generic_visit(node)
    
    def visit_If(self, node):
        """Record if statements."""
        self.conditionals.append(node)
        self.generic_visit(node)
    
    def visit_Assign(self, node):
        """Record assignments."""
        self.assignments.append(node)
        self.generic_visit(node)
    
    def visit_Import(self, node):
        """Record imports."""
        for name in node.names:
            self.imports.append(('import', name.name, name.asname))
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Record from imports."""
        for name in node.names:
            self.imports.append(('from', node.module, name.name, name.asname))
        self.generic_visit(node)


class ASTTransformer(ast.NodeTransformer):
    """
    Extended AST transformer with additional utilities for code transformation.
    
    This class provides methods for transforming the AST to implement
    various optimization techniques.
    """
    
    def __init__(self):
        self.transformed_nodes = 0
        self.variable_renames = {}
        self.function_renames = {}
        self.removed_nodes = 0
    
    def replace_variable(self, old_name: str, new_name: str):
        """
        Register a variable replacement.
        
        Args:
            old_name: Original variable name
            new_name: New variable name
        """
        self.variable_renames[old_name] = new_name
    
    def replace_function(self, old_name: str, new_name: str):
        """
        Register a function replacement.
        
        Args:
            old_name: Original function name
            new_name: New function name
        """
        self.function_renames[old_name] = new_name
    
    def visit_Name(self, node):
        """Replace variable names according to the rename map."""
        if isinstance(node.ctx, ast.Load) and node.id in self.variable_renames:
            self.transformed_nodes += 1
            return ast.Name(id=self.variable_renames[node.id], ctx=node.ctx)
        return self.generic_visit(node)
    
    def visit_Call(self, node):
        """Replace function calls according to the rename map."""
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id in self.function_renames:
            self.transformed_nodes += 1
            node.func.id = self.function_renames[node.func.id]
        return node
    
    def remove_node(self, node):
        """
        Mark a node for removal.
        
        This is a helper method that can be used by optimization techniques
        to remove nodes from the AST.
        
        Args:
            node: The node to remove
            
        Returns:
            None, indicating that the node should be removed
        """
        self.removed_nodes += 1
        return None


def get_node_source(node, source_code):
    """
    Get the source code for a specific AST node.
    
    Args:
        node: The AST node
        source_code: The complete source code
        
    Returns:
        The source code corresponding to the node
    """
    if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
        lines = source_code.splitlines()
        if node.lineno == node.end_lineno:
            return lines[node.lineno - 1][node.col_offset:node.end_col_offset]
        else:
            result = [lines[node.lineno - 1][node.col_offset:]]
            result.extend(lines[node.lineno:node.end_lineno - 1])
            result.append(lines[node.end_lineno - 1][:node.end_col_offset])
            return '\n'.join(result)
    return ast.unparse(node)


def is_constant_expression(node):
    """
    Check if an AST node represents a constant expression.
    
    Args:
        node: The AST node to check
        
    Returns:
        True if the node is a constant expression, False otherwise
    """
    if isinstance(node, ast.Constant):
        return True
    
    if isinstance(node, ast.BinOp):
        return is_constant_expression(node.left) and is_constant_expression(node.right)
    
    if isinstance(node, ast.UnaryOp):
        return is_constant_expression(node.operand)
    
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(is_constant_expression(elt) for elt in node.elts)
    
    if isinstance(node, ast.Dict):
        return (all(is_constant_expression(k) for k in node.keys if k is not None) and
                all(is_constant_expression(v) for v in node.values))
    
    return False


def evaluate_constant_expression(node):
    """
    Evaluate a constant expression represented by an AST node.
    
    Args:
        node: The AST node representing a constant expression
        
    Returns:
        The evaluated value of the constant expression
    
    Raises:
        ValueError: If the node is not a constant expression
    """
    if not is_constant_expression(node):
        raise ValueError("Not a constant expression")
    
    # Convert the AST node to a Python expression and evaluate it
    code = ast.unparse(node)
    return eval(code)
