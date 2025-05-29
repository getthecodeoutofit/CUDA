
import ast
import copy
from typing import Dict, List, Optional, Set, Tuple, Union, Callable

import numpy as np
from numba import cuda

from ast_utils import ASTVisitor, ASTTransformer, is_constant_expression, evaluate_constant_expression
from cuda_kernels import parallel_pattern_match, parallel_constant_folding, parallel_dead_code_analysis


class OptimizationTechnique:

    def __init__(self):
        self.stats = {}

    def optimize(self, tree: ast.AST) -> ast.AST:

        raise NotImplementedError("Subclasses must implement this method")


class ConstantFolder(OptimizationTechnique):


    def __init__(self):
        super().__init__()
        self.folded_expressions = 0

    def optimize(self, tree: ast.AST) -> ast.AST:
        # First pass: collect all constant expressions
        visitor = ConstantFoldingVisitor()
        visitor.visit(tree)

        # Evaluate constant expressions
        for expr in visitor.constant_expressions:
            try:
                # Evaluate the expression directly
                value = evaluate_constant_expression(expr)
                visitor.evaluated_expressions[ast.dump(expr)] = value
            except Exception as e:
                # Skip expressions that can't be evaluated
                if isinstance(expr, ast.BinOp):
                    # For numeric expressions, try CUDA evaluation
                    if (isinstance(expr.left, ast.Constant) and
                        isinstance(expr.right, ast.Constant) and
                        isinstance(expr.left.value, (int, float)) and
                        isinstance(expr.right.value, (int, float))):

                        # Map operator to an integer code
                        op_code = -1
                        if isinstance(expr.op, ast.Add):
                            op_code = 0
                        elif isinstance(expr.op, ast.Sub):
                            op_code = 1
                        elif isinstance(expr.op, ast.Mult):
                            op_code = 2
                        elif isinstance(expr.op, ast.Div):
                            op_code = 3
                        elif isinstance(expr.op, ast.Pow):
                            op_code = 4
                        elif isinstance(expr.op, ast.Mod):
                            op_code = 5

                        if op_code >= 0:
                            # Evaluate the expression
                            if op_code == 0:  # Addition
                                result = expr.left.value + expr.right.value
                            elif op_code == 1:  # Subtraction
                                result = expr.left.value - expr.right.value
                            elif op_code == 2:  # Multiplication
                                result = expr.left.value * expr.right.value
                            elif op_code == 3:  # Division
                                if expr.right.value != 0:  # Avoid division by zero
                                    result = expr.left.value / expr.right.value
                                else:
                                    continue
                            elif op_code == 4:  # Power
                                result = expr.left.value ** expr.right.value
                            elif op_code == 5:  # Modulo
                                if expr.right.value != 0:  # Avoid modulo by zero
                                    result = expr.left.value % expr.right.value
                                else:
                                    continue

                            visitor.evaluated_expressions[ast.dump(expr)] = result

        # Second pass: replace constant expressions with their values
        transformer = ConstantFoldingTransformer(visitor.evaluated_expressions)
        new_tree = transformer.visit(tree)

        self.folded_expressions = transformer.folded_expressions
        self.stats = {
            "folded_expressions": self.folded_expressions
        }

        return new_tree


class ConstantFoldingVisitor(ASTVisitor):

    def __init__(self):
        super().__init__()
        self.constant_expressions = []
        self.evaluated_expressions = {}

    def visit_BinOp(self, node):
        """Collect binary operations that can be constant-folded."""
        if is_constant_expression(node):
            self.constant_expressions.append(node)
        self.generic_visit(node)

    def visit_UnaryOp(self, node):
        """Collect unary operations that can be constant-folded."""
        if is_constant_expression(node):
            self.constant_expressions.append(node)
        self.generic_visit(node)


class ConstantFoldingTransformer(ASTTransformer):
    """Transformer to replace constant expressions with their values."""

    def __init__(self, evaluated_expressions):
        super().__init__()
        self.evaluated_expressions = evaluated_expressions
        self.folded_expressions = 0

    def visit_BinOp(self, node):
        """Replace constant binary operations with their values."""
        node = self.generic_visit(node)
        node_str = ast.dump(node)

        if node_str in self.evaluated_expressions:
            self.folded_expressions += 1
            return ast.Constant(value=self.evaluated_expressions[node_str])

        if is_constant_expression(node):
            try:
                value = evaluate_constant_expression(node)
                self.folded_expressions += 1
                return ast.Constant(value=value)
            except:
                pass

        return node

    def visit_UnaryOp(self, node):
        """Replace constant unary operations with their values."""
        node = self.generic_visit(node)

        if is_constant_expression(node):
            try:
                value = evaluate_constant_expression(node)
                self.folded_expressions += 1
                return ast.Constant(value=value)
            except:
                pass

        return node


class DeadCodeEliminator(OptimizationTechnique):

    def __init__(self):
        super().__init__()
        self.eliminated_statements = 0

    def optimize(self, tree: ast.AST) -> ast.AST:
        # First pass: collect variable uses and definitions
        visitor = DeadCodeVisitor()
        visitor.visit(tree)

        var_uses = [0] * len(visitor.variables)
        var_defs = [0] * len(visitor.variables)

        var_to_idx = {var: i for i, var in enumerate(visitor.variables)}

        for var in visitor.used_variables:
            if var in var_to_idx:
                var_uses[var_to_idx[var]] = 1

        for var in visitor.defined_variables:
            if var in var_to_idx:
                var_defs[var_to_idx[var]] = 1


        dead_vars_result = parallel_dead_code_analysis(var_uses, var_defs)

        dead_variables = set()
        for i, is_dead in enumerate(dead_vars_result):
            if is_dead == 1:
                var = list(visitor.variables)[i]
                dead_variables.add(var)

        transformer = DeadCodeTransformer(dead_variables, visitor.used_variables)
        new_tree = transformer.visit(tree)

        self.eliminated_statements = transformer.eliminated_statements
        self.stats = {
            "eliminated_statements": self.eliminated_statements,
            "dead_variables": len(dead_variables)
        }

        return new_tree


class DeadCodeVisitor(ASTVisitor):


    def __init__(self):
        super().__init__()
        self.variables = set()
        self.used_variables = set()
        self.defined_variables = set()
        self.unreachable_code = []

    def visit_Name(self, node):

        self.variables.add(node.id)

        if isinstance(node.ctx, ast.Load):
            self.used_variables.add(node.id)
        elif isinstance(node.ctx, ast.Store):
            self.defined_variables.add(node.id)

        self.generic_visit(node)

    def visit_If(self, node):

        if isinstance(node.test, ast.Constant):
            if node.test.value:

                if node.orelse:
                    self.unreachable_code.append(('else', node))
            else:

                self.unreachable_code.append(('if', node))

        self.generic_visit(node)

    def visit_While(self, node):

        if isinstance(node.test, ast.Constant) and not node.test.value:

            self.unreachable_code.append(('while', node))

        self.generic_visit(node)


class DeadCodeTransformer(ASTTransformer):

    def __init__(self, dead_variables, used_variables):
        super().__init__()
        self.dead_variables = dead_variables
        self.used_variables = used_variables
        self.eliminated_statements = 0

    def visit_Assign(self, node):

        node = self.generic_visit(node)


        all_dead = True
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id not in self.dead_variables:
                all_dead = False
                break

        if all_dead:
            self.eliminated_statements += 1
            return None

        return node

    def visit_If(self, node):
        node = self.generic_visit(node)

        if isinstance(node.test, ast.Constant):
            if node.test.value:
                # The else branch is unreachable
                if node.orelse:
                    self.eliminated_statements += len(node.orelse)
                    node.orelse = []
            else:
                self.eliminated_statements += len(node.body)
                return ast.Module(body=node.orelse, type_ignores=[]) if isinstance(node, ast.Module) else node.orelse

        return node

    def visit_While(self, node):
        node = self.generic_visit(node)

        if isinstance(node.test, ast.Constant) and not node.test.value:
            self.eliminated_statements += len(node.body)
            return None

        return node

class LoopUnroller(OptimizationTechnique):

    def __init__(self, unroll_factor=4, max_iterations=16):
        super().__init__()
        self.unroll_factor = unroll_factor
        self.max_iterations = max_iterations
        self.unrolled_loops = 0

    def optimize(self, tree: ast.AST) -> ast.AST:
        transformer = LoopUnrollingTransformer(self.unroll_factor, self.max_iterations)
        new_tree = transformer.visit(tree)

        self.unrolled_loops = transformer.unrolled_loops
        self.stats = {
            "unrolled_loops": self.unrolled_loops
        }

        return new_tree


class LoopUnrollingTransformer(ASTTransformer):

    def __init__(self, unroll_factor, max_iterations):
        super().__init__()
        self.unroll_factor = unroll_factor
        self.max_iterations = max_iterations
        self.unrolled_loops = 0


class CommonSubexpressionEliminator(OptimizationTechnique):
    def __init__(self):
        super().__init__()
        self.eliminated_expressions = 0

    def optimize(self, tree: ast.AST) -> ast.AST:
        return tree


class FunctionInliner(OptimizationTechnique):

    def __init__(self, max_inline_size=20):
        super().__init__()
        self.max_inline_size = max_inline_size
        self.inlined_functions = 0

    def optimize(self, tree: ast.AST) -> ast.AST: #accespts the AST
        return tree


class StrengthReducer(OptimizationTechnique):
    def __init__(self):
        super().__init__()
        self.reduced_operations = 0

    def optimize(self, tree: ast.AST) -> ast.AST:
        return tree
