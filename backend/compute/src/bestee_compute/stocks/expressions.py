"""Dedicated module for the small arithmetic-expression language the DSL uses.

Two DSL decorators accept a tiny Python arithmetic expression:

* ``COMPUTED_METRIC`` -> scalar metrics over ``stock.numeric_metrics``
  (:class:`~bestee_compute.stocks.decorators.ComputedMetricDecorator`),
  evaluated with :func:`evaluate_scalar`.
* ``TIME_SERIES_DERIVED`` -> element-wise series over
  ``stock.numeric_time_series``
  (:class:`~bestee_compute.stocks.decorators.TimeSeriesDerivedDecorator`),
  evaluated with :func:`evaluate_series`.

This module owns the whole pipeline: the per-dialect grammar
(:class:`ExpressionGrammar` plus the two constants below), parsing and
allow-list validation (:func:`parse`), name discovery (:func:`collect_names`),
and the two evaluators.  A decorator just picks a grammar and an evaluator.

The evaluators stay separate because their numeric domains differ: scalars
propagate ``None`` and guard divide-/mod-by-zero, while series evaluate
element-wise over numpy arrays (broadcasting included).
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

# ── Grammar ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExpressionGrammar:
    """Allow-list and error label for one expression dialect.

    Attributes:
        label: Command name used to prefix error messages (e.g.
            ``"COMPUTED_METRIC"``).
        binops: Permitted binary-operator AST node types.
        unaryops: Permitted unary-operator AST node types.
    """

    label: str
    binops: tuple[type[ast.operator], ...]
    unaryops: tuple[type[ast.unaryop], ...]


#: Scalar-metric dialect: ``+ - * / ** %`` and unary ``±``.
COMPUTED_METRIC_GRAMMAR = ExpressionGrammar(
    label="COMPUTED_METRIC",
    binops=(ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod),
    unaryops=(ast.UAdd, ast.USub),
)

#: Element-wise series dialect: ``+ - * /`` and unary minus.
TIME_SERIES_DERIVED_GRAMMAR = ExpressionGrammar(
    label="TIME_SERIES_DERIVED",
    binops=(ast.Add, ast.Sub, ast.Mult, ast.Div),
    unaryops=(ast.USub,),
)


# ── Parsing / validation ─────────────────────────────────────────────


def parse(expression: str, grammar: ExpressionGrammar) -> ast.expr:
    """Parse and validate *expression*, returning its AST body node.

    Parses in ``eval`` mode (a single expression, no statements) and checks
    every node against *grammar*'s allow-list of operators, numeric literals,
    and names.

    Raises:
        ValueError: If *expression* isn't valid ``eval``-mode Python, or
            contains a construct outside *grammar*'s allow-list.
    """
    try:
        tree = ast.parse(expression, mode="eval").body
    except SyntaxError as err:
        msg = (
            f"{grammar.label} expression {expression!r} is not valid "
            f"Python syntax: {err.msg}"
        )
        raise ValueError(msg) from err
    _validate(tree, grammar)
    return tree


def collect_names(node: ast.AST) -> set[str]:
    """Return the identifier names referenced anywhere in *node*."""
    return {sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name)}


def _validate(node: ast.AST, grammar: ExpressionGrammar) -> None:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            continue
        if isinstance(sub, ast.Constant):
            if not isinstance(sub.value, int | float):
                msg = f"{grammar.label} constants must be numeric, got {sub.value!r}"
                raise ValueError(msg)
            continue
        if isinstance(sub, ast.UnaryOp) and isinstance(sub.op, grammar.unaryops):
            continue
        if isinstance(sub, ast.BinOp) and isinstance(sub.op, grammar.binops):
            continue
        if isinstance(sub, ast.Expression):
            continue
        if isinstance(sub, ast.operator | ast.unaryop | ast.expr_context):
            continue
        msg = (
            f"{grammar.label} expression contains an unsupported "
            f"construct: {type(sub).__name__}"
        )
        raise ValueError(msg)


# ── Evaluation ───────────────────────────────────────────────────────


def evaluate_scalar(node: ast.AST, env: Mapping[str, float | None]) -> float | None:
    """Evaluate *node* as a scalar over *env* (``COMPUTED_METRIC`` semantics).

    Names resolve against *env* (a missing name yields ``None``).  ``None``
    propagates through every operator, and divide-/mod-by-zero yield ``None``
    so a single bad operand doesn't sink the whole metric.
    """
    if isinstance(node, ast.Expression):
        return evaluate_scalar(node.body, env)
    if isinstance(node, ast.BinOp):
        left = evaluate_scalar(node.left, env)
        right = evaluate_scalar(node.right, env)
        if left is None or right is None:
            return None
        op = node.op
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.Div):
            if right == 0:
                return None
            return left / right
        if isinstance(op, ast.Pow):
            return float(left**right)
        if isinstance(op, ast.Mod):
            if right == 0:
                return None
            return left % right
        msg = f"Unsupported binary operator: {type(op).__name__}"
        raise ValueError(msg)
    if isinstance(node, ast.UnaryOp):
        operand_value = evaluate_scalar(node.operand, env)
        if operand_value is None:
            return None
        if isinstance(node.op, ast.USub):
            return -operand_value
        if isinstance(node.op, ast.UAdd):
            return operand_value
        msg = f"Unsupported unary operator: {type(node.op).__name__}"
        raise ValueError(msg)
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, int | float):
            return float(value)
        msg = f"Unsupported literal: {value!r}"
        raise ValueError(msg)
    if isinstance(node, ast.Name):
        return env.get(node.id)
    msg = f"Unsupported expression node: {type(node).__name__}"
    raise ValueError(msg)


_SERIES_BINOP_FUNCS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def evaluate_series(node: ast.AST, operands: Mapping[str, np.ndarray]) -> Any:
    """Evaluate *node* element-wise over *operands* (``TIME_SERIES_DERIVED``).

    Names resolve to the numpy arrays in *operands*; operators apply
    element-wise (numpy broadcasting included) and constants become floats.
    """
    if isinstance(node, ast.Name):
        return operands[node.id]
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -evaluate_series(node.operand, operands)
    if isinstance(node, ast.BinOp) and type(node.op) in _SERIES_BINOP_FUNCS:
        left = evaluate_series(node.left, operands)
        right = evaluate_series(node.right, operands)
        return _SERIES_BINOP_FUNCS[type(node.op)](left, right)
    msg = (
        "TIME_SERIES_DERIVED expression contains an unsupported "
        f"construct at runtime: {type(node).__name__}"
    )
    raise ValueError(msg)
