"""Shared arithmetic-expression front-end for the DSL decorators.

:class:`~bestee_compute.stocks.decorators.ComputedMetricDecorator` (scalar
metrics) and :class:`~bestee_compute.stocks.decorators.TimeSeriesDerivedDecorator`
(element-wise series) both turn a small Python arithmetic expression into a
validated AST and discover the names it references.  That parse → validate →
collect-names step is identical apart from which operators each dialect allows
and the label used in error messages, so it lives here and is configured per
caller with an :class:`ExpressionGrammar`.

Evaluation deliberately stays in each decorator: the numeric semantics differ
(``None``-propagating scalars with divide-by-zero guards vs. element-wise numpy
arrays), so a single shared evaluator would be more convoluted than two
purpose-built ones.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


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
