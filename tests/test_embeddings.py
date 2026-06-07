import numpy as np
import pytest

from bestee_chat.embeddings import cosine_scores


def test_scores_each_row() -> None:
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    scores = cosine_scores(matrix, np.array([1.0, 0.0]))
    assert scores.shape == (2,)
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0)


def test_is_scale_invariant() -> None:
    scores = cosine_scores(np.array([[2.0, 0.0]]), np.array([5.0, 0.0]))
    assert scores[0] == pytest.approx(1.0)


def test_zero_row_scores_zero() -> None:
    scores = cosine_scores(np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([1.0, 0.0]))
    assert scores[0] == pytest.approx(0.0)


def test_empty_matrix_returns_empty() -> None:
    scores = cosine_scores(np.zeros((0, 0)), np.array([1.0, 0.0]))
    assert scores.shape == (0,)


def test_scores_stay_in_range() -> None:
    matrix = np.random.default_rng(0).normal(size=(6, 8))
    scores = cosine_scores(matrix, np.ones(8))
    assert scores.shape == (6,)
    assert np.all(scores <= 1.0 + 1e-9)
    assert np.all(scores >= -1.0 - 1e-9)
