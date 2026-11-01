"""
Unit tests for markovpath.MarkovPath
"""

import math
import numpy as np
import pandas as pd
import pytest
from markovpath import MarkovPath


def test_transition_matrix_row_stochastic():
    """Verify that every non-terminal state's outbound probabilities strictly sum to 1.0."""
    paths = [
        ["A", "B", "C", "A", "C"],
        ["B", "D", "B", "A"],
        ["A", "C", "B", "A", "D"]
    ]
    model = MarkovPath(smoothing=0.0)
    model.fit(paths)

    matrix = model.transition_matrix
    # Check states with outbound transitions
    for state in ["A", "B", "C"]:
        row_sum = matrix.loc[state].sum()
        assert math.isclose(row_sum, 1.0, rel_tol=1e-5), f"Row sum for state {state} is {row_sum}, expected 1.0"


def test_state_probabilities_sum_to_one():
    """Verify that prior state probabilities sum to 1.0."""
    paths = [
        ["A", "B", "C"],
        ["B", "C", "D"],
        ["A", "D"]
    ]
    model = MarkovPath()
    model.fit(paths)

    probs = model.state_probabilities
    total_prob = sum(probs.values())
    assert math.isclose(total_prob, 1.0, rel_tol=1e-5)


def test_predict_next():
    """Verify top-k next page prediction and ranking."""
    paths = [
        ["Home", "Products", "Cart"],
        ["Home", "Products", "Cart"],
        ["Home", "Products", "Reviews"],
        ["Home", "About"]
    ]
    model = MarkovPath()
    model.fit(paths)

    # From Home: Products (3 times), About (1 time)
    preds = model.predict_next("Home", top_k=2)
    assert len(preds) == 2
    assert preds[0][0] == "Products"
    assert math.isclose(preds[0][1], 0.75, rel_tol=1e-5)
    assert preds[1][0] == "About"
    assert math.isclose(preds[1][1], 0.25, rel_tol=1e-5)

    # Return only state names without probabilities
    names_only = model.predict_next("Home", top_k=1, return_proba=False)
    assert names_only == ["Products"]


def test_score_path():
    """Verify joint path probability using chain rule."""
    paths = [
        ["A", "B", "C"],
        ["A", "B", "D"]
    ]
    model = MarkovPath()
    model.fit(paths)

    # P("A", "B", "C") = P(A) * P(B|A) * P(C|B)
    # A total visits: 2, B: 2, C: 1, D: 1. Total states = 6.
    # P(A) = 2/6 = 1/3
    # P(B|A) = 2/2 = 1.0
    # P(C|B) = 1/2 = 0.5
    # Expected path prob = (1/3) * 1.0 * 0.5 = 1/6 = 0.166666...
    score = model.score_path(["A", "B", "C"])
    assert math.isclose(score, 1.0 / 6.0, rel_tol=1e-5)

    # Log scale scoring
    log_score = model.score_path(["A", "B", "C"], log_scale=True)
    assert math.isclose(math.exp(log_score), score, rel_tol=1e-5)


def test_smoothing_on_unseen_transition():
    """Verify Laplace smoothing prevents zero crash on unseen transitions."""
    paths = [["A", "B"], ["B", "C"]]
    model = MarkovPath(smoothing=1.0)
    model.fit(paths)

    # Transition A -> C never occurred in data, but smoothing gives non-zero prob
    unseen_score = model.score_path(["A", "C"])
    assert unseen_score > 0.0


def test_simulation():
    """Verify Monte Carlo simulation generates valid sequence."""
    paths = [
        ["start", "A", "B", "exit"],
        ["start", "B", "A", "exit"]
    ]
    model = MarkovPath()
    model.fit(paths)

    sim = model.simulate("start", max_steps=10, stop_states={"exit"}, seed=42)
    assert sim[0] == "start"
    assert sim[-1] == "exit"
    assert len(sim) >= 3


def test_self_loops_toggle():
    """Verify whether self transitions are included or ignored."""
    paths = [["A", "A", "B"]]
    
    # Enabled
    m1 = MarkovPath(allow_self_loops=True).fit(paths)
    assert m1.transition_matrix.loc["A", "A"] > 0

    # Disabled
    m2 = MarkovPath(allow_self_loops=False).fit(paths)
    assert m2.transition_matrix.loc["A", "A"] == 0
