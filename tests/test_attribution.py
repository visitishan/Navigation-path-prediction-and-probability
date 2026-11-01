"""
Unit tests for markovpath.attribution (Removal Effect)
"""

import math
import pandas as pd
from markovpath import MarkovPath, compute_removal_effect


def test_compute_removal_effect():
    """Verify Removal Effect calculates valid attribution weights."""
    paths = [
        ["Facebook", "Google", "Checkout"],
        ["Google", "Checkout"],
        ["Organic", "Checkout"],
        ["Facebook", "Organic"]
    ]
    conversions = [1, 1, 1, 0]

    attr_df = compute_removal_effect(paths, conversions=conversions)

    assert not attr_df.empty
    assert "channel" in attr_df.columns
    assert "removal_effect" in attr_df.columns
    assert "attribution_weight" in attr_df.columns
    assert "conversions_attributed" in attr_df.columns

    # Attribution weights must sum to 1.0
    total_weight = attr_df["attribution_weight"].sum()
    assert math.isclose(total_weight, 1.0, rel_tol=1e-5)

    # Attributed conversions must sum to total conversions (3)
    total_attributed = attr_df["conversions_attributed"].sum()
    assert math.isclose(total_attributed, 3.0, rel_tol=1e-5)


def test_model_removal_effect_wrapper():
    """Verify MarkovPath.removal_effect wrapper."""
    paths = [
        ["start", "A", "B", "conversion"],
        ["start", "B", "conversion"],
        ["start", "A", "exit"]
    ]
    model = MarkovPath()
    model.fit(paths)

    attr = model.removal_effect(paths)
    assert "channel" in attr.columns
    assert len(attr) >= 2
