"""
Unit tests for legacy path_nav compatibility
"""

import math
import pandas as pd
import path_nav as nv


def test_legacy_workflow():
    data = {
        "other_data": [1, 4, 5],
        "path": [
            ["A", "B", "C", "A", "C"],
            ["B", "D", "B", "A"],
            ["A", "C", "B", "A", "D"]
        ],
        "conversions": [0, 0, 1],
    }
    df = pd.DataFrame(data)

    # 1. State probability
    state_prob = nv.state_probability(df, "path")
    assert "State" in state_prob.columns
    assert "State_probability" in state_prob.columns
    assert math.isclose(state_prob["State_probability"].sum(), 1.0, rel_tol=1e-5)

    # 2. Add start end
    df_aug = nv.add_start_end(df, "path", "conversions")
    assert df_aug["path"].iloc[0][0] == "start"
    assert df_aug["path"].iloc[0][-1] == "exit"
    assert df_aug["path"].iloc[2][-1] == "conversion"

    # Original df should not be mutated
    assert df["path"].iloc[0][0] != "start"

    # 3. Transition probability
    trans_df = nv.transition_probability(df_aug, "path")
    assert "from_sitesection" in trans_df.columns
    assert "to_sitesection" in trans_df.columns
    assert "transition_probability" in trans_df.columns

    # Check sum for valid non-terminal states
    for state in ["start", "A", "B", "C", "D"]:
        sub = trans_df[trans_df["from_sitesection"] == state]
        assert math.isclose(sub["transition_probability"].sum(), 1.0, rel_tol=1e-5)

    # 4. Path probability
    path_prob_df = nv.path_probability(df_aug, "path", trans_df)
    assert "path_probability" in path_prob_df.columns
    assert (path_prob_df["path_probability"] > 0).all()
