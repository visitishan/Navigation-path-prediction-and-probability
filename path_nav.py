# -*- coding: utf-8 -*-
"""
path_nav (Legacy Compatibility Module)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This module provides backward-compatible functions for earlier versions of
the library, backed by the corrected and accelerated `markovpath` engine.

For modern code, use:
    from markovpath import MarkovPath
"""

import warnings
from ast import literal_eval
import pandas as pd
from markovpath import MarkovPath


def _warn_legacy():
    warnings.warn(
        "path_nav procedural functions are legacy. Please migrate to "
        "`from markovpath import MarkovPath` for better performance and API features.",
        DeprecationWarning,
        stacklevel=3
    )


def add_start_end(df, visit_path, leads_flag):
    """
    Adds dummy start and conversion/exit text value to visit_path based on leads_flag.
    Safely creates new lists without mutating original rows in place.

    :df: dataframe, dataframe that needs to be updated
    :visit_path: str, name of column which contains path data
    :leads_flag: str, name of column that has conversion flagged as 1

    Returns
    :df: dataframe, updated dataframe with start, conversion and exit values
    """
    _warn_legacy()
    out_df = df.copy()
    new_paths = []
    for _, row in out_df.iterrows():
        orig = list(row[visit_path])
        p = ["start"] + orig
        if row[leads_flag] == 1:
            p.append("conversion")
        else:
            p.append("exit")
        new_paths.append(p)
    out_df[visit_path] = new_paths
    return out_df


def get_unique_points(df, visit_path):
    """
    Returns the set of unique sitesections and the flattened list of all points.

    :df: dataframe, dataframe that has path
    :visit_path: str, name of column which contains path data

    Returns
    :x: set, unique values of sitesections in complete dataframe
    :complete_sitesections: list, all state occurrences
    """
    _warn_legacy()
    complete_sitesections = []
    for path in df[visit_path]:
        complete_sitesections.extend(list(path))
    x = set(complete_sitesections)
    return x, complete_sitesections


def transition_probability(df, visit_path, allow_self_loops=True):
    """
    Calculates exact transition probabilities between states.
    Guarantees sum of probabilities per from-state equals 1.0.

    :df: dataframe, dataframe that has path
    :visit_path: str, name of column which contains path data
    :allow_self_loops: bool, default True

    Returns
    :transition_df: dataframe with columns ['from_sitesection', 'to_sitesection', 'transition_probability']
    """
    _warn_legacy()
    paths = list(df[visit_path])
    model = MarkovPath(smoothing=0.0, allow_self_loops=allow_self_loops)
    model.fit(paths)

    records = []
    for s_from in model.states_:
        # Skip terminal absorption states as source
        if s_from in ("conversion", "exit"):
            continue
        for s_to in model.states_:
            if s_to == "start":
                continue
            prob = model.transition_matrix.loc[s_from, s_to]
            records.append({
                "from_sitesection": s_from,
                "to_sitesection": s_to,
                "transition_probability": float(prob)
            })

    return pd.DataFrame(records)


def path_probability(df, visit_path, transition_df=None):
    """
    Calculates path probability for each path in dataframe using the chain rule.

    :df: dataframe, dataframe that has path column
    :visit_path: str, name of column which contains path data
    :transition_df: dataframe, optional for legacy signature

    Returns
    :path_probability_df: dataframe with added 'path_probability' column
    """
    _warn_legacy()
    out_df = df.copy()
    paths = list(out_df[visit_path])
    model = MarkovPath(smoothing=1e-5)
    model.fit(paths)
    out_df["path_probability"] = model.score_paths(paths)
    return out_df


def state_probability(df, visit_path):
    """
    Calculates the prior probability distribution of states.

    :df: dataframe, dataframe that has path column
    :visit_path: str, name of column which contains path data

    Returns
    :state_probability: dataframe with ['State', 'State_probability']
    """
    _warn_legacy()
    paths = list(df[visit_path])
    model = MarkovPath()
    model.fit(paths)
    records = [{"State": s, "State_probability": p} for s, p in model.state_probabilities.items()]
    return pd.DataFrame(records)


def convert_to_list(df, visit_path):
    """Converts the path column string representation to Python list."""
    out_df = df.copy()
    out_df[visit_path] = out_df[visit_path].apply(
        lambda x: literal_eval(x) if isinstance(x, str) else x
    )
    return out_df


def convert_to_str(df, visit_path):
    """Converts the path column to string type."""
    out_df = df.copy()
    out_df[visit_path] = out_df[visit_path].astype(str)
    return out_df
