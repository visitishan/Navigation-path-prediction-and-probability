"""
markovpath.attribution
~~~~~~~~~~~~~~~~~~~~~~

Markov Chain Multi-Touch Attribution (MTA) and Removal Effect modeling.
Computes the exact analytic contribution of individual touchpoints/pages
to final conversions using Absorbing Markov Chains.
"""

from collections import Counter, defaultdict
import numpy as np
import pandas as pd


def compute_removal_effect(paths, conversions=None, conversion_state="conversion", exit_state="exit", start_state="start"):
    """
    Computes the Removal Effect and normalized attribution weights for each
    touchpoint in user paths using Absorbing Markov Chain theory.

    Parameters
    ----------
    paths : list of lists or pd.Series
        Collection of navigation paths. Each path is a sequence of states.
    conversions : list or pd.Series, optional
        Binary flags (1 for conversion, 0 for non-conversion).
        If provided and path doesn't already contain conversion/exit tokens,
        they will be automatically appended.
    conversion_state : str, default 'conversion'
        Identifier for the successful conversion absorbing state.
    exit_state : str, default 'exit'
        Identifier for the drop-off/abandonment absorbing state.
    start_state : str, default 'start'
        Identifier for the starting state.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ['channel', 'removal_effect', 'attribution_weight', 'conversions_attributed']
    """
    # 1. Standardize paths
    processed_paths = []
    has_conv_flags = conversions is not None and len(conversions) == len(paths)

    for idx, raw_p in enumerate(paths):
        p = list(raw_p)
        if len(p) == 0:
            continue
        if p[0] != start_state:
            p = [start_state] + p

        if has_conv_flags:
            conv_flag = conversions[idx] if isinstance(conversions, (list, tuple)) else conversions.iloc[idx]
            target_end = conversion_state if conv_flag == 1 else exit_state
            if p[-1] not in (conversion_state, exit_state):
                p = p + [target_end]
        processed_paths.append(p)

    if not processed_paths:
        return pd.DataFrame(columns=['channel', 'removal_effect', 'attribution_weight', 'conversions_attributed'])

    # 2. Count transitions
    transition_counts = defaultdict(Counter)
    all_states = set()

    for p in processed_paths:
        for i in range(len(p) - 1):
            s_from = p[i]
            s_to = p[i + 1]
            transition_counts[s_from][s_to] += 1
            all_states.add(s_from)
            all_states.add(s_to)

    # Identifying absorbing states and transient states
    absorbing_states = [s for s in [conversion_state, exit_state] if s in all_states]
    if conversion_state not in absorbing_states:
        # No conversions present
        return pd.DataFrame(columns=['channel', 'removal_effect', 'attribution_weight', 'conversions_attributed'])

    transient_states = [s for s in sorted(list(all_states)) if s not in absorbing_states and s != start_state]
    transient_states_with_start = [start_state] + transient_states

    # 3. Helper to calculate conversion probability from start using Absorbing Markov Chain:
    # Transition matrix format:
    # P = [ Q   R ]
    #     [ 0   I ]
    # Fundamental matrix: N = (I - Q)^(-1)
    # Absorption probabilities: B = N * R
    def calculate_conversion_rate(active_transient_states, removed_state=None):
        m = len(active_transient_states)
        state_idx = {s: i for i, s in enumerate(active_transient_states)}
        
        Q = np.zeros((m, m), dtype=float)
        R = np.zeros((m, len(absorbing_states)), dtype=float)
        abs_idx = {s: i for i, s in enumerate(absorbing_states)}

        for s_from in active_transient_states:
            if s_from == removed_state:
                # If state is removed, transitions route to exit (drop-off)
                if exit_state in abs_idx:
                    R[state_idx[s_from], abs_idx[exit_state]] = 1.0
                continue

            out_counts = transition_counts.get(s_from, {})
            total_out = sum(out_counts.values())
            if total_out == 0:
                # Default absorbing to exit
                if exit_state in abs_idx:
                    R[state_idx[s_from], abs_idx[exit_state]] = 1.0
                continue

            for s_to, cnt in out_counts.items():
                prob = cnt / total_out
                if s_to in state_idx:
                    Q[state_idx[s_from], state_idx[s_to]] = prob
                elif s_to in abs_idx:
                    R[state_idx[s_from], abs_idx[s_to]] = prob

        # Fundamental matrix N = inv(I - Q)
        try:
            I = np.eye(m)
            N = np.linalg.inv(I - Q)
            B = np.dot(N, R)
            conv_idx = abs_idx[conversion_state]
            start_i = state_idx[start_state]
            return float(B[start_i, conv_idx])
        except np.linalg.LinAlgError:
            return 0.0

    # Baseline conversion rate
    base_cr = calculate_conversion_rate(transient_states_with_start)
    if base_cr <= 1e-9:
        # Zero baseline conversion
        return pd.DataFrame({
            'channel': transient_states,
            'removal_effect': [0.0] * len(transient_states),
            'attribution_weight': [0.0] * len(transient_states),
            'conversions_attributed': [0.0] * len(transient_states),
        })

    # Total observed conversions count
    total_conversions = sum(
        1 for p in processed_paths if len(p) > 0 and p[-1] == conversion_state
    )

    # Calculate removal effect for each channel
    removal_effects = {}
    for state in transient_states:
        cr_without_state = calculate_conversion_rate(transient_states_with_start, removed_state=state)
        # Drop in conversion rate
        re = max(0.0, (base_cr - cr_without_state) / base_cr)
        removal_effects[state] = re

    total_re = sum(removal_effects.values())
    
    records = []
    for state in transient_states:
        re = removal_effects[state]
        weight = (re / total_re) if total_re > 0 else (1.0 / len(transient_states) if transient_states else 0.0)
        attributed_conversions = weight * total_conversions
        records.append({
            'channel': state,
            'removal_effect': re,
            'attribution_weight': weight,
            'conversions_attributed': attributed_conversions
        })

    result_df = pd.DataFrame(records)
    if not result_df.empty:
        result_df = result_df.sort_values(by='attribution_weight', ascending=False).reset_index(drop=True)
    return result_df
