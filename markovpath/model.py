"""
markovpath.model
~~~~~~~~~~~~~~~~

Core MarkovPath implementation for sequence modeling, trail probability
estimation, and next-link pattern prediction using First-Order Markov Chains.

Theoretical Reference:
- Sarukkai, R. R. (2000). Link prediction and path analysis using Markov chains.
  Computer Networks, 33(1-6), 377-386.
"""

from collections import Counter, defaultdict
import math
import numpy as np
import pandas as pd
from .attribution import compute_removal_effect


class MarkovPath:
    """
    First-Order Markov Chain model for user navigation paths, clickstream analysis,
    and sequence likelihood estimation.

    Parameters
    ----------
    smoothing : float, default 0.0
        Additive Laplace smoothing factor (alpha) to avoid zero-probability
        for unseen state transitions.
    allow_self_loops : bool, default True
        Whether to model transitions from a state to itself (A -> A).
    """

    def __init__(self, smoothing=0.0, allow_self_loops=True):
        if smoothing < 0:
            raise ValueError("smoothing parameter must be non-negative")
        self.smoothing = float(smoothing)
        self.allow_self_loops = bool(allow_self_loops)

        # Internal state
        self.states_ = []
        self.state_probabilities_ = {}
        self.transition_matrix_ = pd.DataFrame()
        self._transitions = defaultdict(dict)
        self.is_fitted_ = False

    def fit(self, paths, conversions=None, add_start_end=False,
            start_token="start", conversion_token="conversion", exit_token="exit"):
        """
        Fit the Markov chain model on a collection of navigation paths.

        Parameters
        ----------
        paths : list of lists or pd.Series
            Navigation paths. Each path is a list/sequence of state identifiers.
        conversions : list or pd.Series, optional
            Conversion indicators (1 for converted, 0 for dropped off).
        add_start_end : bool, default False
            If True, automatically adds start_token and conversion_token/exit_token
            to each path.
        start_token : str, default 'start'
            Start state token to prepend when add_start_end=True.
        conversion_token : str, default 'conversion'
            Successful conversion token to append when conversions=1.
        exit_token : str, default 'exit'
            Exit token to append when conversions=0.

        Returns
        -------
        self : MarkovPath
            Fitted estimator.
        """
        processed_paths = []
        has_conv = conversions is not None and len(conversions) == len(paths)

        for idx, raw_p in enumerate(paths):
            p = list(raw_p)
            if not p:
                continue
            if add_start_end:
                if p[0] != start_token:
                    p = [start_token] + p
                if has_conv:
                    c = conversions[idx] if isinstance(conversions, (list, tuple)) else conversions.iloc[idx]
                    end_token = conversion_token if c == 1 else exit_token
                    if p[-1] != end_token:
                        p = p + [end_token]
            processed_paths.append(p)

        if not processed_paths:
            raise ValueError("Input paths cannot be empty.")

        # 1. State Frequencies (Prior / Empirical State Probability)
        all_states_flat = []
        transition_counts = defaultdict(Counter)

        for path in processed_paths:
            all_states_flat.extend(path)
            for i in range(len(path) - 1):
                s_from = path[i]
                s_to = path[i + 1]
                if not self.allow_self_loops and s_from == s_to:
                    continue
                transition_counts[s_from][s_to] += 1

        state_counter = Counter(all_states_flat)
        total_steps = len(all_states_flat)
        self.state_probabilities_ = {s: cnt / total_steps for s, cnt in state_counter.items()}
        self.states_ = sorted(list(state_counter.keys()))
        num_states = len(self.states_)

        # 2. Transition Probability Matrix
        # P(j | i) = (N(i -> j) + alpha) / (sum_k N(i -> k) + |V| * alpha)
        matrix = pd.DataFrame(0.0, index=self.states_, columns=self.states_)
        self._transitions = defaultdict(dict)

        for s_from in self.states_:
            out_dict = transition_counts[s_from]
            total_out = sum(out_dict.values())
            denom = total_out + (num_states * self.smoothing)

            for s_to in self.states_:
                cnt = out_dict.get(s_to, 0)
                if denom > 0:
                    prob = (cnt + self.smoothing) / denom
                else:
                    # If state has no outbound transitions (e.g. terminal state), prob is 0
                    prob = 0.0

                if prob > 0:
                    matrix.loc[s_from, s_to] = prob
                    self._transitions[s_from][s_to] = prob

        self.transition_matrix_ = matrix
        self.is_fitted_ = True
        return self

    def _check_is_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted yet. Call .fit() before making predictions.")

    @property
    def transition_matrix(self):
        """Returns the transition probability matrix as a pandas DataFrame."""
        self._check_is_fitted()
        return self.transition_matrix_

    @property
    def state_probabilities(self):
        """Returns the empirical state probabilities as a dictionary."""
        self._check_is_fitted()
        return dict(self.state_probabilities_)

    def predict_next(self, current_state, top_k=None, return_proba=True):
        """
        Predict the most likely next states given current_state.

        Parameters
        ----------
        current_state : str or any
            The current active state/page.
        top_k : int, optional
            Number of top candidates to return. If None, returns all non-zero transitions.
        return_proba : bool, default True
            If True, returns list of (state, probability) tuples.
            If False, returns list of state names.

        Returns
        -------
        list
            Ranked list of predicted next states.
        """
        self._check_is_fitted()
        out_probs = self._transitions.get(current_state, {})
        if not out_probs:
            return []

        # Sort descending by probability
        ranked = sorted(out_probs.items(), key=lambda item: item[1], reverse=True)
        if top_k is not None:
            ranked = ranked[:top_k]

        if return_proba:
            return ranked
        return [state for state, _ in ranked]

    def score_path(self, path, log_scale=False):
        """
        Compute the probability of a path / sequence using the chain rule:
        P(Path) = P(S_0) * Prod_{t=0}^{T-1} P(S_{t+1} | S_t)

        Parameters
        ----------
        path : list of str
            Sequence of states.
        log_scale : bool, default False
            If True, returns the log-likelihood (ln P(Path)), preventing numerical
            underflow on long sequences.

        Returns
        -------
        float
            Path probability or log-likelihood.
        """
        self._check_is_fitted()
        if not path:
            return -math.inf if log_scale else 0.0

        first_state = path[0]
        # Initial prior state probability
        p0 = self.state_probabilities_.get(first_state, self.smoothing if self.smoothing > 0 else 1e-12)

        if log_scale:
            total_log_prob = math.log(max(p0, 1e-15))
            for i in range(len(path) - 1):
                s_from = path[i]
                s_to = path[i + 1]
                trans_p = self._transitions.get(s_from, {}).get(s_to, self.smoothing if self.smoothing > 0 else 0.0)
                if trans_p <= 0.0:
                    return -math.inf
                total_log_prob += math.log(trans_p)
            return total_log_prob
        else:
            total_prob = p0
            for i in range(len(path) - 1):
                s_from = path[i]
                s_to = path[i + 1]
                trans_p = self._transitions.get(s_from, {}).get(s_to, self.smoothing if self.smoothing > 0 else 0.0)
                total_prob *= trans_p
                if total_prob == 0.0:
                    break
            return total_prob

    def score_paths(self, paths, log_scale=False):
        """
        Batch scoring of multiple paths.

        Parameters
        ----------
        paths : list of lists or pd.Series
            Collection of paths.
        log_scale : bool, default False
            Whether to return log-likelihoods.

        Returns
        -------
        list of float
            Scores for each input path.
        """
        return [self.score_path(p, log_scale=log_scale) for p in paths]

    def simulate(self, start_state, max_steps=20, stop_states=("exit", "conversion", "<end>"), seed=None):
        """
        Generate a synthetic navigation path using Monte Carlo sampling from the
        learned transition probabilities.

        Parameters
        ----------
        start_state : str
            The initial state to start simulation from.
        max_steps : int, default 20
            Maximum length of simulated path.
        stop_states : tuple or set
            Set of absorbing/terminal states that trigger sequence termination.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        list
            Simulated sequence of states.
        """
        self._check_is_fitted()
        rng = np.random.default_rng(seed)
        stop_set = set(stop_states) if stop_states else set()

        if start_state not in self.states_:
            raise ValueError(f"start_state '{start_state}' is not in trained states: {self.states_}")

        current_state = start_state
        trajectory = [current_state]

        for _ in range(max_steps - 1):
            if current_state in stop_set:
                break

            out_dist = self._transitions.get(current_state, {})
            if not out_dist:
                break

            states = list(out_dist.keys())
            probs = list(out_dist.values())
            # Renormalize to ensure exact sum to 1.0 for np.random.choice
            total_p = sum(probs)
            if total_p == 0:
                break
            norm_probs = [p / total_p for p in probs]

            next_state = rng.choice(states, p=norm_probs)
            trajectory.append(next_state)
            current_state = next_state

        return trajectory

    def removal_effect(self, paths, conversions=None, conversion_state="conversion", exit_state="exit", start_state="start"):
        """
        Compute Marketing Multi-Touch Attribution and Removal Effect for all touchpoints.

        Parameters
        ----------
        paths : list of lists or pd.Series
            User navigation paths.
        conversions : list or pd.Series, optional
            Conversion flags (1/0).
        conversion_state : str, default 'conversion'
            Conversion absorbing state.
        exit_state : str, default 'exit'
            Exit absorbing state.
        start_state : str, default 'start'
            Start state identifier.

        Returns
        -------
        pd.DataFrame
            Attribution table showing channel removal effect and normalized weights.
        """
        return compute_removal_effect(
            paths=paths,
            conversions=conversions,
            conversion_state=conversion_state,
            exit_state=exit_state,
            start_state=start_state
        )

    def to_dataframe(self, format="matrix"):
        """
        Export transition probabilities.

        Parameters
        ----------
        format : {'matrix', 'long'}, default 'matrix'
            'matrix' returns a 2D pivoted DataFrame (rows=from, cols=to).
            'long' returns a 3-column DataFrame ['from_state', 'to_state', 'probability'].

        Returns
        -------
        pd.DataFrame
        """
        self._check_is_fitted()
        if format == "matrix":
            return self.transition_matrix_.copy()
        elif format == "long":
            records = []
            for s_from, targets in self._transitions.items():
                for s_to, prob in targets.items():
                    records.append({
                        "from_state": s_from,
                        "to_state": s_to,
                        "probability": prob
                    })
            return pd.DataFrame(records)
        else:
            raise ValueError(f"Unknown format '{format}'. Choose 'matrix' or 'long'.")
