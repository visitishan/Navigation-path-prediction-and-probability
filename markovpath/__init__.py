"""
markovpath
~~~~~~~~~~

A fast, production-ready Python library for navigation path prediction,
sequence probability estimation, and Markov attribution modeling.

Rooted in research:
Sarukkai, R. R. (2000). Link prediction and path analysis using Markov chains.
Computer Networks, 33(1-6), 377-386.
"""

from .model import MarkovPath
from .attribution import compute_removal_effect

__version__ = "2.0.0"
__all__ = ["MarkovPath", "compute_removal_effect"]
