"""
CRNs - Chemical Reaction Networks Package

This package provides tools for analyzing chemical reaction networks,
including conservation law computation, network simulation, and analysis.
"""

from .reaction_network import ReactionNetwork
from .simulator import ReactionNetworkSimulator
from .analysis import *
from .generation import *
from .utils import *
from .sgolay2 import *

# Automatically build __all__ from all imported names
__all__ = [name for name in globals() if not name.startswith('_')]

__version__ = '1.0.0' 