"""
Custom Types for Intelli-Stream.

This module defines common, project-wide type aliases and simple data structures
to improve code clarity and maintainability.
"""
from typing import NamedTuple

import numpy as np

from src.core.schemas import Detections

# Type for a single video frame
type Frame = np.ndarray

# Data structure for results passed from the inference worker thread
class InferenceResult(NamedTuple):
    """Data structure for results from the inference worker."""
    frame: Frame
    detections: Detections
