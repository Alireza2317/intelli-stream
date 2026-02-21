"""
Custom Types for Intelli-Stream.

This module defines common, project-wide type aliases and simple data structures
to improve code clarity and maintainability.
"""

from typing import NamedTuple

import numpy as np

from src.config.config import InferenceConfig
from src.core.schemas import Detections

# Type for a single video frame
type Frame = np.ndarray


# Data structure for inputs passed to the inference worker threa
class InferenceTask(NamedTuple):
	"""Data structure for inputs to the inference worker."""

	frame: Frame
	config: InferenceConfig


# Data structure for results passed from the inference worker thread
class InferenceResult(NamedTuple):
	"""Data structure for results from the inference worker."""

	frame: Frame
	detections: Detections
