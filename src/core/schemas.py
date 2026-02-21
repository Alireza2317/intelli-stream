"""
Pydantic Schemas for Intelli-Stream.

This module defines the core data structures used for passing information
between different parts of the application, such as the inference engine
and the UI.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class BoundingBox(BaseModel):
	"""Represents a single detected object."""

	x1: int
	y1: int
	x2: int
	y2: int
	confidence: float
	class_name: str


class Detections(BaseModel):
	"""Represents a collection of detected objects in a single frame."""

	items: list[BoundingBox]

	@classmethod
	def from_ultralytics(cls, results: Any) -> Detections:
		"""
		Factory method to create a Detections object from ultralytics results.
		"""
		boxes = results.boxes
		names = results.names

		detected_items = [
			BoundingBox(
				x1=int(box.xyxy[0][0]),
				y1=int(box.xyxy[0][1]),
				x2=int(box.xyxy[0][2]),
				y2=int(box.xyxy[0][3]),
				confidence=float(box.conf[0]),
				class_name=names[int(box.cls[0])],
			)
			for box in boxes
		]
		return cls(items=detected_items)

