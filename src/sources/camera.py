"""
Camera Source Management.

This module provides a unified interface for capturing video frames from
various sources, such as USB webcams and RTSP streams.
"""

from __future__ import annotations

import os
from abc import ABC
from types import TracebackType

import cv2

from src.config.config import AppConfig, CameraConfig
from src.core.types import Frame


class Camera(ABC):
	"""Abstract base class for all camera sources."""

	def __init__(self, source: str | int):
		self._source = source
		self._capture = cv2.VideoCapture(self._source)

	def __enter__(self) -> Camera:
		"""Context manager entry point, returns the instance."""
		if not self.is_opened():
			raise ConnectionError(f"Failed to open camera source: {self._source}")
		return self

	def __exit__(
		self,
		exc_type: type[BaseException] | None,
		exc_val: BaseException | None,
		exc_tb: TracebackType | None,
	) -> None:
		"""Context manager exit point, ensures the camera is released."""
		self.release()

	def is_opened(self) -> bool:
		"""Checks if the camera stream is open."""
		return self._capture.isOpened()

	def read(self) -> Frame | None:
		"""
		Reads a single frame from the camera.

		Returns:
		    A numpy array representing the frame, or None if the frame
		    could not be read.
		"""
		success, frame = self._capture.read()
		return frame if success else None

	def release(self) -> None:
		"""Releases the camera resource."""
		if self._capture.isOpened():
			self._capture.release()

	def __repr__(self) -> str:
		return f"<{self.__class__.__name__} at {self._source}>"


class USBCamera(Camera):
	"""Represents a standard USB webcam."""

	def __init__(self, config: CameraConfig):
		super().__init__(config.device_id)
		height, width = config.resolution
		# Set camera resolution from the config
		self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
		self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


class RTSPCamera(Camera):
	"""Represents an RTSP network camera stream."""

	def __init__(self, rtsp_url: str):
		super().__init__(rtsp_url)


def get_camera(config: AppConfig) -> Camera:
	"""
	Factory function to get the appropriate camera source.

	Decides whether to use a USB camera or an RTSP stream based on
	the application configuration.

	Args:
	    config: The application's configuration object.

	Returns:
	    An initialized Camera object.
	"""
	camera_config = config.camera
	if camera_config.source_type == "rtsp":
		if not config.rtsp_url:
			raise ValueError(
				"RTSP source is selected but no rtsp_url is provided in the config."
			)

		# Set the transport protocol environment variable if specified
		if camera_config.rtsp_transport == "tcp":
			os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
		else:
			if "OPENCV_FFMPEG_CAPTURE_OPTIONS" in os.environ:
				del os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]

		print(f"Connecting to RTSP stream: {config.rtsp_url}")
		return RTSPCamera(config.rtsp_url)

	print(
		f"Connecting to local USB camera (device {camera_config.device_id}) "
		f"at {camera_config.resolution[1]}x{camera_config.resolution[0]}"
	)
	return USBCamera(camera_config)
