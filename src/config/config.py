from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import (
	BaseSettings,
	PydanticBaseSettingsSource,
	SettingsConfigDict,
	TomlConfigSettingsSource,
)


class ModelConfig(BaseModel):
	"""Configuration for the YOLO model."""

	name: str = Field(default="yolov8n.pt", description="Name of the model file.")
	path: Path | None = Field(
		default=None, description="Optional path to the model file."
	)
	quantized_name: str | None = Field(
		default=None,
		description="Name of the quantized model file (e.g., .tflite, .onnx).",
	)


class InferenceConfig(BaseModel):
	"""Configuration for the inference process."""

	confidence_threshold: float = Field(
		default=0.5, ge=0.0, le=1.0, description="Confidence threshold for detections."
	)

	iou_threshold: float = Field(
		default=0.4,
		ge=0.0,
		le=1.0,
		description="IOU threshold for Non-Maximum Suppression.",
	)

	resolution: tuple[int, int] = Field(
		default=(480, 640), description="Resolution for input frames (height, width)."
	)

	frame_skip: int = Field(
		default=0,
		ge=0,
		description="Number of frames to skip between inferences. 0 means process every frame.",
	)


class RealTimeConfig(InferenceConfig):
	"""Configuration for Real-Time (high-performance) mode."""

	resolution: tuple[int, int] = (720, 1280)
	frame_skip: int = 0


class EfficiencyConfig(InferenceConfig):
	"""Configuration for Efficiency (low-power) mode."""

	resolution: tuple[int, int] = (320, 320)
	frame_skip: int = 5
	confidence_threshold: float = 0.4


class CameraConfig(BaseModel):
	"""Configuration for camera sources."""

	source_type: str = Field(
		default="usb", description='Type of camera source. Options: "usb", "rtsp".'
	)
	device_id: int = Field(default=0, description="Device ID for USB camera.")
	resolution: tuple[int, int] = Field(
		default=(720, 1280),
		description="Capture resolution [height, width] for USB camera.",
	)
	rtsp_transport: str = Field(
		default="tcp", description="Transport protocol for RTSP: 'tcp' or 'udp'."
	)
	connection_timeout: int = Field(
		default=10, description="Timeout in seconds for camera connection."
	)


# Main Application Settings
class AppConfig(BaseSettings):
	"""
	Application configuration.
	Loads settings from config.toml.
	Allows overriding with environment variables.
	"""

	model_config = SettingsConfigDict(env_prefix="APP_", env_nested_delimiter="__")

	model: ModelConfig = Field(default_factory=ModelConfig)
	real_time_mode: InferenceConfig = Field(default_factory=RealTimeConfig)
	efficiency_mode: InferenceConfig = Field(default_factory=EfficiencyConfig)
	camera: CameraConfig = Field(default_factory=CameraConfig)
	rtsp_url: str | None = Field(
		default=None, description="RTSP URL for IP camera streams."
	)

	@classmethod
	def settings_customise_sources(
		cls,
		settings_cls: type[BaseSettings],
		init_settings: PydanticBaseSettingsSource,
		env_settings: PydanticBaseSettingsSource,
		dotenv_settings: PydanticBaseSettingsSource,
		file_secret_settings: PydanticBaseSettingsSource,
	) -> tuple[PydanticBaseSettingsSource, ...]:
		return (
			TomlConfigSettingsSource(
				settings_cls=settings_cls, toml_file="config.toml"
			),
			init_settings,
			env_settings,
			dotenv_settings,
			file_secret_settings,
		)


# Create a single, project-wide instance of the settings.
# This instance is imported and used across the application.
settings = AppConfig()
