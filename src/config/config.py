from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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


# Main Application Settings
class AppConfig(BaseSettings):
	"""
	Application configuration.
	Loads settings from config.toml.
	Allows overriding with environment variables.
	"""

	model_config = SettingsConfigDict(
		env_prefix="APP_", env_nested_delimiter="__", toml_file="config.toml"
	)

	model: ModelConfig = Field(default_factory=ModelConfig)
	real_time_mode: InferenceConfig = Field(default_factory=RealTimeConfig)
	efficiency_mode: InferenceConfig = Field(default_factory=EfficiencyConfig)
	rtsp_url: str | None = Field(
		default=None, description="RTSP URL for IP camera streams."
	)


# Create a single, project-wide instance of the settings.
# This instance is imported and used across the application.
settings = AppConfig()
