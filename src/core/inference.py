import cv2
import numpy as np
from ultralytics.models import YOLO

from src.config.config import InferenceConfig, ModelConfig
from src.core.schemas import Detections

type Frame = np.ndarray


class InferenceEngine:
	"""
	Handles loading the model and running inference.
	"""

	def __init__(self, model_config: ModelConfig):
		"""
		Initializes the InferenceEngine.

		Args:
		    model_config (ModelConfig): Configuration for the YOLO model.
		"""
		self.model_config = model_config
		self.model = self._load_model()

	def _load_model(self) -> YOLO:
		"""Loads the YOLO model."""
		model_path = self.model_config.path or self.model_config.name
		try:
			model = YOLO(str(model_path))
			print(f"Model '{model_path}' loaded successfully.")
			return model
		except Exception as e:
			print(f"Error loading model '{model_path}': {e}")
			raise

	def predict(self, frame: Frame, infer_config: InferenceConfig) -> Detections:
		"""
		Performs inference on a single frame.

		Args:
		    frame (np.ndarray): The input image/frame as a NumPy array.
		    infer_config (InferenceConfig): Configuration for the inference process.

		Returns:
		    Detections: A structured object containing all detected items.
		"""
		raw_results = self.model(
			frame,
			conf=infer_config.confidence_threshold,
			iou=infer_config.iou_threshold,
			imgsz=infer_config.resolution,
			verbose=False,
		)

		return Detections.from_ultralytics(raw_results[0])

	def draw_detections(self, frame: Frame, detections: Detections) -> Frame:
		"""
		Draws bounding boxes and labels on the frame.

		Args:
		    frame (np.ndarray): The frame to draw on.
		    detections (Detections): The structured detection results.

		Returns:
		    np.ndarray: The frame with detections drawn on it.
		"""
		annotated_frame: Frame = frame.copy()
		for box in detections.items:
			# Draw the bounding box
			cv2.rectangle(
				annotated_frame,
				(box.x1, box.y1),
				(box.x2, box.y2),
				color=(0, 255, 0),
				thickness=2,
			)

			# Prepare the label
			label: str = f"{box.class_name}: {box.confidence:.2f}"

			# Draw a filled rectangle behind the text for better readability
			(label_width, label_height), baseline = cv2.getTextSize(
				label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2
			)
			cv2.rectangle(
				annotated_frame,
				(box.x1, box.y1 - label_height - 10),
				(box.x1 + label_width, box.y1),
				(0, 255, 0),
				cv2.FILLED,
			)

			# Draw the text label
			cv2.putText(
				annotated_frame,
				text=label,
				org=(box.x1, box.y1 - 5),
				fontFace=cv2.FONT_HERSHEY_SIMPLEX,
				fontScale=0.5,
				color=(0, 0, 0),
				thickness=1,
			)
		return annotated_frame
