"""
The InferenceWorker runs the core InferenceEngine in a separate, non-blocking thread,
ensuring the main application (e.g., GUI) remains responsive.
"""

from __future__ import annotations

import queue
import threading
from types import TracebackType

from src.config.config import InferenceConfig
from src.core.inference import InferenceEngine
from src.core.schemas import Detections
from src.core.types import Frame, InferenceResult, InferenceTask


class InferenceWorker:
	"""
	A thread-safe worker that runs inference in the background.
	Can be used as a context manager to ensure cleanup.
	"""

	def __init__(self, engine: InferenceEngine):
		self.engine = engine
		self.input_queue: queue.Queue[InferenceTask] = queue.Queue(maxsize=1)
		self.output_queue: queue.Queue[InferenceResult] = queue.Queue(maxsize=1)
		self._stop_event = threading.Event()
		self._thread = threading.Thread(target=self._run, daemon=True)

	def __enter__(self) -> InferenceWorker:
		"""Starts the worker when entering the context."""
		self.start()
		return self

	def __exit__(
		self,
		exc_type: type[BaseException] | None,
		exc_val: BaseException | None,
		exc_tb: TracebackType | None,
	) -> None:
		"""Stops the worker when exiting the context."""
		self.stop()

	def _run(self) -> None:
		"""The main loop of the inference thread."""
		while not self._stop_event.is_set():
			try:
				# Wait for a new task to arrive
				task: InferenceTask = self.input_queue.get(timeout=1)
			except queue.Empty:
				print("Waiting for an inference task...")
				continue

			# Perform inference using the config from the task
			detections: Detections = self.engine.predict(task)

			# Draw detections on the frame
			annotated_frame = self.engine.get_annotated_frame(task.frame, detections)

			# Create the result object
			result: InferenceResult = InferenceResult(
				frame=annotated_frame, detections=detections
			)

			# Clear the output queue and put the new result
			if not self.output_queue.empty():
				try:
					# Clear it
					self.output_queue.get_nowait()
				except queue.Empty:
					# Unlikely to reach here!
					pass
			self.output_queue.put(result)

	def start(self):
		"""Starts the background inference thread."""
		if not self._thread.is_alive():
			self._thread.start()

	def stop(self):
		"""Stops the background inference thread."""
		self._stop_event.set()
		if self._thread.is_alive():
			self._thread.join()

	def schedule_task(self, frame: Frame, config: InferenceConfig) -> None:
		"""
		Submits a new frame and its config for processing.
		This is non-blocking. If the input queue is full, it will drop the task.
		"""
		if self.input_queue.full():
			# Drop the frame to avoid latency buildup
			return
		task = InferenceTask(frame=frame, config=config)
		self.input_queue.put_nowait(task)

	def get_latest_result(self) -> InferenceResult | None:
		"""
		Returns the latest processed frame and its detections.
		This is a non-blocking operation.
		"""
		try:
			return self.output_queue.get_nowait()
		except queue.Empty:
			return None
