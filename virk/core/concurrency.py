import threading
import queue
import logging
import atexit
from typing import Callable, Any, Tuple

logger = logging.getLogger("virk.concurrency")

class AsyncWorker:
    """
    Background worker that consumes tasks from a queue.
    Used to offload IO-heavy or non-critical CPU tasks (bundling, fingerprinting)
    from the main inference thread.
    """
    
    def __init__(self, max_queue_size: int = 100):
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="virk-worker")
        self._thread.start()
        
        # Ensure cleanup on exit
        atexit.register(self.shutdown)

    def submit(self, func: Callable, *args, **kwargs) -> bool:
        """
        Submit a task to the background worker.
        Returns True if accepted, False if queue is full (load shedding).
        """
        if self._stop_event.is_set():
            return False
            
        try:
            self._queue.put_nowait((func, args, kwargs))
            return True
        except queue.Full:
            logger.warning("VIRK background queue full. Dropping task.")
            return False

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                # Timeout allows checking stop_event periodically
                item = self._queue.get(timeout=1.0)
                func, args, kwargs = item
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in background task {func.__name__}: {e}")
                finally:
                    self._queue.task_done()
            except queue.Empty:
                continue

    def shutdown(self, wait: bool = True):
        """Signal worker to stop and optionally wait for queue to drain."""
        self._stop_event.set()
        if wait:
            self._thread.join(timeout=5.0)
