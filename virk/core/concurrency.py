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
        if max_queue_size < 1:
            raise ValueError("max_queue_size must be positive")
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="virk-worker")
        self._thread.start()
        
        # Ensure cleanup on exit
        atexit.register(self.shutdown)

    def submit(self, func: Callable, *args, **kwargs) -> bool:
        """
        Submit a task to the background worker.
        Returns True if accepted, False if queue is full (load shedding).
        """
        with self._state_lock:
            if self._stop_event.is_set():
                return False
            try:
                self._queue.put_nowait((func, args, kwargs))
                return True
            except queue.Full:
                logger.warning("VIRK background queue full. Dropping task.")
                return False

    def _worker_loop(self):
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                # Timeout allows checking stop_event periodically
                item = self._queue.get(timeout=1.0)
                func, args, kwargs = item
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    logger.exception("Error in background task %r: %s", func, e)
                finally:
                    self._queue.task_done()
            except queue.Empty:
                continue

    def shutdown(self, wait: bool = True):
        """Reject new work and drain accepted tasks; optionally wait for completion."""
        with self._state_lock:
            self._stop_event.set()
        if wait and threading.current_thread() is not self._thread:
            self._thread.join()
            atexit.unregister(self.shutdown)
