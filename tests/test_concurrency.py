import time
import pytest
import threading
from virk.core.concurrency import AsyncWorker

def test_async_worker_basic():
    worker = AsyncWorker(max_queue_size=10)
    
    result = []
    
    def task(x):
        result.append(x * 2)
        
    # Submit task
    assert worker.submit(task, 10)
    assert worker.submit(task, 20)
    
    # Give time to process
    time.sleep(0.5)
    
    worker.shutdown()
    
    assert 20 in result
    assert 40 in result
    assert len(result) == 2

def test_async_worker_queue_full():
    # Tiny queue to force overflow
    worker = AsyncWorker(max_queue_size=1)
    
    def slow_task():
        time.sleep(0.5)
        
    # Fill the worker (1 executing, 1 in queue)
    worker.submit(slow_task) 
    worker.submit(slow_task)
    
    # Should overflow now? Maybe queue.Queue(1) holds 1 item.
    # We submit fast.
    
    # Try submitting many
    rejected = False
    for _ in range(10):
        if not worker.submit(slow_task):
            rejected = True
            break
            
    worker.shutdown(wait=False)
    
    assert rejected, "Worker should reject tasks when full"

def test_async_worker_exception_handling():
    worker = AsyncWorker()
    
    def bad_task():
        raise ValueError("Boom")
        
    # Should not crash the thread
    worker.submit(bad_task)
    time.sleep(0.1)
    
    result = []
    def good_task():
        result.append("OK")
        
    worker.submit(good_task)
    time.sleep(0.1)
    
    worker.shutdown()
    
    assert result == ["OK"], "Worker thread should survive exceptions"


def test_shutdown_drains_accepted_work():
    started, release = threading.Event(), threading.Event()
    worker = AsyncWorker()
    result = []

    def blocked_task():
        started.set()
        assert release.wait(timeout=5)
        result.append("first")

    try:
        assert worker.submit(blocked_task)
        assert started.wait(timeout=5)
        assert worker.submit(result.append, "second")
        worker.shutdown(wait=False)
        assert not worker.submit(result.append, "rejected")
    finally:
        release.set()
        worker.shutdown()
    assert result == ["first", "second"]
    assert worker._queue.unfinished_tasks == 0


def test_failing_callable_object_does_not_kill_worker():
    class FailingTask:
        def __call__(self):
            raise ValueError("callable without __name__")

    worker = AsyncWorker()
    result = []
    assert worker.submit(FailingTask())
    assert worker.submit(result.append, "survived")
    worker.shutdown()
    assert result == ["survived"]
