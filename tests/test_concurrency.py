import time
import pytest
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
