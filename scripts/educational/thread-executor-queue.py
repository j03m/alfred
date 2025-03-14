import concurrent.futures
import queue
import time
import random
import threading

def worker_function(task_queue, result_queue):
    """Worker thread function. Get a task from the queue and process it."""
    while True:
        try:
            task = task_queue.get(timeout=1)  # Get a task with a timeout
        except queue.Empty:
            break  # No more tasks
        print(f"Worker processing task: {task}")
        # Simulate some work
        time.sleep(random.uniform(0.5, 1.5))
        result = f"Result of {task}"
        result_queue.put(result)  # Add result to the result queue
        task_queue.task_done()  # Signal that the task is complete


def producer(task_queue, num_tasks):
    """Producer thread function. Adds tasks to the queue."""
    for i in range(num_tasks):
        task = f"Task {i}"
        print(f"Producer adding task: {task}")
        task_queue.put(task)
    task_queue.join()  # Wait for the queue to be empty before stopping


if __name__ == '__main__':
    num_tasks = 10
    num_workers = 3

    task_queue = queue.Queue()  # Queue for pending tasks
    result_queue = queue.Queue()  # Queue for results

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:

        # Start the producer thread
        producer_thread = threading.Thread(target=producer, args=(task_queue, num_tasks))
        producer_thread.start()

        # Start worker threads to consume tasks from the queue
        for _ in range(num_workers):
            executor.submit(worker_function, task_queue, result_queue)

        # Handle the results
        while not result_queue.empty():
            result = result_queue.get()
            print(f"Received result: {result}")

        producer_thread.join()
