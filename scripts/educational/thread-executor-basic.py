import concurrent.futures
import time

def task(n):
    """Simulates a time-consuming task."""
    print(f"Task {n} starting")
    time.sleep(2)
    print(f"Task {n} finished")
    return f"Result of task {n}"

if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit tasks to the executor
        futures = [executor.submit(task, i) for i in range(5)]

        # Process the results as they become available
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                print(f"Got result: {result}")
            except Exception as e:
                print(f"Task raised an exception: {e}")