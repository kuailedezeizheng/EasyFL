import time


def record_time(func):
    times_cache = {}

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        times_cache.setdefault(func.__name__, []).append(execution_time)
        print(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds.")
        return result

    def average_time():
        for func_name, times in times_cache.items():
            avg_time = sum(times) / len(times)
            print(f"Average execution time for '{func_name}': {avg_time:.4f} seconds (total calls: {len(times)}).")

    wrapper.average_time = average_time
    return wrapper
