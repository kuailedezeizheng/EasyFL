import csv
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
        with open('./result/time/average_execution_time.csv', 'w', newline='') as csvfile:
            fieldnames = ['Function', 'Average Time', 'Total Calls']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for func_name, times in times_cache.items():
                avg_time = sum(times) / len(times)
                total_calls = len(times)
                writer.writerow({'Function': func_name, 'Average Time': avg_time, 'Total Calls': total_calls})
                print(f"函数 '{func_name}' 平均执行时间为 {avg_time:.4f} 秒（总调用次数：{total_calls}）。")

    wrapper.average_time = average_time
    return wrapper
