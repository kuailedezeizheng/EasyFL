import datetime


def add_timestamp() -> str:
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y%m%d%H%M%S")  # 格式化时间
    return timestamp
