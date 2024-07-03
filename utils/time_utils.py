import time
from datetime import datetime

def get_cur_time_string():
    return timestamp2string(time.time())

def timestamp2string(timestmap):
    struct_time = time.localtime(timestmap)
    str = time.strftime("%Y-%m-%d %H:%M:%S", struct_time)
    return str

def timestamp2string_less(timestmap):
    struct_time = time.localtime(timestmap)
    str = time.strftime("%M:%S", struct_time)
    return str

def timestamp2string_more(timestamp):
    # 将浮点型时间戳转换为 datetime 对象
    dt = datetime.fromtimestamp(timestamp)
    
    # 格式化 datetime 对象为字符串，包括毫秒
    # %f 代表微秒，取前3位得到毫秒
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def string2timstamp(str):
    struct_time = time.strptime(str, "%Y-%m-%d %H:%M:%S")
    timestamp = time.mktime(struct_time)
    return timestamp


EPS = 0.01
def less(timestamp1, timestamp2):
    if timestamp2 - timestamp1 > EPS:
        return True
    else:
        return False
    

def large(timestamp1, timestamp2):
    if timestamp1 - timestamp2 > EPS:
        return True
    else:
        return False

def equal(timestamp1, timestamp2):
    if abs(timestamp1 - timestamp2) <= EPS:
        return True
    else:
        return False

def large_equal(timestamp1, timestamp2):
    if large(timestamp1, timestamp2) or equal(timestamp1, timestamp2):
        return True
    else:
        return False
    


if __name__ == "__main__":
    # timestamp = 1697166349.94
    # print(timestamp2string(timestamp))
    print(string2timstamp("2023-8-21 17:52:00"))
