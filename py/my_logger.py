import logging
import numpy as np

# 初始化 logger
logger = logging.getLogger("np_logger")
logger.setLevel(logging.DEBUG)

# 控制台输出格式
console_handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
console_handler.setFormatter(formatter)

if not logger.hasHandlers():
    logger.addHandler(console_handler)

# 普通打印函数
def log(level, *args):
    msg = " ".join(str(a) for a in args)
    logger.log(level, msg)

# 格式化数组打印
def log_array(level, array, tag="", precision=2, align=True):
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    formatted = np.array2string(
        array,
        precision=precision,
        suppress_small=True,
        max_line_width=100,
        formatter={'float_kind': lambda x: f"{x:.{precision}f}" if align else str(x)}
    )
    message = f"\n{tag}\n{formatted}" if tag else f"\n{formatted}"
    logger.log(level, message)

# 快捷函数
def debug(*args): log(logging.DEBUG, *args)
def info(*args): log(logging.INFO, *args)
def warning(*args): log(logging.WARNING, *args)
def error(*args): log(logging.ERROR, *args)

def debug_array(array, tag="", precision=2, align=True): log_array(logging.DEBUG, array, tag, precision, align)
def info_array(array, tag="", precision=2, align=True): log_array(logging.INFO, array, tag, precision, align)
def warning_array(array, tag="", precision=2, align=True): log_array(logging.WARNING, array, tag, precision, align)
def error_array(array, tag="", precision=2, align=True): log_array(logging.ERROR, array, tag, precision, align)
