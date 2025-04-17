# logger.py

import sys
import time

class Logger:
    def __init__(self, enabled=True, level='info', logfile=None):
        self.enabled = enabled
        self.level_order = {'debug': 0, 'info': 1, 'warning': 2, 'error': 3}
        self.level = level
        self.logfile = logfile
        self.out = open(logfile, 'a') if logfile else sys.stdout

    def log(self, msg, level='info'):
        if not self.enabled or self.level_order[level] < self.level_order[self.level]:
            return
        timestamp = time.strftime("%H:%M:%S")
        output = f"[{timestamp}] [{level.upper()}] {msg}"
        print(output, file=self.out)
        if self.logfile:
            self.out.flush()

    def debug(self, msg): self.log(msg, level='debug')
    def info(self, msg): self.log(msg, level='info')
    def warning(self, msg): self.log(msg, level='warning')
    def error(self, msg): self.log(msg, level='error')

    def close(self):
        if self.logfile and self.out and self.out != sys.stdout:
            self.out.close()


