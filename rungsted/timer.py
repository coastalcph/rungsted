import time


class Timer(object):
    def begin(self):
        self.begin = time.time()

    def end(self):
        self.end = time.time()

    def elapsed(self):
        return self.end - self.begin

