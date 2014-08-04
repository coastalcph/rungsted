import time

class Timer(object):
    def begin(self):
        self._begin = time.time()

    def end(self):
        self._end = time.time()

    def elapsed(self):
        return self._end - self._begin

