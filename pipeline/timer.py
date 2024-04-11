import time

class Timer:
    
    def __init__(self, momentum=0.9):
        self.time = time.time()
        self.running_delta = None
        self.momentum = momentum

    def tick(self):
        now = time.time()
        delta = now - self.time
        self.time = now
        if self.running_delta is None:
            self.running_delta = delta
        else:
            self.running_delta *= self.momentum
            self.running_delta += delta * (1 - self.momentum)

    def fps(self):
        if self.running_delta is None:
            return 0
        if self.running_delta != 0:
            return 1 / self.running_delta
        return 0