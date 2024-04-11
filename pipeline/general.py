import numpy as np

class Pose:
    def __init__(self, R: np.ndarray, t: np.ndarray):
        assert R.shape == (3, 3), "R must has shape [3, 3]"
        assert t.shape == (3, 1) or t.shape == 3, "t must has shape [3, 1] or [3]"

        self.R = R
        self.t = t.reshape((3, 1))

    def transform(self, x: np.ndarray):
        assert x.shape[0] == 3, "input must has shape [3, ...]"
        return self.R @ x + self.t
