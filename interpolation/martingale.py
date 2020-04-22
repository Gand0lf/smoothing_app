import numpy as np

class martingale(object):
    """yield curve naive backtest E(X_t|F_t-1)=X_t-1 martingale
    index/row corresponds time, columns to pillars"""

    def __init__(self, data, critical=2, start: int = 0):
        self.data = data
        self.critical = critical
        self.start = start

    @property
    def diffs(self):
        return self.data.diff()

    @property
    def subset(self):
        return self.diffs[self.start:]

    @property
    def stdprederror(self):
        return self.subset.std(axis=0)

    @property
    def inlimit(self):
        critical = self.critical
        shiftbase = self.data.shift()
        StdErrArray = np.broadcast_to(np.array(self.stdprederror), (self.data.shape[0], self.data.shape[1]))
        shiftbase_up = shiftbase + critical * StdErrArray
        self._shiftbase_up = shiftbase_up
        shiftbase_down = shiftbase - critical * StdErrArray
        self._shiftbase_down = shiftbase_down
        return (shiftbase_down < self.data) & (self.data < shiftbase_up)