import numpy as np


class HeInitialization:

    @staticmethod
    def init(dims):
        ws = []
        bs = []

        for i in range(1, len(dims)):
            w = np.random.randn(dims[i], dims[i - 1]) * np.sqrt(2 / dims[i - 1])
            b = np.zeros((dims[i], 1)) + 1
            ws = ws + [w]
            bs = bs + [b]

        return ws, bs


class XavierInitialization:

    @staticmethod
    def init(dims):
        ws = []
        bs = []

        for i in range(1, len(dims)):
            w = np.random.randn(dims[i], dims[i - 1]) * np.sqrt(1 / dims[i - 1])
            b = np.zeros((dims[i], 1)) + 1
            ws = ws + [w]
            bs = bs + [b]

        return ws, bs
