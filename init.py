import numpy as np
import pickle as pk
import os.path as path


class HeInitialization:

    @staticmethod
    def init(dims):
        ws = []
        bs = []

        for i in range(1, len(dims)):
            w = np.random.randn(dims[i], dims[i - 1]) * np.sqrt(2 / dims[i - 1])
            b = np.zeros((dims[i], 1))
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
            b = np.zeros((dims[i], 1))
            ws = ws + [w]
            bs = bs + [b]

        return ws, bs


# Read the matrixes from file and set them as initial parameter
class PresetInitialization:

    @staticmethod
    def init(pkl_name):
        pkl_name = path.join("models", pkl_name)
        f = open(pkl_name, "rb")
        return pk.load(f)

