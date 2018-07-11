import numpy as np

class Checker:
    @staticmethod
    def check(ws, bs, X, Y, model):
        (_, cache) = model.forward_probagation(ws, bs, X, Y, test=True)
        m = X.shape[1]
        true = 0
        maxi = np.argmax(cache["A"][-1], axis=0)

        for i in range(m):
            if Y[maxi[i], i] == 1:
                true += 1

        return true / m * 100
