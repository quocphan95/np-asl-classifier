from activation_functions import *


class NNModel:

    def __init__(self, dims, keepprobs, activates):
        self._dims = dims
        self._keepprobs = keepprobs
        self._activate = activates

    def forward_probagation(self, ws, bs, X, Y, test = False):
        Z = []
        A = [X]
        al = X  # A0

        for l in range(1, len(self._dims)):
            keepprob = 1.0 if test else self._keepprobs[l - 1]

            zl = np.dot(ws[l - 1], al) + bs[l - 1]
            al = self._activate[l - 1].calculate(zl)
            d = np.random.rand(al.shape[0], al.shape[1]) < keepprob
            al = al * d / self._keepprobs[l - 1]
            Z.append(zl)
            A.append(al)

        cache = {
            "Z": Z,
            "A": A
        }
        m = X.shape[1]
        cost = -1 / m * np.sum(Y * np.log(al))
        return cost, cache

    def backward_propagation(self, ws, X, Y, cache):
        m = X.shape[1]
        dz = cache["A"][-1] - Y  # dzl
        dws = []
        dbs = []

        for l in range(len(self._dims) - 1, 0, -1):
            dw = 1 / m * np.dot(dz, cache["A"][l - 1].T)
            db = 1 / m * np.sum(dz, axis=1, keepdims=True)

            if l > 1:
                dz = np.dot(ws[l - 1].T, dz) * self._activate[l - 2].derivative(cache["Z"][l - 2])

            dws = [dw] + dws
            dbs = [db] + dbs

        return dws, dbs

    # optimize the model
    def fit(self, ws, bs, X, Y, numiter=1, learning_rate=0.001, callback=lambda iter, cost: cost):
        i = 0
        costs = []

        while i < numiter:
            (_, cache) = self.forward_probagation(ws, bs, X, Y)
            (dW, db) = self.backward_propagation(ws, X, Y, cache)

            for j in range(0, len(ws)):
                ws[j] = ws[j] - learning_rate * dW[j]
                bs[j] = bs[j] - learning_rate * db[j]

            (cost, _) = self.forward_probagation(ws, bs, X, Y)
            costs = costs + [callback(i, cost)]
            i = i + 1

        return costs

    def gd_checking(self, ws, bs, X, Y):

        def params2vector(ws, bs):
            vector = np.array([]).reshape(0, 1)

            for w in ws:
                vector = np.concatenate((vector, w.reshape(-1, 1)), axis=0)

            for b in bs:
                vector = np.concatenate((vector, b.reshape(-1, 1)), axis=0)

            return vector

        def vector2params(vector):
            begin = 0
            ws = []
            bs = []

            for l in range(1, len(self._dims)):
                length = self._dims[l - 1] * self._dims[l]
                w = vector[begin:(begin + length), 0].reshape(self._dims[l], self._dims[l - 1])
                ws = ws + [w]
                begin = begin + length

            for l in range(1, len(self._dims)):
                length = self._dims[l]
                b = vector[begin:(begin + length), 0].reshape(-1, 1)
                bs = bs + [b]
                begin = begin + length

            return ws, bs

        def gradient2vector(dws, dbs):
            vector = np.array([]).reshape(0, 1)

            for dw in dws:
                vector = np.concatenate((vector, dw.reshape(-1, 1)), axis=0)

            for db in dbs:
                vector = np.concatenate((vector, db.reshape(-1, 1)), axis=0)

            return vector

        (_, cache) = self.forward_probagation(ws, bs, X, Y)
        (dws, dbs) = self.backward_propagation(ws, X, Y, cache)
        epsilon = 1e-7
        paramvector = params2vector(ws, bs)
        grads = gradient2vector(dws, dbs)
        gradapproxs = np.zeros((len(paramvector), 1))

        for i in range(0, len(paramvector)):
            paramvectorplus = np.copy(paramvector)
            paramvectorminus = np.copy(paramvector)
            paramvectorplus[i][0] = paramvectorplus[i][0] + epsilon
            paramvectorminus[i][0] = paramvectorminus[i][0] - epsilon
            (wsplus, bsplus) = vector2params(paramvectorplus)
            (wsminus, bsminus) = vector2params(paramvectorminus)
            (J_plus, _) = self.forward_probagation(wsplus, bsplus, X, Y)
            (J_minus, _) = self.forward_probagation(wsminus, bsminus, X, Y)
            gradapproxs[i][0] = (J_plus - J_minus) / (2 * epsilon)

        nominator = np.linalg.norm(grads - gradapproxs)
        denominator = np.linalg.norm(grads) + np.linalg.norm(gradapproxs)

        return nominator / denominator
