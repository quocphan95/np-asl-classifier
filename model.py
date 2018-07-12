from activation_functions import *


class NNModel:

    def __init__(self, dims, HPs):
        self._dims = dims
        self._keepprobs = HPs.keepprobs
        self._activate = HPs.activates
        self._learning_rate = HPs.learning_rate
        self._beta1 = HPs.beta1
        self._beta2 = HPs.beta2
        self._eps = HPs.eps
        self._numiter = HPs.numiter

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
    def fit(self, ws, bs, X, Y, callback=lambda iter, cost: cost):
        i = 0
        costs = []
        #vdws = []
        #vdbs = []
        #sdws = []
        #sdbs = []

        #for j in range(0, len(ws)):
            #vdws = vdws + [np.zeros(ws[j].shape)]
            #vdbs = vdbs + [np.zeros(bs[j].shape)]
            #sdws = sdws + [np.zeros(ws[j].shape)]
            #sdbs = sdbs + [np.zeros(bs[j].shape)]

        while i < self._numiter:
            (_, cache) = self.forward_probagation(ws, bs, X, Y)
            (dws, dbs) = self.backward_propagation(ws, X, Y, cache)

            for j in range(0, len(ws)):
                #vdws[j] = self._beta1 * vdws[j] + (1 - self._beta1) * dws[j]
                #vdbs[j] = self._beta1 * vdbs[j] + (1 - self._beta1) * dbs[j]
                #sdws[j] = beta2 * sdws[j] + (1 - beta2) * np.square(dws[j])
                #sdbs[j] = beta2 * sdbs[j] + (1 - beta2) * np.square(dbs[j])
                #ws[j] = ws[j] - self._learning_rate * dws[j]
                #bs[j] = bs[j] - self._learning_rate * dbs[j]
                ws[j] = ws[j] - self._learning_rate * dws[j]
                bs[j] = bs[j] - self._learning_rate * dbs[j]

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
