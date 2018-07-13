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
        self._numiter = HPs.numiter  # number of epoch
        self._batchsize = HPs.batch_size

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

    # optimize the model by using mini-batch GD
    def fit_mini_batch(self, ws, bs, X, Y, callback=lambda epoch, i, cost: cost):
        costs = []
        vdws = []
        vdbs = []
        sdws = []
        sdbs = []

        for j in range(len(ws)):
            vdws = vdws + [np.zeros(ws[j].shape)]
            vdbs = vdbs + [np.zeros(bs[j].shape)]
            sdws = sdws + [np.zeros(ws[j].shape)]
            sdbs = sdbs + [np.zeros(bs[j].shape)]
        iter_per_epoch = int(X.shape[1] / self._batchsize)

        for epoch in range(self._numiter):

            for iter in range(iter_per_epoch):
                begin_index = iter * self._batchsize
                X_mini = X[:, begin_index: (begin_index + self._batchsize)].reshape(-1, self._batchsize)
                Y_mini = Y[:, begin_index: (begin_index + self._batchsize)].reshape(-1, self._batchsize)
                (_, cache) = self.forward_probagation(ws, bs, X_mini, Y_mini)
                (dws, dbs) = self.backward_propagation(ws, X_mini, Y_mini, cache)

                for i in range(len(ws)):
                    vdws[i] = self._beta1 * vdws[i] + (1 - self._beta1) * dws[i]
                    vdbs[i] = self._beta1 * vdbs[i] + (1 - self._beta1) * dbs[i]
                    sdws[i] = self._beta2 * sdws[i] + (1 - self._beta2) * np.square(dws[i])
                    sdbs[i] = self._beta2 * sdbs[i] + (1 - self._beta2) * np.square(dbs[i])
                    #ws[i] = ws[i] - self._learning_rate * vdws[i]
                    #bs[i] = bs[i] - self._learning_rate * vdbs[i]
                    #ws[i] = ws[i] - self._learning_rate * dws[i] / (np.sqrt(sdws[i]) + self._eps)
                    #bs[i] = bs[i] - self._learning_rate * dbs[i] / (np.sqrt(sdbs[i]) + self._eps)
                    ws[i] = ws[i] - self._learning_rate * dws[i]
                    bs[i] = bs[i] - self._learning_rate * dbs[i]

                (cost, _) = self.forward_probagation(ws, bs, X, Y)
                costs = costs + [callback(epoch, iter, cost)]

        return costs

    # optimize the model
    def fit(self, ws, bs, X, Y, callback=lambda epoch, cost: cost):
        costs = []

        for epoch in range(self._numiter):
            (_, cache) = self.forward_probagation(ws, bs, X, Y)
            (dws, dbs) = self.backward_propagation(ws, X, Y, cache)

            for i in range(len(ws)):
                ws[i] = ws[i] - self._learning_rate * dws[i]
                bs[i] = bs[i] - self._learning_rate * dbs[i]

            (cost, _) = self.forward_probagation(ws, bs, X, Y)
            costs = costs + [callback(epoch, cost)]

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
