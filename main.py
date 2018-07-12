from init import *
from input import *
from model import *
from hyperparameters import HyperParameters as HPs
from checker import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    np.random.seed(1)
    # Read data
    (X, Y) = ImageInput.get_training_set(HPs.m, HPs.classes, HPs.experclass)
    train_index = int(HPs.m * 0.8)
    X_train = X[:, 0:train_index]
    Y_train = Y[:, 0:train_index]
    X_test = X[:, train_index:-1]
    Y_test = Y[:, train_index:-1]
    print("Reading examples done!")

    # Create model
    dims = (X.shape[0],) + HPs.dims
    model = NNModel(dims, HPs)
    (ws, bs) = HeInitialization.init(dims)
    print("Creating model done!")

    # Train model
    print("Training model:")

    def train_callback(iter, cost):
        if iter % 10 == 0:
            print("iteration", ("{0:>6}").format(str(iter)), ", cost =", cost)
        return cost

    J_trains = model.fit(ws, bs, X_train, Y_train, train_callback)
    (J_test, _) = model.forward_probagation(ws, bs, X_test, Y_test, True)

    print("Finish training model, J_train = ", J_trains[-1])
    print("Train: ", Checker.check(ws, bs, X_train, Y_train, model))
    print("Test : ", Checker.check(ws, bs, X_test, Y_test, model))

    # draw chart
    plt.xlabel("Iterations")
    plt.ylabel("Cost function")
    iters = np.array(range(HPs.numiter)).reshape(-1, 1)
    plt.plot(iters, np.array(J_trains).reshape(-1, 1), "b-")
    plt.show()