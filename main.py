import pickle as pk
from init import *
from input import *
from model import *
from hyperparameters import HyperParameters as HPs
from checker import *
import matplotlib.pyplot as plt
import os.path as path
import datetime

if __name__ == "__main__":
    np.random.seed(1)
    # Read data
    print("Reading examples")
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

    def train_mini_callback(epoch, iter, cost):
        print("epoch", ("{0:>6}").format(str(epoch)), ", iteration ", ("{0:>6}").format(str(iter)), ", cost =", cost)
        return cost

    def train_callback(epoch, cost):
        if epoch % 10 == 0:
            print("epoch", ("{0:>6}").format(str(epoch)), ", cost =", cost)
        return cost

    J_trains = model.fit_mini_batch(ws, bs, X_train, Y_train, train_mini_callback)
    (J_test, _) = model.forward_probagation(ws, bs, X_test, Y_test, True)

    print("Finish training model, J_train = ", J_trains[-1])
    print("Finish training model, J_test  = ", J_test)
    print("Train: ", Checker.check(ws, bs, X_train, Y_train, model))
    print("Test : ", Checker.check(ws, bs, X_test, Y_test, model))

    # draw chart
    plt.xlabel("Iterations")
    plt.ylabel("Cost function")
    plt.plot(np.arange(0, len(J_trains), 1), np.array(J_trains).reshape(-1, 1), "b-")
    plt.show()

    now = datetime.datetime.now()
    name = "{0}_{1}_{2}_{3}_{4}_{5}.pkl".format(now.year, now.month, now.day, now.hour, now.minute, now.second)

    f = open(path.join("models", name), "wb")
    pk.dump((ws, bs), f)