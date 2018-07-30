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
    # (ws, bs) = HeInitialization.init(dims)
    (ws, bs) = PresetInitialization.init("2018_7_16_13_33_39.pkl")
    print("Creating model done!")

    # Train model
    print("Training model:")

    def train_mini_callback(epoch, iter, cost):
        print("epoch {epoch:>6}, iteration {iter:>6}, cost = {cost:>20.10}".format(**locals()))
        return None

    def train_callback(epoch, cost):
        if epoch % 10 == 0:
            print("epoch {epoch:>6}, cost = {cost:>20.10}".format(**locals()))
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
    name = "{0.year}_{0.month}_{0.day}_{0.hour}_{0.minute}_{0.second}.pkl".format(now)

    f = open(path.join("models", name), "wb")
    pk.dump((ws, bs), f)