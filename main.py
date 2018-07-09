from init import *
from input import *
from model import *
from hyperparameters import HyperParameters as HPs

if __name__ == "__main__":

    # Read data
    (X, Y) = ImageInput.getbatch(0, HPs.m, HPs.classes, HPs.experclass)
    train_index = int(HPs.m * 0.8)
    X_train = X[:, 0:train_index]
    Y_train = Y[:, 0:train_index]
    X_test = X[:, train_index:-1]
    Y_test = Y[:, train_index:-1]
    print("Reading examples done!")

    # Init model
    """
    dims = (X.shape[0],) + HPs.dims
    model = NNModel(dims, HPs.keepprobs, HPs.activates)
    (ws, bs) = XavierInitialization.init(dims)
    print("Creating model done!")
    print(model.gd_checking(ws, bs, X, Y))
    """
    dims = (3,4,2)
    model = NNModel(dims, (1, 1), (tanh, softmax) )
    (ws, bs) = XavierInitialization.init(dims)
    X = np.random.randn(3, 5)
    Y = np.zeros((2, 5))
    for i in range(0, 5):
        choice = np.random.choice([0, 1])
        Y[choice][i] = 1

    print(model.gd_checking(ws, bs, X, Y))
    """
    # Train model
    print("Training model:")
    J_trains = []
    J_tests = []
    J_train = 0
    J_test = 0

    for i in range(0, HPs.numiter):
        J_train = model.fit(ws, bs, X_train, Y_train, 1, HPs.learning_rate)
        (J_test, _) = model.forward_probagation(ws, bs, X_test, Y_test)
        J_trains = J_trains + [J_train]
        J_tests = J_tests + [J_test]

        if i % 50 == 0:
            print("iteration", i, ", cost =", J_train)

    print("Finish training model, J_train = ", J_train)
    print("Finish training model, J_test = ", J_test)

    # draw chart
    plt.xlabel("Iterations")
    plt.ylabel("Cost function")
    iters = np.array(range(0, HPs.numiter)).reshape(-1, 1)
    plt.plot(iters, np.array(J_trains).reshape(-1, 1), "b-", iters, np.array(J_tests).reshape(-1, 1), "r-")
    plt.show()"""
