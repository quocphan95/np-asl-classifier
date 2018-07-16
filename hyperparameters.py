from activation_functions import *


class HyperParameters:
    experclass = 3000
    classes = (
        "A", "B", "C", "D", "E",
        "F", "G", "H", "I", "J",
        "K", "K", "M", "N", "O",
        "P", "Q", "R", "S", "T",
        "U", "V", "W", "X", "Y",
        "Z", "del", "nothing", "space")
    m = 2560
    batch_size = 64

    dims = (160, 160, 160, 160, 160, 160, 160, 160, 80, 80, 80, 80, 80, 80, 80, 80, len(classes))
    keepprobs = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    activates = (tanh, tanh, tanh, tanh, tanh, tanh, tanh, tanh, tanh, tanh, tanh, tanh, tanh, tanh, tanh, tanh,softmax)
    learning_rate = 0.0005
    beta1 = 0.9  # momentum beta
    beta2 = 0.9
    eps = 10e-8
    numiter = 200
    imagesize = 32, 32
