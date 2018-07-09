from activation_functions import *


class HyperParameters:
    experclass = 3000
    classes = (
        "A", "B", "C", "D", "E",
        "F", "G", "H", "I", "J",
        "K", "K", "M", "N", "O",
        "P", "Q", "R", "S", "T",
        "U", "V", "W", "X", "Y",
        "Z", "del", "nothing", "space"
    )
    m = 64
    dims = (50, 50, len(classes))
    keepprobs = (1.0, 1.0, 1.0)
    activates = (tanh, tanh, softmax)
    learning_rate = 0.1
    numiter = 100
