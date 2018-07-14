import sys
import os.path as path
import pickle as pk
import numpy as np
from input import *
from model import *
from hyperparameters import HyperParameters as HPs


if __name__ == "__main__":
    model_path = path.join("models", sys.argv[1])
    example_path = path.join("demo_inputs", sys.argv[2])
    f = open(model_path, "rb")
    (ws, bs) = pk.load(f)
    x, y = ImageInput.get_training_example(example_path, HPs.classes, sys.argv[2][0])
    dims = (x.shape[0],) + HPs.dims
    model = NNModel(dims, HPs)
    (_, cache) = model.forward_probagation(ws, bs, x, y, test=True)
    al = cache["A"][-1]
    maxi = np.argmax(al, axis=0)
    print(HPs.classes[maxi[0]])

