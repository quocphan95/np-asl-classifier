import numpy as np


class tanh:
    @staticmethod
    def calculate(Z):
        return np.tanh(Z)

    @staticmethod
    def derivative(Z):
        t = tanh.calculate(Z)
        return 1 - np.multiply(t, t)  # 1 - (tanh(Z))^2


class sigmoid:
    @staticmethod
    def calculate(Z):
        return 1 / (1 + np.exp(Z))

    @staticmethod
    def derivative(Z):
        t = sigmoid.calculate(Z)
        return t * (1 - t)


class softmax:
    @staticmethod
    def calculate(Z):
        t = np.exp(Z)
        return t / t.sum(axis=0, keepdims=True)

class stable_softmax:
    @staticmethod
    def calculate(Z):
        maxz = np.max(Z, axis=0, keepdims=True)
        t = np.exp(Z - maxz)
        return t / t.sum(axis=0, keepdims=True)


class linear:
    @staticmethod
    def calculate(Z):
        return Z

    @staticmethod
    def derivative(Z):
        return 1

class relu:
    @staticmethod
    def calculate(Z):
        return np.maximum(Z, 0)

    @staticmethod
    def derivative(Z):
        return (Z > 0).astype(int)
