import cv2
from hyperparameters import HyperParameters as HPs

class Preprocessor:
    @staticmethod
    def preprocess(img):
        img = cv2.resize(img, dsize=HPs.imagesize, interpolation=cv2.INTER_CUBIC)
        return img.reshape(-1, 1) / 255