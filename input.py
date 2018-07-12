import os.path as path
import numpy as np
from preprocessor import *

class ImageInput():

    @staticmethod
    def get_training_set(set_size, classes, experclass):
        num_of_classes = len(classes)

        assert set_size <= experclass + num_of_classes

        X = np.array([])
        Y = np.array([]).reshape(len(classes), 0)
        ex_class = 0  # index of class name of the example to be read
        ex_index = 1  # index of that example in its class

        for i in range(set_size):
            imagepath = path.join("asl_alphabet_train", classes[ex_class], classes[ex_class] + str(ex_index) + ".jpg")
            img = cv2.imread(imagepath)
            x = Preprocessor.preprocess(img)

            if i == 0:
                X = X.reshape(x.shape[0], 0)

            y = np.zeros((len(classes), 1))
            y[ex_class][0] = 1
            X = np.concatenate((X, x), axis=1)
            Y = np.concatenate((Y, y), axis=1)
            ex_class = ex_class + 1
            ex_index = ex_index + ex_class // num_of_classes
            ex_class = ex_class % num_of_classes

        return X, Y
