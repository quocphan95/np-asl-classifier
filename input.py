import os.path as path
import numpy as np
import matplotlib.pyplot as plt


class ImageInput():

    @staticmethod
    def getbatch(batch_number, batch_size, classes, experclass):
        num_of_classes = len(classes)

        assert (batch_number + 1) * batch_size <= experclass + num_of_classes
        X = np.array([])
        Y = np.array([]).reshape(len(classes), 0)
        row = batch_number * batch_size // num_of_classes + 1
        col = batch_number * batch_size % num_of_classes

        for i in range(0, batch_size):
            row = row + col // num_of_classes
            col = col % num_of_classes
            imagepath = path.join("asl_alphabet_train", classes[col], classes[col] + str(row) + ".jpg")
            x = plt.imread(imagepath)
            x = x.reshape(x.shape[0] * x.shape[1] * x.shape[2], 1)

            if i == 0:
                X = X.reshape(x.shape[0], 0)
            y = np.zeros((len(classes), 1))
            y[col][0] = 1
            X = np.concatenate((X, x), axis=1)
            Y = np.concatenate((Y, y), axis=1)
            col = col + 1

        return X, Y
