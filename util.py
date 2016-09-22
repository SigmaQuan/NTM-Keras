"""
contourf.
"""
import matplotlib.pyplot as plt
import numpy as np

from keras.callbacks import Callback         # Add by Steven Robot
from keras import backend as K


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.acces = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.acces.append(logs.get('acc'))

def show(w, w_title):
    """
    Show a weight matrix.
    :param w: the weight matrix.
    :param w_title: the title of the weight matrix
    :return: None.
    """
    # show w_z matrix of update gate.
    axes_w = plt.gca()
    plt.imshow(w)
    plt.colorbar()
    # plt.colorbar(orientation="horizontal")
    plt.xlabel("$w_{1}$")
    plt.ylabel("$w_{2}$")
    axes_w.set_xticks([])
    axes_w.set_yticks([])
    matrix_size = "$:\ %d \\times\ %d$" % (len(w[0]), len(w))
    w_title += matrix_size
    plt.title(w_title)

    # show the matrix.
    plt.show()

if __name__ == "__main__":
    w = np.random.random((8, 10))
    title = " "
    show(w, title)
