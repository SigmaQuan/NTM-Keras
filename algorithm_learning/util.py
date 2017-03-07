"""
Add class LossHistory for recoding the history information of loss and
accuracy during the training processing.
"""
# import matplotlib.pyplot as plt
# import numpy as np

from keras.callbacks import Callback         # Add by Steven Robot
# from keras import backend as K


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.acces = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.acces.append(logs.get('acc'))
