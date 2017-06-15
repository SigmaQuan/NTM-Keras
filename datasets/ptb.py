# -*- coding: utf-8 -*-
import numpy as np
import os


def load_data(path='simple-examples.tgz'):
    """Loads the Penn Treebank (PTB) dataset.

    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """