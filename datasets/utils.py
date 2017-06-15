# -*- coding: utf-8 -*-
import numpy as np
import random
import time


def initialize_random_seed():
    np.random.seed(time.time())
    random.seed(time.time())


def generate_random_binomial_(row, col):
    return np.random.binomial(
        1, 0.5, (row, col)).astype(np.uint8)
