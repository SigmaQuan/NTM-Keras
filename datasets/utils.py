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


def generate_weightings(row, col):
    write_weightings = np.zeros((row, col), dtype=np.float32)
    read_weightings = np.zeros((row, col), dtype=np.float32)
    r = (row * 3) / 4
    for i in np.arange(0, col/2):
        write_weightings[r][i] = 1
        read_weightings[r][i + col/2] = 1
        r -= 1

    return write_weightings, read_weightings
