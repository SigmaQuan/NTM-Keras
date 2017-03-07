"""
Learning word-level language modeling on PTB data set with NTM.
"""

import os


FOLDER = "experiment_results/ptb/"
if not os.path.isdir(FOLDER):
    os.makedirs(FOLDER)
    print("create folder: %s" % FOLDER)
