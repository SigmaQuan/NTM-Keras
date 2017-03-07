"""
Learning question answering on bAbI data set with NTM.
"""

import os


FOLDER = "experiment_results/babi/"
if not os.path.isdir(FOLDER):
    os.makedirs(FOLDER)
    print("create folder: %s" % FOLDER)

