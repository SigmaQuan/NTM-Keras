"""
Learning word-level language modeling on CBT data set with NTM.
"""

import os


FOLDER = "experiment_results/cbt/"
if not os.path.isdir(FOLDER):
    os.makedirs(FOLDER)
    print("create folder: %s" % FOLDER)
