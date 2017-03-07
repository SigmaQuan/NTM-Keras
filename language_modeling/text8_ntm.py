"""
Learning word-level language modeling on text8 data set with NTM.
"""

import os


FOLDER = "experiment_results/text8/"
if not os.path.isdir(FOLDER):
    os.makedirs(FOLDER)
    print("create folder: %s" % FOLDER)

