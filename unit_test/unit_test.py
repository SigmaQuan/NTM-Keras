import mahotas as mh
from matplotlib import pyplot as plt
import numpy as np


iteration = 2
image = mh.imread("../experiment/copy_data_predict_%3d.png"%iteration)
plt.imshow(image)
plt.show()

image = mh.colors.rgb2gray(image, dtype=np.uint8)
plt.imshow(image)