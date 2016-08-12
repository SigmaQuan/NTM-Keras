# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import datetime

# my fake data
dates = np.array([datetime.datetime(2000,1,1) + datetime.timedelta(days=i) for i in range(365*5)])
data = np.sin(np.arange(365*5)/365.0*2*np.pi - 0.25*np.pi) + np.random.rand(365*5) /3

# creates fig with 2 subplots
fig = plt.figure(figsize=(10.0, 6.0))
ax = plt.subplot2grid((2,1), (0, 0))
ax2 = plt.subplot2grid((2,1), (1, 0))
## plot dates
ax2.plot_date( dates, data )

# rotates labels
plt.setp( ax2.xaxis.get_majorticklabels(), rotation=-45 )

# shift labels to the right
for tick in ax2.xaxis.get_majorticklabels():
    tick.set_horizontalalignment("right")

plt.tight_layout()
plt.show()
