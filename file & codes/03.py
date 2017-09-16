import numpy as np
import matplotlib.pyplot as plt

grey_hieght = 28+ 4 * np.random.randn(500)
lab_hiegt = 24+ 4* np.random.randn(500)

plt.hist([grey_hieght,lab_hiegt],stacked = True,color = ['r','b'])
plt.show()

