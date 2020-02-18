import numpy as np
import matplotlib.pyplot as plt



a = np.array([[76.62,76.99,77.18],[75.35, 75.96, 75.81],[74.83, 74.87, 74.67],[74.58, 74.40, 74.53]])
a = a.mean(1)
plt.plot(a, '-o')
plt.show()