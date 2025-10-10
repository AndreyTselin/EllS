import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


N = 4.32+0.073j
y = np.sqrt(1 - (1 * np.sin(np.deg2rad(50)) / N) ** 2)

#print(y)

x = 2 * np.pi * (100 / 500) * N * y

#print(x)

#print(np.identity(2) @ np.array([[1, 2], [3, 4]]))

z = np.array([[1], [-1]])
print(z[1, 0])