import numpy as np
import matplotlib.pyplot as plt

def generate_random_colors(n):
    return ["#" +''.join([np.random.choice('0123456789ABCDEF') for j in range(6)]) for _ in range(n)]

def generate_spectrum_colors(n):
    cmap = plt.cm.get_cmap('viridis', n)  # You can choose different colormaps like 'jet', 'hsv', 'rainbow', etc.
    return np.array([cmap(i) for i in range(n)])