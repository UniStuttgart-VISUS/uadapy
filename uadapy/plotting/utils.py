import numpy as np
import matplotlib.pyplot as plt

def generate_random_colors(length):
    return ["#"+''.join([np.random.choice('0123456789ABCDEF') for j in range(6)]) for _ in range(length)]

def generate_spectrum_colors(length):
    cmap = plt.cm.get_cmap('viridis', length)  # You can choose different colormaps like 'jet', 'hsv', 'rainbow', etc.
    return np.array([cmap(i) for i in range(length)])