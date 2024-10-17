import numpy as np
import matplotlib.pyplot as plt
import glasbey as gb
from matplotlib.colors import ListedColormap

def generate_random_colors(n):
    return ["#" +''.join([np.random.choice('0123456789ABCDEF') for j in range(6)]) for _ in range(n)]

def generate_spectrum_colors(n):
    cmap = plt.cm.get_cmap('viridis', n)  # You can choose different colormaps like 'jet', 'hsv', 'rainbow', etc.
    return np.array([cmap(i) for i in range(n)])

def get_colors(n):
    # Set2 colormap from matplotlib
    set2_cmap = plt.get_cmap('Set2')
    set2_colors = set2_cmap.colors

    # List to hold the final colors
    final_colors = []

    # First use the Set2 colormap
    for i in range(min(n, len(set2_colors))):
        final_colors.append(set2_colors[i])

    # If more colors are needed, extend using Glasbey's generated palette
    if n > len(set2_colors):
        # Generate the additional colors using glasbey
        glasbey_colors = gb.create_palette(palette_size=n - len(set2_colors))

        # Extend final colors with glasbey's colors
        final_colors.extend(glasbey_colors)

    return final_colors

def create_shaded_set2_colormap(alpha_values):
    """
    Create a custom colormap by varying alpha values for each Set2 color.

    Parameters:
    - alpha_values: list or array of alpha values to apply to each Set2 color (e.g., [0.3, 0.6, 1]).

    Returns:
    - A custom colormap with 8 Set2 colors, each with 3 alpha variations (total of 24 colors).
    """
    # Get the 8 Set2 colors from matplotlib
    set2_colors = plt.get_cmap('Set2').colors

    # Initialize a list to hold the new colors with alpha variations
    shaded_colors = []

    # Loop over each Set2 color and create alpha variations
    for color in set2_colors:
        for alpha in alpha_values:
            rgba_color = list(color) + [alpha]  # Add the alpha value to the RGB color
            shaded_colors.append(rgba_color)    # Append the RGBA color to the list

    # Create a ListedColormap with the shaded colors
    custom_cmap = ListedColormap(shaded_colors)

    return custom_cmap
