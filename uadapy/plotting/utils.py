import numpy as np
import matplotlib.pyplot as plt
import colorsys
from PIL import ImageColor

def generate_random_colors(length):
    return ["#"+''.join([np.random.choice('0123456789ABCDEF') for j in range(6)]) for _ in range(length)]

def generate_spectrum_colors(length, which='viridis'):
    cmap = plt.cm.get_cmap(which, length)  # You can choose different colormaps like 'jet', 'hsv', 'rainbow', etc.
    return np.array([cmap(i) for i in range(length)])

# color_name such as 'red' or 'magenta' to (1.0, 0.0, 0.0) or (1.0. 0.0. 1.0)
def rgb_from_named_color(color_name: str) -> tuple:
    color = ImageColor.colormap[color_name]
    return rgb_from_hex_string(color)


# hex_string of the form '#------' e.g '#ff00ff' to (1.0, 0.0, 1.0)
def rgb_from_hex_string(hex_string: str) -> tuple:
    return tuple(int(hex_string[i:i + 2], 16) / 255.0 for i in (1, 3, 5))


# rgb in the form of (1.0, 0.0, 0.0) to '#ff0000'
def hex_string_from_rgb(rgb: tuple) -> str:
    return '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))


def any_color_to_rgb(color) -> tuple:
    if isinstance(color, tuple) or isinstance(color, list):
        return color
    if isinstance(color, str) and color.startswith('#'):
        return rgb_from_hex_string(color)
    if isinstance(color, str):
        return rgb_from_named_color(color)
    return rgb_from_hex_string('#ff00ff')


def scale_brightness(rgb: tuple, s: float) -> tuple:
    hsv = colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])
    return colorsys.hsv_to_rgb(hsv[0], hsv[1], hsv[2]*s)


def scale_saturation(rgb: tuple, s: float) -> tuple:
    hsv = colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])
    return colorsys.hsv_to_rgb(hsv[0], hsv[1]*s, hsv[2])
