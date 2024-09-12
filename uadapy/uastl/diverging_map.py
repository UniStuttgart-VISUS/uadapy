from colorspacious import cspace_convert
import numpy as np

def rgb_to_lab(rgb):
    """
    Convert RGB to CIE-L*a*b* color space.
    """
    # Convert RGB to Lab using colorspacious
    lab = cspace_convert(rgb, "sRGB1", "CIELab")
    return lab

def lab_to_msh(lab):
    """
    Convert CIE-L*a*b* to Msh color space.
    """
    L, a, b = lab
    M = np.sqrt(L**2 + a**2 + b**2)
    s = np.arccos(L / M) if M != 0 else 0
    h = np.arctan2(b, a)
    return np.array([M, s, h])

def angle_diff(angle1, angle2):
    """
    Calculate the angular difference.
    """
    diff = np.abs(angle1 - angle2)
    if diff > np.pi:
        diff = 2 * np.pi - diff
    return diff

def adjust_hue(msh, m_value):
    """
    Adjust the hue value.
    """
    if msh[1] > 0.05:
        hue = msh[2]
    else:
        hue = 0
    return hue

def msh_to_lab(msh):
    """
    Convert Msh color space to CIE-L*a*b*.
    """
    M, s, h = msh
    L = M * np.cos(s)
    a = M * np.sin(s) * np.cos(h)
    b = M * np.sin(s) * np.sin(h)
    return np.array([L, a, b])

def lab_to_rgb(lab):
    """
    Convert CIE-L*a*b* to RGB color space.
    """
    # Convert Lab to RGB using colorspacious
    rgb = cspace_convert(lab, "CIELab", "sRGB1")
    # Ensure RGB values are within [0, 1]
    rgb = np.clip(rgb, 0, 1)
    return rgb

def diverging_map_1val(s, rgb1, rgb2):
    """Interpolate a diverging color map."""
    lab1 = rgb_to_lab(rgb1)
    lab2 = rgb_to_lab(rgb2)
    msh1 = lab_to_msh(lab1)
    msh2 = lab_to_msh(lab2)
    if msh1[1] > 0.05 and msh2[1] > 0.05 and angle_diff(msh1[2], msh2[2]) > 0.33 * np.pi:
        Mmid = max(msh1[0], msh2[0])
        Mmid = max(88.0, Mmid)
        if s < 0.5:
            msh2[0] = Mmid
            msh2[1] = 0.0
            msh2[2] = 0.0
            s *= 2.0
        else:
            msh1[0] = Mmid
            msh1[1] = 0.0
            msh1[2] = 0.0
            s = 2.0 * s - 1.0

    if msh1[1] < 0.05 and msh2[1] > 0.05:
        msh1[2] = adjust_hue(msh2, msh1[0])
    elif msh2[1] < 0.05 and msh1[1] > 0.05:
        msh2[2] = adjust_hue(msh1, msh2[0])

    msh_tmp = [
        (1 - s) * msh1[0] + s * msh2[0],
        (1 - s) * msh1[1] + s * msh2[1],
        (1 - s) * msh1[2] + s * msh2[2],
    ]
    lab_tmp = msh_to_lab(msh_tmp)
    rgb_tmp = lab_to_rgb(lab_tmp)
    return rgb_tmp

def diverging_map(s, rgb1, rgb2):
    """Create a diverging colormap."""
    map = np.zeros((len(s), 3))
    for i, val in enumerate(s):
        map[i, :] = diverging_map_1val(val, rgb1, rgb2)
    return map
