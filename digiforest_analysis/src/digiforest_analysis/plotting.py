def adjust_lightness(color, amount=0.5):
    """
    From https://stackoverflow.com/a/49601444
    """
    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color]
    except Exception:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def darken(color, amount=0.0):
    """Darkens a color. 0.0 means no change"""
    import numpy as np

    return adjust_lightness(color, 1.0 - np.clip(amount, 0.0, 1.0))


def lighten(color, amount=0.0):
    """Darkens a color. 0.0 means no change"""
    import numpy as np

    return adjust_lightness(color, 1.0 + np.clip(amount, 0.0, 1.0))
