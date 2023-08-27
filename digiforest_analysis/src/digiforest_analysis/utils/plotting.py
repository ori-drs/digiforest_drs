import seaborn as sns
import numpy as np

from matplotlib.colors import ListedColormap


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


n_colors = 10
color_palette = sns.color_palette("colorblind", n_colors=n_colors, as_cmap=False)
color_palette_str = [
    "blue",
    "orange",
    "green",
    "red",
    "pink",
    "brown",
    "light_pink",
    "gray",
    "yellow",
    "light_blue",
]
color_palette_str = {k: v for k, v in zip(color_palette_str, color_palette)}
mpl_colorblind_cmap = ListedColormap(color_palette)

# Okabe-Ito colormap: https://clauswilke.com/dataviz/color-pitfalls.html
okabeito_palette_str = {
    "orange": np.array([230, 159, 0]) / 255,
    "light_blue": np.array([86, 180, 233]) / 255,
    "green": np.array([0, 158, 115]) / 255,
    "yellow": np.array([240, 228, 66]) / 255,
    "blue": np.array([0, 114, 178]) / 255,
    "red": np.array([213, 94, 0]) / 255,
    "pink": np.array([204, 121, 167]) / 255,
    "black": np.array([0, 0, 0]) / 255,
}
okabeito_palette = okabeito_palette_str.values()
mpl_okabeito_cmap = ListedColormap(okabeito_palette)

gray_palette = [(c, c, c) for c in np.linspace(0, 1, n_colors + 1)]

gray_palette_str = {f"{n_colors - p}0": v for p, v in enumerate(gray_palette)}
gray_palette_str["black"] = gray_palette_str["100"]
gray_palette_str["white"] = gray_palette_str["00"]

mpl_gray_cmap = ListedColormap(gray_palette)

blue_palette = sns.light_palette(
    color_palette_str["blue"], reverse=True, n_colors=n_colors + 1
)
blue_palette_str = {f"{n_colors - p}0": v for p, v in enumerate(blue_palette)}
mpl_blue_cmap = ListedColormap(blue_palette)

div_palette = sns.blend_palette(
    [color_palette[0], [1, 1, 1], color_palette[3]], n_colors=n_colors, as_cmap=False
)
mpl_div_cmap = ListedColormap(div_palette)
