import matplotlib.pyplot as plt


def plot(trees, ax, cmap="tab20", **kwargs):
    from matplotlib.patches import Circle

    color_map = plt.get_cmap(cmap)

    for tree in trees:
        x = tree["info"]["dbh_model"]["position"][0]
        y = tree["info"]["dbh_model"]["position"][1]
        r = tree["info"]["dbh"] / 2
        i = tree["info"]["id"]

        # Plot dbh
        circle = Circle((x, y), r, color=color_map(i % 20)[:3], alpha=1.0)
        ax.add_patch(
            circle,
        )
        ax.text(
            x + 0.1,
            y + 0.1,
            f"{i}",
            fontsize=9,
            ha="center",
            va="bottom",
        )
    from digiforest_analysis.utils import plotting

    ax.set_axisbelow(True)
    ax.set_aspect("equal")
    ax.grid(which="major", color=plotting.gray_palette_str["20"], linewidth=0.7)
    ax.grid(
        which="minor",
        color=plotting.gray_palette_str["10"],
        linestyle=":",
        linewidth=0.5,
    )
    ax.minorticks_on()

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Tree Locations")
    ax.autoscale_view()
    ax.set_aspect("equal")
    # ax.legend()
    ax.grid(True)
