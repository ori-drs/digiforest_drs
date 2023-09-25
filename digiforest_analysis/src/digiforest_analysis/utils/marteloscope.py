def plot(trees, ax, **kwargs):
    from matplotlib.patches import Circle

    for tree in trees:
        x = tree["info"]["dbh_model"]["position"][0]
        y = tree["info"]["dbh_model"]["position"][1]
        r = tree["info"]["dbh"] / 2
        i = tree["info"]["id"]
        color = tree["info"]["color"]

        # Plot dbh
        circle = Circle((x, y), r, color=color, alpha=1.0)
        ax.add_patch(
            circle,
        )
        ax.text(
            x + 0.1,
            y + 0.1,
            f"{i}",
            fontsize=7,
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
    ax.set_title("Marteloscope")
    ax.autoscale_view()
    ax.set_aspect("equal")
    # ax.legend()
    ax.grid(True)
