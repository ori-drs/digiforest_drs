from matplotlib import pyplot as plt
import matplotlib
import csv

# load csv
with open("/home/ori/git/digiforest_drs/trees/logs/full_run.csv", "r") as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    # skip header
    next(reader)
    # load data
    data = list(reader)
    data = [
        {"id": int(id), "co_x": float(co_x), "co_y": float(co_y), "dbh": float(dbh)}
        for id, co_x, co_y, _, _, dbh, *_ in data
        if float(dbh) < 0.5
    ]

matplotlib.use("TkAgg")
plt.figure(figsize=(16, 9))
plt.title("Martelloscope - Tree Manager State")
plt.gca().set_aspect("equal", adjustable="box")
plt.scatter([d["co_x"] for d in data], [d["co_y"] for d in data], s=0.001)
plt.xlabel("x [m]")
plt.ylabel("y [m]")
for d in data:
    # add circle for each tree
    circle = plt.Circle((d["co_x"], d["co_y"]), d["dbh"] / 2, color="r", fill=False)
    plt.annotate(
        "tree" + str(d["id"]).zfill(3),
        (d["co_x"], d["co_y"]),
        fontsize=8,
        ha="center",
        va="bottom",
    )
    plt.gcf().gca().add_artist(circle)
plt.show()
