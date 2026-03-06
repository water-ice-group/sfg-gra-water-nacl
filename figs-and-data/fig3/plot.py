import os

import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 15,
    "axes.titlesize": 21,
    "axes.grid": True,
    "axes.linewidth": 1.25,
    "xtick.top": True,
    "ytick.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 1.25,
    "ytick.major.width": 1.25,
    "xtick.minor.width": 1.25,
    "ytick.minor.width": 1.25,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "grid.linestyle": "dashed",
    "grid.linewidth": 1,
    "grid.alpha": 0.3,
    "figure.figsize": (5, 4),
    "legend.fontsize": 12,
})
plt.rcParams.update({"mathtext.default": "regular"})


norm_to_scale_up_aesthetic = 100
lw_set = 1.9
z_ref_idx = "zref_0"

wat_dir = "./processed_data_all"

dirs = {
    "Free": "./processed_data_free",
    "Na+": "./processed_data_cat",
    "Cl-": "./processed_data_an",
}

normalization_factors = {
    "Free": {"WAT": 1 / 84.70752208, "05M": 1 / 73.56918114, "1M": 1 / 62.75964024, "2M": 1 / 46.2921424},
    "Cl-": {"WAT": 1 / 84.70752208, "05M": 1 / 5.56201666, "1M": 1 / 10.09718637, "2M": 1 / 15.67624448},
    "Na+": {"WAT": 1 / 84.70752208, "05M": 1 / 4.22471612, "1M": 1 / 8.77088239, "2M": 1 / 16.32974},
}

color_map_free = {
    "05M": "#364fa2",
    "1M": "#4983c4",
    "2M": "#3ab6e8",
}

color_map_na = {
    "05M": "#6a3d9a",
    "1M": "#9b6fd0",
    "2M": "#c2a4e0",
}

color_map_cl = {
    "05M": "#1b7837",
    "1M": "#5aae61",
    "2M": "#b8e186",
}

wat_special_color = "k"
line_list = {m: "-" for m in ["WAT", "05M", "1M", "2M"]}

labels = {
    "05M": "0.5M NaCl",
    "1M": "1M NaCl",
    "2M": "2M NaCl",
    "WAT": "Pure water",
}


fig, axes = plt.subplots(1, 3, figsize=(9.5, 4.4), sharey=True)
molarities = ["05M", "1M", "2M"]

wat_file = f"{wat_dir}/WAT_{z_ref_idx}.npz"
wat_positions, wat_mean_values, wat_std_values = None, None, None

if os.path.exists(wat_file):
    data = np.load(wat_file)
    wat_positions = data["positions"]
    wat_mean_values = data["mean_values"]
    wat_std_values = data["std_values"]
else:
    print("Pure water file missing.")

panel_color_maps = {
    "Free": color_map_free,
    "Na+": color_map_na,
    "Cl-": color_map_cl,
}

for ax, (key, storing_dir) in zip(axes, dirs.items()):
    cmap_used = panel_color_maps[key]

    if wat_positions is not None and "WAT" in normalization_factors[key]:
        norm = normalization_factors[key]["WAT"]
        mean_wat = wat_mean_values * norm * norm_to_scale_up_aesthetic
        std_wat = wat_std_values * norm * norm_to_scale_up_aesthetic

        ax.plot(
            wat_positions,
            mean_wat,
            label="Pure H$_2$O",
            color=wat_special_color,
            linestyle="--",
            linewidth=lw_set,
        )
        ax.fill_between(
            wat_positions,
            mean_wat - std_wat,
            mean_wat + std_wat,
            color=wat_special_color,
            alpha=0.2,
        )

    for molarity in molarities:
        filename = f"{storing_dir}/{molarity}_{z_ref_idx}.npz"
        if not os.path.exists(filename):
            print(f"File {filename} not found, skipping.")
            continue

        data = np.load(filename)
        norm = normalization_factors[key][molarity]
        positions = data["positions"]
        mean_values = data["mean_values"] * norm
        std_values = data["std_values"] * norm

        ax.plot(
            positions,
            mean_values * norm_to_scale_up_aesthetic,
            label=labels[molarity],
            color=cmap_used[molarity],
            linestyle=line_list[molarity],
            linewidth=lw_set,
            alpha=1.0,
        )
        ax.fill_between(
            positions,
            mean_values * norm_to_scale_up_aesthetic - std_values * norm_to_scale_up_aesthetic,
            mean_values * norm_to_scale_up_aesthetic + std_values * norm_to_scale_up_aesthetic,
            alpha=0.15,
            color=cmap_used[molarity],
        )

    ax.set_xlim(2950, 3850)
    ax.axhline(0, color="dimgray", linestyle="-", linewidth=2, alpha=0.35, zorder=-2)
    ax.set_xlabel(r"Frequency [cm$^{-1}$]")

axes[0].set_ylabel(r"Norm. Im($\chi^{(2)}$) [a.u.]")

ypos_leg = 0.87
xpos_leg = -0.01

for ax in axes:
    ax.legend(
        loc="upper left",
        handletextpad=0.3,
        labelspacing=0.1,
        ncols=2,
        handlelength=0.4,
        edgecolor="k",
        columnspacing=0.9,
        bbox_to_anchor=(xpos_leg, ypos_leg),
        bbox_transform=ax.transAxes,
    )

axes[1].plot(
    [3715, 3715],
    [0.00298 * norm_to_scale_up_aesthetic, 0.0038 * norm_to_scale_up_aesthetic],
    color="#4b4b4b",
    lw=4.5,
    solid_capstyle="projecting",
)
axes[1].annotate(
    "",
    xy=(3715, 0.00415 * norm_to_scale_up_aesthetic),
    xytext=(3715, 0.00385 * norm_to_scale_up_aesthetic),
    arrowprops=dict(
        arrowstyle="-|>",
        color="#4b4b4b",
        lw=0,
        mutation_scale=30,
    ),
)

axes[1].plot(
    [3470, 3470],
    [-0.0011 * norm_to_scale_up_aesthetic, -0.0005 * norm_to_scale_up_aesthetic],
    color="#4b4b4b",
    lw=4.5,
    solid_capstyle="projecting",
)
axes[1].annotate(
    "",
    xy=(3470, -0.0001 * norm_to_scale_up_aesthetic),
    xytext=(3470, -0.0004 * norm_to_scale_up_aesthetic),
    arrowprops=dict(
        arrowstyle="-|>",
        color="#4b4b4b",
        lw=0,
        mutation_scale=30,
    ),
)

axes[2].plot(
    [3709, 3709],
    [0.0026 * norm_to_scale_up_aesthetic, 0.0030 * norm_to_scale_up_aesthetic],
    color="#4b4b4b",
    lw=4.5,
    solid_capstyle="projecting",
)
axes[2].annotate(
    "",
    xy=(3709, 0.0033 * norm_to_scale_up_aesthetic),
    xytext=(3709, 0.0030 * norm_to_scale_up_aesthetic),
    arrowprops=dict(
        arrowstyle="-|>",
        color="#4b4b4b",
        mutation_scale=30,
        lw=0,
    ),
)

axes[2].plot(
    [3250, 3250],
    [-0.00073 * norm_to_scale_up_aesthetic, -0.0006 * norm_to_scale_up_aesthetic],
    color="#4b4b4b",
    lw=4.5,
    solid_capstyle="projecting",
)
axes[2].annotate(
    "",
    xy=(3250, -0.0002 * norm_to_scale_up_aesthetic),
    xytext=(3250, -0.0005 * norm_to_scale_up_aesthetic),
    arrowprops=dict(
        arrowstyle="-|>",
        color="#4b4b4b",
        mutation_scale=30,
        lw=0,
    ),
)

axes[2].plot(
    [3450, 3500],
    [-0.0013 * norm_to_scale_up_aesthetic, -0.0013 * norm_to_scale_up_aesthetic],
    color="#4b4b4b",
    lw=4.5,
    solid_capstyle="projecting",
)

axes[1].plot(
    [3250, 3250],
    [-0.00074 * norm_to_scale_up_aesthetic, -0.0004 * norm_to_scale_up_aesthetic],
    color="#4b4b4b",
    lw=4.5,
    solid_capstyle="projecting",
)
axes[1].annotate(
    "",
    xy=(3250, 0.0 * norm_to_scale_up_aesthetic),
    xytext=(3250, -0.0003 * norm_to_scale_up_aesthetic),
    arrowprops=dict(
        arrowstyle="-|>",
        color="#4b4b4b",
        lw=0,
        mutation_scale=30,
    ),
)

axes[0].set_title(r"Im($\chi^{(2)}_{water}$)", pad=12, fontsize=16)
axes[1].set_title(r"Im($\chi^{(2)}_{wat/cation}$)", pad=12, fontsize=16)
axes[2].set_title(r"Im($\chi^{(2)}_{wat/anion}$)", pad=12, fontsize=16)

for ax in axes:
    ax.set_yticklabels([])

axes[0].text(
    0.01,
    0.26,
    "0.0",
    transform=axes[0].transAxes,
    ha="left",
    va="bottom",
    fontsize=11,
    color="black",
)

plt.tight_layout()
plt.subplots_adjust(wspace=0.05)
os.makedirs("../plots-output", exist_ok=True)
plt.savefig("../plots-output/fig-3-theoretical.pdf", dpi=300, bbox_inches="tight")
plt.show()
print("fig-3-theoretical ready")


mult_factor = 10
lw_set = 1.9

color_map = {
    "WAT": "k",
    "05M": "#364fa2",
    "1M": "#4983c4",
    "2M": "#3ab6e8",
}

line_list = {m: "-" for m in ["WAT", "05M", "1M", "2M"]}

labels = {
    "05M": "0.5M NaCl",
    "1M": "1M NaCl",
    "2M": "2M NaCl",
    "WAT": "Pure water",
}

molarities = ["WAT", "05M", "1M", "2M"]
storing_dir = "./sfg-data"

fig, ax = plt.subplots(figsize=(4.4, 4.5))

for molarity in molarities:
    filename = f"{storing_dir}/{molarity}_{z_ref_idx}.npz"
    if not os.path.exists(filename):
        print(f"File {filename} not found, skipping.")
        continue

    data = np.load(filename)
    positions = data["positions"]
    mean_values = data["mean_values"] * mult_factor
    std_values = data["std_values"] * mult_factor

    ax.plot(
        positions,
        mean_values,
        label=labels[molarity],
        color=color_map[molarity],
        linestyle=line_list[molarity],
        alpha=1.0,
        linewidth=lw_set,
    )
    ax.fill_between(
        positions,
        mean_values - std_values,
        mean_values + std_values,
        alpha=0.2,
        color=color_map[molarity],
    )

ax.set_xlim(2950, 3850)
ax.set_ylim(-0.075 * mult_factor, 0.14 * mult_factor)
ax.axhline(0, color="dimgray", linestyle="-", linewidth=2, alpha=0.35, zorder=-2)
ax.set_xlabel(r"Frequency [cm$^{-1}$]")
ax.set_ylabel(r"Im($\chi^{(2)}$) [a.u.]", labelpad=-7)

ax.legend(
    loc="upper left",
    bbox_to_anchor=(-0.01, 0.87),
    handletextpad=0.3,
    labelspacing=0.1,
    ncols=1,
    handlelength=0.6,
    edgecolor="k",
)

ax.set_xticks([3000, 3250, 3500, 3750])

ax.annotate(
    "",
    xy=(3675, 0.11 * mult_factor),
    xycoords="data",
    xytext=(3575, 0.11 * mult_factor),
    textcoords="data",
    arrowprops=dict(arrowstyle="-|>", color="#4b4b4b", lw=3.5),
)
ax.text(
    3475,
    0.11 * mult_factor,
    "Free\nO-H",
    ha="center",
    va="center",
    fontsize=14,
    color="#4b4b4b",
    alpha=1.0,
    transform=ax.transData,
)

ax.annotate(
    "",
    xy=(3450, -0.04 * mult_factor),
    xycoords="data",
    xytext=(3350, 0.015 * mult_factor),
    textcoords="data",
    arrowprops=dict(arrowstyle="-|>", color="#4b4b4b", lw=3.5),
)
ax.text(
    3300,
    0.025 * mult_factor,
    "H-bonded O-H",
    ha="center",
    va="center",
    fontsize=14,
    color="#4b4b4b",
    alpha=1.0,
    transform=ax.transData,
)

ax.set_title(r"Im($\chi^{(2)}_{total}$)", pad=12, fontsize=16)

plt.tight_layout()
plt.subplots_adjust(wspace=0.15)
os.makedirs("../plots-output", exist_ok=True)
plt.savefig("../plots-output/fig-3-tot.pdf", dpi=300, bbox_inches="tight")
plt.show()
print("fig-3-tot ready")
