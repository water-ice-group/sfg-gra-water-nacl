import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 14,
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
    "mathtext.default": "regular",
})
mpl.rcParams["hatch.linewidth"] = 3.0


atomic_mass_oxygen = 15.999
z_dist_ref = (27.850 - 7.500) / 2
z_tot = 27.850 - 7.500

lw_set = 3
color_water = "#0b1166"
color_na = "#592693"
color_cl = "#429B46"


fig, (ax_bottom, ax_top) = plt.subplots(
    2, 1, figsize=(3.35, 5), sharex=True, gridspec_kw={"height_ratios": [1, 30]}
)
fig.subplots_adjust(hspace=0.05)


def plot_density_profiles(ax):
    data_water = np.load("./density-profiles-data/profile_0_2M.npz")
    pos_water = data_water["positions"] + z_dist_ref
    mean_water = data_water["mean_values"] * 1e3 / atomic_mass_oxygen
    std_water = data_water["std_values"] * 1e3 / atomic_mass_oxygen
    ax.plot(mean_water, pos_water, linewidth=lw_set, label=r"Oxygen (H$\mathrm{_2}$O)", color=color_water, zorder=-5)
    ax.fill_betweenx(pos_water, mean_water - std_water, mean_water + std_water, color=color_water, alpha=0.35, zorder=-5)

    data_cl = np.load("./density-profiles-data/profile_4_2M.npz")
    pos_cl = data_cl["positions"] + z_dist_ref
    mean_cl = data_cl["mean_values"]
    std_cl = data_cl["std_values"]
    ax.plot(mean_cl * 20, pos_cl, linewidth=lw_set, label=r"Cl$^-$ ($\times$20)", color=color_cl, linestyle="--", zorder=-3)
    ax.fill_betweenx(pos_cl, mean_cl * 20 - std_cl * 20, mean_cl * 20 + std_cl * 20, color=color_cl, alpha=0.35, zorder=-3)

    data_na = np.load("./density-profiles-data/profile_3_2M.npz")
    pos_na = data_na["positions"] + z_dist_ref
    mean_na = data_na["mean_values"]
    std_na = data_na["std_values"]
    ax.plot(mean_na * 20, pos_na, linewidth=lw_set, label=r"Na$^+$ ($\times$20)", color=color_na, linestyle=":", zorder=-4)
    ax.fill_betweenx(pos_na, mean_na * 20 - std_na * 20, mean_na * 20 + std_na * 20, color=color_na, alpha=0.35, zorder=-4)


plot_density_profiles(ax_top)
plot_density_profiles(ax_bottom)

ax_bottom.set_ylim(0.4, 0)
ax_top.set_ylim(z_tot / 2, 2.02)

ax_top.text(0, 1.03, "/", ha="center", va="center", fontsize=14, rotation=90, color="black", transform=ax_top.transAxes)
ax_top.text(0, 1.01, "/", ha="center", va="center", fontsize=14, rotation=90, color="black", transform=ax_top.transAxes)
ax_top.text(1, 1.03, "/", ha="center", va="center", fontsize=14, rotation=90, color="black", transform=ax_top.transAxes)
ax_top.text(1, 1.01, "/", ha="center", va="center", fontsize=14, rotation=90, color="black", transform=ax_top.transAxes)
ax_top.text(0, 1.1, "0", ha="center", va="center", fontsize=14, color="black", transform=ax_top.transAxes)
ax_top.text(0.42, 1.1, "100", ha="center", va="center", fontsize=14, color="black", transform=ax_top.transAxes)
ax_top.text(0.79, 1.1, "200", ha="center", va="center", fontsize=14, color="black", transform=ax_top.transAxes)

ax_bottom.spines["bottom"].set_visible(False)
ax_bottom.set_yticks([0.0])
ax_top.spines["top"].set_visible(False)
ax_top.tick_params(labeltop=True)
ax_top.set_xticklabels([])
ax_top.xaxis.set_tick_params(which="major", bottom=True, top=False)

ax_bottom.set_xlabel(r"Density [M]", labelpad=30)
ax_bottom.xaxis.set_label_position("top")
ax_bottom.xaxis.tick_top()
ax_top.set_ylabel(r"Depth [$\AA$]")

ax_top.legend(loc="lower right", handletextpad=0.3, labelspacing=0.1, ncols=1, handlelength=1, edgecolor="k")

os.makedirs("../plots-output", exist_ok=True)
plt.savefig("../plots-output/fig-2-density.pdf", dpi=300, bbox_inches="tight")
plt.show()


hatch_map = {
    "Free": "--",
    "Cat.": "///",
    "An.": "\\\\\\",
}

face_colors = {
    "All": "#0b1166",
    "Free": "#4983c4",
    "Cat.": "#9b6fd0",
    "An.": "#5aae61",
}

edge_colors = {
    "All": "#0b1166",
    "Free": "#4983c4",
    "Cat.": "#6a3d9a",
    "An.": "#1b7837",
}

x_positions = np.array([0.0, 0.5, 1.0, 1.5])

pct_free = np.array([100.0000, 87.2262, 76.8698, 56.7665])
pct_cat = np.array([0.0000, 6.2968, 12.5043, 23.6325])
pct_an = np.array([0.0000, 6.2659, 11.3336, 19.4643])

xs_all = np.array([0.0, 0.5, 1.0, 1.5])
avg_all = np.array([0.093414, 0.091854, 0.088152, 0.086674])
err_all = np.array([0.001099, 0.001131, 0.000615, 0.000922])

xs_free = np.array([0.0, 0.5, 1.0, 1.5])
avg_free = np.array([0.093414, 0.080121, 0.067763, 0.049202])
err_free = np.array([0.001099, 0.001067, 0.000840, 0.000961])

xs_cat_full = np.array([0.0, 0.5, 1.0, 1.5])
avg_cat_full = np.array([0.000000, 0.005784, 0.011023, 0.020483])
err_cat_full = np.array([0.000000, 0.000090, 0.000567, 0.000102])

xs_an_full = np.array([0.0, 0.5, 1.0, 1.5])
avg_an_full = np.array([0.000000, 0.005756, 0.009991, 0.016871])
err_an_full = np.array([0.000000, 0.000141, 0.000286, 0.000355])


def plot_series(ax, xs, ys, yerr, key, z=5):
    marker_map = {
        "All": "s",
        "Free": "o",
        "Cat.": "^",
        "An.": "v",
    }
    linestyle_map = {
        "All": "-",
        "Free": "--",
        "Cat.": "--",
        "An.": "--",
    }

    ax.plot(
        xs,
        ys,
        color=edge_colors[key],
        linestyle=linestyle_map[key],
        linewidth=2.6,
        marker=marker_map[key],
        markersize=7.5 if key == "All" else 8.0,
        markerfacecolor=face_colors[key],
        markeredgecolor=edge_colors[key],
        markeredgewidth=2.0,
        zorder=z,
    )

    ax.errorbar(
        xs,
        ys,
        yerr=yerr,
        fmt="none",
        ecolor=edge_colors[key],
        elinewidth=1.8,
        capsize=3,
        capthick=1.8,
        zorder=z - 0.1,
    )


fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(4.3, 5), sharex=True, gridspec_kw={"height_ratios": [0.35, 0.65], "hspace": 0.08}
)

bar_w = 0.25
intra_gap = 0.05
group_gap = 0.40

x_centers = np.array(x_positions)
n_groups = len(x_centers)
n_bars = 3

cluster_width = n_bars * bar_w + (n_bars - 1) * intra_gap
raw_centers = np.zeros(n_groups)
for i in range(1, n_groups):
    raw_centers[i] = raw_centers[i - 1] + cluster_width + group_gap

a = (x_centers[-1] - x_centers[0]) / (raw_centers[-1] - raw_centers[0]) if raw_centers[-1] != raw_centers[0] else 1.0
b = x_centers[0] - a * raw_centers[0]


def map_pos(raw_x):
    return a * raw_x + b


bar_offsets = np.array([
    -(cluster_width / 2) + 0 * (bar_w + intra_gap),
    -(cluster_width / 2) + 1 * (bar_w + intra_gap),
    -(cluster_width / 2) + 2 * (bar_w + intra_gap),
])

x_raw_groups = raw_centers[:, None] + bar_offsets[None, :]
shift = 0.05
x_plot = map_pos(x_raw_groups) + shift

x_free = x_plot[:, 0].copy()
x_free[0] = x_plot[0, 1]

ax_top.bar(
    x_free,
    pct_free,
    width=a * bar_w,
    label="H$_2$O solv. H$_2$O",
    color=face_colors["Free"],
    edgecolor=edge_colors["Free"],
    linewidth=1.5,
    zorder=-2,
)
ax_top.bar(
    x_free,
    pct_free,
    width=a * bar_w,
    facecolor="none",
    edgecolor=edge_colors["Free"],
    hatch=hatch_map["Free"],
    linewidth=0,
    alpha=0.3,
    zorder=-1,
)

ax_top.bar(
    x_plot[1:, 1],
    pct_cat[1:],
    width=a * bar_w,
    label="H$_2$O solv. Na$^{+}$",
    color=face_colors["Cat."],
    edgecolor=edge_colors["Cat."],
    linewidth=1.5,
    zorder=-2,
)
ax_top.bar(
    x_plot[1:, 1],
    pct_cat[1:],
    width=a * bar_w,
    facecolor="none",
    edgecolor=edge_colors["Cat."],
    hatch=hatch_map["Cat."],
    linewidth=0,
    alpha=0.3,
    zorder=-1,
)

ax_top.bar(
    x_plot[1:, 2],
    pct_an[1:],
    width=a * bar_w,
    label="H$_2$O solv. Cl$^{-}$",
    color=face_colors["An."],
    edgecolor=edge_colors["An."],
    linewidth=1.5,
    zorder=-2,
)
ax_top.bar(
    x_plot[1:, 2],
    pct_an[1:],
    width=a * bar_w,
    facecolor="none",
    edgecolor=edge_colors["An."],
    hatch=hatch_map["An."],
    linewidth=0,
    alpha=0.3,
    zorder=-1,
)

ax_top.set_ylim(0, 110)
ax_top.set_yticks([25, 50, 75, 100])
ax_top.set_ylabel("Type [%]")
ax_top.tick_params(axis="x", labelbottom=False)

plot_series(ax_bot, xs_all, avg_all, err_all, "All", z=5)
plot_series(ax_bot, xs_free, avg_free, err_free, "Free", z=6)
plot_series(ax_bot, xs_cat_full, avg_cat_full, err_cat_full, "Cat.", z=6)
plot_series(ax_bot, xs_an_full, avg_an_full, err_an_full, "An.", z=6)

pure_line = Line2D(
    [0], [0],
    color=edge_colors["All"], lw=3, linestyle="-", marker="s", markersize=7.5,
    markerfacecolor=face_colors["All"], markeredgecolor=edge_colors["All"], markeredgewidth=2,
    label="All H$_2$O",
)

free_line = Line2D(
    [0], [0],
    color=edge_colors["Free"], lw=3, linestyle="--", marker="o", markersize=8.5,
    markerfacecolor=face_colors["Free"], markeredgecolor=edge_colors["Free"], markeredgewidth=2,
    label="H$_2$O not solv. ions",
)

na_line = Line2D(
    [0], [0],
    color=edge_colors["Cat."], lw=3, linestyle="--", marker="^", markersize=8,
    markerfacecolor=face_colors["Cat."], markeredgecolor=edge_colors["Cat."], markeredgewidth=2,
    label="H$_2$O solv. Na$^{+}$",
)

cl_line = Line2D(
    [0], [0],
    color=edge_colors["An."], lw=3, linestyle="--", marker="v", markersize=8,
    markerfacecolor=face_colors["An."], markeredgecolor=edge_colors["An."], markeredgewidth=2,
    label="H$_2$O solv. Cl$^{-}$",
)

ax_bot.legend(
    handles=[pure_line, free_line, na_line, cl_line],
    loc="center left",
    edgecolor="k",
    bbox_to_anchor=(0.01, 0.44),
    labelspacing=0.05,
    handlelength=0.8,
)

custom_xtick_labels = ["Pure\nH$_2$O", "0.5M\nNaCl", "1M\nNaCl", "2M\nNaCl"]
ax_bot.set_xticks(x_positions)
ax_bot.set_xticklabels(custom_xtick_labels)
ax_bot.grid(alpha=0.3)
ax_bot.set_ylim(-0.02 / 2, 0.205 / 2)
ax_bot.set_xlim(-0.13, 1.73)
ax_bot.set_ylabel("Surface density \n[$N_{H_{2}O}$/Å$^2$]")

ax_top.set_xlim(ax_bot.get_xlim())

plt.tight_layout()
os.makedirs("../plots-output", exist_ok=True)
plt.savefig("../plots-output/fig-2-percentage-dens.pdf", dpi=300, bbox_inches="tight")
plt.show()
print("Plots generated and saved to ../plots-output/")
