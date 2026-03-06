import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 15,
    "axes.titlesize": 21,
    "axes.grid": True,
})
plt.rcParams.update({"mathtext.default": "regular"})

plw = 1.25
plt.rcParams["axes.linewidth"] = plw
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.major.width"] = plw
plt.rcParams["ytick.major.width"] = plw
plt.rcParams["xtick.minor.width"] = plw
plt.rcParams["ytick.minor.width"] = plw
plt.rcParams["xtick.minor.visible"] = False
plt.rcParams["ytick.minor.visible"] = False
plt.rcParams["xtick.major.size"] = 5
plt.rcParams["ytick.major.size"] = 5
plt.rcParams["grid.linestyle"] = "dashed"
plt.rcParams["grid.linewidth"] = 1
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["figure.figsize"] = (5, 4)
plt.rcParams["legend.fontsize"] = 12


filename_water = "data-water.txt"
filename_hcl = "data-hcl.txt"
filename_nacl = "data-nacl.txt"
filename_granacl = "gra-nacl.txt"
filename_grawat = "gra-pure-water.txt"

line_width = 1.9
line_colors = ["#4ca64d", "#4983c4", "#4983c4"]
wat_color = "#0b1166"

mult_factor = 35

water = pd.read_csv(filename_water, sep=r"\s+|,", engine="python")
hcl = pd.read_csv(filename_hcl, sep=r"\s+|,", engine="python")
nacl = pd.read_csv(filename_nacl, sep=r"\s+|,", engine="python")
gra_wat = pd.read_csv(filename_grawat, sep=r"\s+|,", engine="python")
gra_nacl = pd.read_csv(filename_granacl, sep=r"\s+|,", engine="python")

freq_water, im_water = water.iloc[:, 0], water.iloc[:, 1]
freq_hcl, im_hcl = hcl.iloc[:, 0], hcl.iloc[:, 1]
freq_nacl, im_nacl = nacl.iloc[:, 0], nacl.iloc[:, 1]
freq_wat_gra, im_wat_gra = gra_wat.iloc[:, 0], gra_wat.iloc[:, 1] * mult_factor
freq_nacl_gra, im_nacl_gra = gra_nacl.iloc[:, 0], gra_nacl.iloc[:, 1] * mult_factor

fig, axs = plt.subplots(1, 3, figsize=(9, 3.5), sharey=True)

ylims = [-0.82, 0.75]
xlims = [3000, 3775]
ypos_leg = 0.87

axs[0].plot(freq_water, im_water, color=wat_color, lw=line_width, label="Pure H$_2$O", linestyle="--")
axs[0].plot(freq_hcl, im_hcl, color=line_colors[0], lw=line_width + 0.75, label="1M HCl")
axs[0].axhline(0, color="dimgray", linestyle="-", linewidth=2, alpha=0.35, zorder=-2)
axs[0].set_xlabel(r"Frequency [cm$^{-1}$]")
axs[0].set_ylabel(r"Im($\chi^{(2)}$) [a.u.]")
axs[0].set_ylim(ylims)
axs[0].set_xlim(xlims)

yticks = np.arange(-0.75, 0.51, 0.25)
axs[0].set_yticks(yticks)
axs[0].set_yticklabels([f"{y:.2f}" for y in yticks])

axs[0].grid(alpha=0.3)
axs[0].legend(
    loc="upper left",
    handletextpad=0.3,
    labelspacing=0.1,
    ncols=1,
    handlelength=0.8,
    edgecolor="k",
    columnspacing=0.9,
    bbox_to_anchor=(0.025, ypos_leg),
    bbox_transform=axs[0].transAxes,
)

axs[1].plot(freq_water, im_water, color=wat_color, lw=line_width, label="Pure H$_2$O", linestyle="--")
axs[1].plot(freq_nacl, im_nacl, color=line_colors[1], lw=line_width + 0.75, label="1.5M NaCl")
axs[1].axhline(0, color="dimgray", linestyle="-", linewidth=2, alpha=0.35, zorder=-2)
axs[1].set_xlabel(r"Frequency [cm$^{-1}$]")
axs[1].set_ylim(ylims)
axs[1].set_xlim(xlims)
axs[1].grid(alpha=0.3)
axs[1].legend(
    loc="upper left",
    handletextpad=0.3,
    labelspacing=0.1,
    ncols=1,
    handlelength=0.8,
    edgecolor="k",
    columnspacing=0.9,
    bbox_to_anchor=(0.025, ypos_leg),
    bbox_transform=axs[1].transAxes,
)

axs[2].plot(freq_wat_gra, im_wat_gra, color=wat_color, lw=line_width, label="Pure H$_2$O", linestyle="--")
axs[2].plot(freq_nacl_gra, im_nacl_gra, color=line_colors[2], lw=line_width + 0.75, label="1M NaCl")
axs[2].axhline(0, color="dimgray", linestyle="-", linewidth=2, alpha=0.35, zorder=-2)
axs[2].set_xlabel(r"Frequency [cm$^{-1}$]")
axs[2].set_ylim(ylims)
axs[2].set_xlim([3000, 3750])
axs[2].grid(alpha=0.3)
axs[2].legend(
    loc="upper left",
    handletextpad=0.3,
    labelspacing=0.1,
    ncols=1,
    handlelength=0.8,
    edgecolor="k",
    columnspacing=0.9,
    bbox_to_anchor=(0.025, ypos_leg),
    bbox_transform=axs[2].transAxes,
)

norm_to_scale_up_aesthetic = 100
diff_testa = 35

axs[1].plot(
    [3250, 3250],
    [-0.0010 * norm_to_scale_up_aesthetic, -0.0017 * norm_to_scale_up_aesthetic],
    color="#4b4b4b",
    lw=4.5,
    solid_capstyle="projecting",
)
axs[1].annotate(
    "",
    xy=(3250, -0.00003 * norm_to_scale_up_aesthetic),
    xytext=(3250, -0.00006 * norm_to_scale_up_aesthetic),
    arrowprops=dict(
        arrowstyle="-|>",
        color="#4b4b4b",
        mutation_scale=30,
        lw=0,
    ),
)
axs[1].plot(
    [3685, 3685 + diff_testa],
    [0.0066 * norm_to_scale_up_aesthetic, 0.0066 * norm_to_scale_up_aesthetic],
    color="#4b4b4b",
    lw=4.5,
    solid_capstyle="projecting",
)
axs[1].plot(
    [3440, 3440 + diff_testa],
    [-0.005 * norm_to_scale_up_aesthetic, -0.005 * norm_to_scale_up_aesthetic],
    color="#4b4b4b",
    lw=4.5,
    solid_capstyle="projecting",
)

axs[0].plot(
    [3665, 3665],
    [0.0052 * norm_to_scale_up_aesthetic, 0.0062 * norm_to_scale_up_aesthetic],
    color="#4b4b4b",
    lw=4.5,
    solid_capstyle="projecting",
)
axs[0].annotate(
    "",
    xy=(3665, 0.0047 * norm_to_scale_up_aesthetic),
    xytext=(3665, 0.0044 * norm_to_scale_up_aesthetic),
    arrowprops=dict(
        arrowstyle="<|-",
        color="#4b4b4b",
        lw=0,
        mutation_scale=30,
    ),
)
axs[0].plot(
    [3425, 3425],
    [-0.0036 * norm_to_scale_up_aesthetic, -0.0062 * norm_to_scale_up_aesthetic],
    color="#4b4b4b",
    lw=4.5,
    solid_capstyle="projecting",
)
axs[0].annotate(
    "",
    xy=(3425, -0.0066 * norm_to_scale_up_aesthetic),
    xytext=(3425, -0.0069 * norm_to_scale_up_aesthetic),
    arrowprops=dict(
        arrowstyle="<|-",
        color="#4b4b4b",
        lw=0,
        mutation_scale=30,
    ),
)
axs[0].plot(
    [3225, 3225],
    [-0.0011 * norm_to_scale_up_aesthetic, -0.0042 * norm_to_scale_up_aesthetic],
    color="#4b4b4b",
    lw=4.5,
    solid_capstyle="projecting",
)
axs[0].annotate(
    "",
    xy=(3225, -0.0046 * norm_to_scale_up_aesthetic),
    xytext=(3225, -0.0049 * norm_to_scale_up_aesthetic),
    arrowprops=dict(
        arrowstyle="<|-",
        color="#4b4b4b",
        lw=0,
        mutation_scale=30,
    ),
)

plt.tight_layout()
plt.subplots_adjust(wspace=0.1)
os.makedirs("../plots-output", exist_ok=True)
plt.savefig("../plots-output/fig-1.pdf", dpi=300, bbox_inches="tight")
plt.show()
print("Saved fig-1.pdf")
