import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator, NullFormatter
from scipy.ndimage import gaussian_filter


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


colors = [
    (0.2549019607843137, 0.17254901960784313, 0.7411764705882353),
    (0.1803921568627451, 0.7333333333333333, 0.7529411764705882),
    (0.9607843137254902, 0.9137254901960784, 0.13725490196078433),
]
cmap = LinearSegmentedColormap.from_list(
    "custom_blue_to_bright_yellow",
    colors,
    N=256,
)


STORING_PATH = "total-data"
BINS_X = np.linspace(2.25, 4.55, 60)
BINS_Y = np.linspace(0, 180, 60)
SELECTED_REGIONS = ["top", "bot"]

SYMLINTHRESH = 0.09
SYMLINSCALE_TOP = 0.15
SYMLINSCALE_BOT = 0.15
SYMB_BASE = 10

SMOOTH_SIGMA = 0.9

LEVELS_PER_DEC_TOP = 12
LEVELS_PER_DEC_BOT = 12
LIN_PTS = 6


def extract_angle_distance_data(data_all_region):
    z_dist_all, angles_all = [], []
    for frame in data_all_region:
        for angle1, angle2, z_dist, *_ in frame:
            z_dist_all.extend([z_dist, z_dist])
            angles_all.extend([angle1, angle2])
    return np.asarray(z_dist_all), np.asarray(angles_all)


def compute_hist(z, angles, bins_x, bins_y):
    counts, xedges, yedges = np.histogram2d(
        z,
        angles,
        bins=[bins_x, bins_y],
        density=True,
    )
    m = np.max(counts)
    hist = counts / m if m > 0 else counts
    return hist, xedges, yedges


def load_and_merge(path_prefix, conc, regions):
    data = np.load(f"{path_prefix}_{conc}.npy", allow_pickle=True).item()
    merged = []
    for region in regions:
        merged.extend(data.get(region, []))
    return extract_angle_distance_data(merged)


def _fmt_plain(v, _):
    if abs(v) < 1e-12:
        return "0"
    s = f"{v:.2f}"
    return s.rstrip("0").rstrip(".")


def symlog_levels(vmin, vmax, linthresh, base=10, n_per_dec=12, symmetric=False, lin_pts=8):
    lin = np.linspace(0, linthresh, lin_pts, endpoint=True)

    if vmax > linthresh:
        pos_max_dec = int(np.ceil(np.log(vmax / linthresh) / np.log(base)))
        pos_logs = []
        for d in range(pos_max_dec):
            a = linthresh * (base ** d)
            b = linthresh * (base ** (d + 1))
            b = min(b, vmax)
            pos_logs.append(np.geomspace(a, b, n_per_dec, endpoint=(b == vmax)))
        pos = np.unique(np.concatenate(pos_logs)) if pos_logs else np.array([])
    else:
        pos = np.array([])

    if symmetric:
        neg = -np.flip(np.concatenate(([0.0], lin[1:], pos)))
        levels = np.unique(np.concatenate((neg, lin, pos)))
    else:
        levels = np.unique(np.concatenate(([0.0], lin[1:], pos)))

    levels = levels[(levels >= vmin) & (levels <= vmax)]
    return levels


def make_plot(case_label, case_prefix, save_name):
    z_wat, angles_wat = load_and_merge(
        f"{STORING_PATH}/neutral_angles_total",
        "WAT",
        SELECTED_REGIONS,
    )
    z_case, angles_case = load_and_merge(
        f"{STORING_PATH}/{case_prefix}",
        "2M",
        SELECTED_REGIONS,
    )

    hist_wat, _, _ = compute_hist(z_wat, angles_wat, BINS_X, BINS_Y)
    hist_case, _, _ = compute_hist(z_case, angles_case, BINS_X, BINS_Y)

    if SMOOTH_SIGMA and SMOOTH_SIGMA > 0:
        hist_wat = gaussian_filter(hist_wat, sigma=SMOOTH_SIGMA)
        hist_case = gaussian_filter(hist_case, sigma=SMOOTH_SIGMA)

    eps = 1e-12
    hist_wat = hist_wat / (hist_wat.sum() + eps)
    hist_case = hist_case / (hist_case.sum() + eps)

    diff = hist_case - hist_wat

    print(f"▶ {save_name}:")
    print(f"  Sum(hist_case) = {hist_case.sum():.6f}")
    print(f"  Sum(hist_wat)  = {hist_wat.sum():.6f}")
    print(f"  Sum(diff)      = {diff.sum():.6f}")

    marginal_angle_case = np.sum(hist_case, axis=0)
    angle_centers = 0.5 * (BINS_Y[:-1] + BINS_Y[1:])

    x_centers = 0.5 * (BINS_X[:-1] + BINS_X[1:])
    y_centers = 0.5 * (BINS_Y[:-1] + BINS_Y[1:])
    Xc, Yc = np.meshgrid(x_centers, y_centers, indexing="xy")

    fig = plt.figure(figsize=(3.85, 8))
    main_gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.1)

    sub_gs_top = gridspec.GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=main_gs[0],
        width_ratios=[0.85, 0.15],
        wspace=-0.02,
    )
    ax0 = fig.add_subplot(sub_gs_top[0])

    hist_case_scaled = hist_case * 1000
    vmax_case = 2.52

    norm_top = SymLogNorm(
        linthresh=SYMLINTHRESH,
        linscale=SYMLINSCALE_TOP,
        vmin=0.0,
        vmax=vmax_case,
        base=SYMB_BASE,
    )
    levels_top = symlog_levels(
        vmin=0.0,
        vmax=vmax_case,
        linthresh=SYMLINTHRESH,
        base=SYMB_BASE,
        n_per_dec=LEVELS_PER_DEC_TOP,
        symmetric=False,
        lin_pts=LIN_PTS,
    )

    cs0 = ax0.contourf(
        Xc,
        Yc,
        hist_case_scaled.T,
        levels=levels_top,
        cmap=cmap,
        norm=norm_top,
        extend="max",
    )
    levels_top_lines = levels_top[levels_top > 0]
    ax0.contour(
        Xc,
        Yc,
        hist_case_scaled.T,
        levels=levels_top_lines,
        norm=norm_top,
        colors="#444444",
        linewidths=0.4,
        alpha=0.4,
        zorder=10,
    )

    ax0.set_aspect("auto")
    ax0.set_xticklabels([])
    ax0.set_ylabel(r"O-H angle, $\alpha$ [°]")
    ax0.set_yticks([0, 25, 50, 75, 100, 125, 150, 175])
    ax0.set_xticks([2.5, 3.0, 3.5, 4.0, 4.5])

    ax1 = fig.add_subplot(sub_gs_top[1], sharey=ax0)
    ax1.plot(marginal_angle_case, angle_centers, color="#0b1166", linewidth=2, zorder=-5)
    ax1.fill_betweenx(angle_centers, 0, marginal_angle_case, color="#0b1166", alpha=0.35, zorder=-6)
    ax1.axis("off")

    cbar0 = fig.colorbar(
        cs0,
        ax=[ax0, ax1],
        orientation="vertical",
        fraction=0.046,
        pad=0.03,
        extendrect=True,
    )
    cbar0.set_label("Probability density, P")

    ticks_top = [0, 0.25, 0.6, 1.0, 1.5, 2.5]
    cbar0.ax.yaxis.set_minor_locator(NullLocator())
    cbar0.ax.yaxis.set_minor_formatter(NullFormatter())
    cbar0.ax.yaxis.set_major_locator(FixedLocator(ticks_top))
    cbar0.ax.yaxis.set_major_formatter(FuncFormatter(_fmt_plain))
    cbar0.locator = FixedLocator(ticks_top)
    cbar0.minorticks_off()
    cbar0.update_ticks()

    sub_gs_bottom = gridspec.GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=main_gs[1],
        width_ratios=[0.85, 0.15],
        wspace=-0.02,
    )
    ax2 = fig.add_subplot(sub_gs_bottom[0])

    diff_scaled = diff * 1000

    vmin_diff, vmax_diff = -1.01, 1.01
    diff_clipped = np.clip(diff_scaled, vmin_diff, vmax_diff)

    norm_diff = SymLogNorm(
        linthresh=SYMLINTHRESH,
        linscale=SYMLINSCALE_BOT,
        vmin=vmin_diff,
        vmax=vmax_diff,
        base=SYMB_BASE,
    )

    levels_diff = symlog_levels(
        vmin=vmin_diff,
        vmax=vmax_diff,
        linthresh=SYMLINTHRESH,
        base=SYMB_BASE,
        n_per_dec=LEVELS_PER_DEC_BOT,
        symmetric=True,
        lin_pts=LIN_PTS,
    )

    cs2 = ax2.contourf(
        Xc,
        Yc,
        diff_clipped.T,
        levels=levels_diff,
        cmap="bwr",
        norm=norm_diff,
        extend="neither",
    )

    levels_diff_lines = levels_diff[levels_diff != 0]
    ax2.contour(
        Xc,
        Yc,
        diff_clipped.T,
        levels=levels_diff_lines,
        norm=norm_diff,
        colors="#444444",
        linewidths=0.4,
        alpha=0.4,
        zorder=10,
    )

    ax2.set_aspect("auto")
    ax2.set_xlabel("Depth [Å]")
    ax2.set_ylabel(r"O-H angle, $\alpha$ [°]")
    ax2.set_yticks([0, 25, 50, 75, 100, 125, 150, 175])
    ax2.set_xticks([2.5, 3.0, 3.5, 4.0, 4.5])

    ax3 = fig.add_subplot(sub_gs_bottom[1], sharey=ax2)
    ax3.axis("off")

    cbar2 = fig.colorbar(
        cs2,
        ax=[ax2, ax3],
        orientation="vertical",
        fraction=0.046,
        pad=0.02,
        extendrect=True,
    )

    if save_name.startswith("fig-3-free"):
        cbar2.set_label(r"$P^{\mathrm{solv.\,} H_{2}O}_{H_{2}O} - P^{\mathrm{pure}}_{H_{2}O}$")
    elif save_name.startswith("fig-3-cl"):
        cbar2.set_label(r"$P^{\mathrm{solv.\,} Cl^{-}}_{H_{2}O} - P^{\mathrm{pure}}_{H_{2}O}$")
    elif save_name.startswith("fig-3-na"):
        cbar2.set_label(r"$P^{\mathrm{solv.\,} Na^{+}}_{H_{2}O} - P^{\mathrm{pure}}_{H_{2}O}$")
    else:
        cbar2.set_label(r"$\Delta P$")

    ticks_bottom = [-1.0, -0.6, -0.2, 0, 0.2, 0.6, 1.0]
    cbar2.ax.yaxis.set_minor_locator(NullLocator())
    cbar2.ax.yaxis.set_minor_formatter(NullFormatter())
    cbar2.ax.yaxis.set_major_locator(FixedLocator(ticks_bottom))
    cbar2.ax.yaxis.set_major_formatter(FuncFormatter(_fmt_plain))
    cbar2.locator = FixedLocator(ticks_bottom)
    cbar2.minorticks_off()
    cbar2.update_ticks()

    plt.tight_layout()
    os.makedirs("../plots-output", exist_ok=True)
    outpath = f"../plots-output/{save_name}.pdf"
    plt.savefig(outpath, dpi=300, bbox_inches="tight", transparent=True)
    plt.show()
    print(f"Saved: {outpath}")


make_plot(case_label="Free waters (2M)", case_prefix="neutral_angles_total", save_name="fig-4-free_cont")
make_plot(case_label="Cl⁻ waters (2M)", case_prefix="cl_angles_total", save_name="fig-4-cl_cont")
make_plot(case_label="Na⁺ waters (2M)", case_prefix="na_angles_total", save_name="fig-4-na_cont")