import os
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.ticker as mtick


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


def parse_hb_output(text, hb_types):
    means = {}
    stderr = {}

    current = None
    block_header_re = re.compile(r"^\s*(.+):\s*$")
    line_re = re.compile(
        r"^\s*([A-Za-z0-9]+)\s*:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*±\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*$"
    )

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("HB fractions"):
            continue
        if set(line) == {"="}:
            continue

        m_hdr = block_header_re.match(raw)
        if m_hdr and ("±" not in raw) and (not line_re.match(raw)):
            current = m_hdr.group(1).strip()
            means[current] = [np.nan] * len(hb_types)
            stderr[current] = [np.nan] * len(hb_types)
            continue

        m = line_re.match(raw)
        if m and current is not None:
            hb = m.group(1).strip()
            if hb in hb_types:
                idx = hb_types.index(hb)
                means[current][idx] = float(m.group(2))
                stderr[current][idx] = float(m.group(3))

    return means, stderr


def parse_avg_hb_output(text):
    means = {}
    sem = {}

    current = None
    header_re = re.compile(r"^\s*(.+):\s*$")
    line_re = re.compile(
        r"^\s*(Total|Intralayer|Interlayer)\s*:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*±\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*$"
    )

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("Average H-bonds"):
            continue
        if set(line) == {"="}:
            continue

        m = line_re.match(raw)
        if m and current is not None:
            key = m.group(1)
            means[current][key] = float(m.group(2))
            sem[current][key] = float(m.group(3))
            continue

        if ":" in raw and ("±" not in raw) and (not line_re.match(raw)):
            m_hdr = header_re.match(raw)
            if m_hdr:
                current = m_hdr.group(1).strip()
                means[current] = {"Total": np.nan, "Intralayer": np.nan, "Interlayer": np.nan}
                sem[current] = {"Total": np.nan, "Intralayer": np.nan, "Interlayer": np.nan}

    return means, sem


def darken(hexcolor, factor=0.6):
    rgb = np.array(mcolors.to_rgb(hexcolor))
    return mcolors.to_hex(rgb * factor)


def lighten(hexcolor, factor=0.25):
    rgb = np.array(mcolors.to_rgb(hexcolor))
    return mcolors.to_hex(rgb * (1 - factor) + factor)


hb_output = r"""
HB fractions (mean ± stderr)
================================

Pure H$_2$O:
  DDAA: 0.3025 ± 0.0024
  DDA : 0.2475 ± 0.0006
  DAA : 0.1981 ± 0.0003
  DA  : 0.1798 ± 0.0016

0.5M NaCl:
  DDAA: 0.2748 ± 0.0026
  DDA : 0.2283 ± 0.0012
  DAA : 0.2010 ± 0.0010
  DA  : 0.1871 ± 0.0016

1M NaCl:
  DDAA: 0.2407 ± 0.0019
  DDA : 0.2088 ± 0.0014
  DAA : 0.2000 ± 0.0010
  DA  : 0.2004 ± 0.0012

2M NaCl:
  DDAA: 0.1938 ± 0.0039
  DDA : 0.1767 ± 0.0025
  DAA : 0.1937 ± 0.0016
  DA  : 0.2120 ± 0.0017
"""

avg_hb_output = r"""
Average H-bonds (mean ± SEM)
================================

Pure H$_2$O:
  Total      : 3.038 ± 0.005
  Intralayer : 2.189 ± 0.005
  Interlayer : 0.849 ± 0.001

0.5M
NaCl:
  Total      : 2.940 ± 0.007
  Intralayer : 2.121 ± 0.007
  Interlayer : 0.819 ± 0.002

1M
NaCl:
  Total      : 2.815 ± 0.006
  Intralayer : 2.028 ± 0.006
  Interlayer : 0.788 ± 0.002

2M
NaCl:
  Total      : 2.614 ± 0.015
  Intralayer : 1.889 ± 0.015
  Interlayer : 0.724 ± 0.003
"""
avg_hb_output = re.sub(r"(0\.5M|1M|2M)\s*\n\s*(NaCl:)", r"\1 \2", avg_hb_output)


# -------------------------
# HB fractions bar plot
# -------------------------
hb_types = ["DDAA", "DDA", "DAA", "DA"]
means_frac, stderr_frac = parse_hb_output(hb_output, hb_types)

conc_order = ["Pure H$_2$O", "0.5M NaCl", "1M NaCl", "2M NaCl"]
conc_labels = [lbl for lbl in conc_order if lbl in means_frac]
nC = len(conc_labels)

group_gap = 1.25
x = np.arange(len(hb_types)) * group_gap

bar_width = 0.18
offset_scale = 1.2
offsets = (np.arange(nC) - (nC - 1) / 2.0) * (bar_width * offset_scale)

fig, ax = plt.subplots(figsize=(6, 4))

base_colors = ["#0b1166", "#364fa2", "#4983c4", "#3ab6e8"][:nC]
edge_colors = [darken(c, 0.6) for c in base_colors]
plt.rcParams["hatch.linewidth"] = 3.0

hatch = "--"
thick_cap = 1.75

for cidx, (clabel, face, edge) in enumerate(zip(conc_labels, base_colors, edge_colors)):
    vals = [means_frac[clabel][hb_types.index(hb)] for hb in hb_types]
    errs = [stderr_frac[clabel][hb_types.index(hb)] for hb in hb_types]
    xpos = x + offsets[cidx]

    ax.bar(xpos, vals, bar_width, color=face, edgecolor=edge, linewidth=1.5, label=clabel, zorder=-2)
    ax.bar(xpos, vals, bar_width, facecolor="none", edgecolor=edge, hatch=hatch, linewidth=0, alpha=0.3, zorder=-1)

    eb = ax.errorbar(
        xpos, vals, yerr=errs,
        fmt="none", ecolor="k",
        elinewidth=thick_cap, capsize=4, capthick=thick_cap,
        zorder=3,
    )
    for cap in eb[1]:
        cap.set_markeredgewidth(thick_cap)

ax.set_xticks(x)
ax.set_xticklabels(hb_types)
ax.set_ylim(0.15, 0.41)
ax.set_ylabel("Fraction")
ax.legend(
    loc="upper right",
    bbox_to_anchor=(1, 0.64),
    ncols=1,
    handletextpad=0.5,
    labelspacing=0.1,
    handlelength=1,
    edgecolor="k",
)
ax.margins(x=0.05)

plt.tight_layout()
out_dir = "../plots-output"
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, "fig-5-hb-fraction.pdf"), dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved figure fig-5-hb-fraction.pdf")

# -------------------------
# Average H-bonds triplet
# -------------------------
means_avg, sem_avg = parse_avg_hb_output(avg_hb_output)

labels_order = ["Pure H$_2$O", "0.5M NaCl", "1M NaCl", "2M NaCl"]
labels = [lbl for lbl in labels_order if lbl in means_avg]

vals_total = np.array([means_avg[lbl]["Total"] for lbl in labels], dtype=float)
vals_intra = np.array([means_avg[lbl]["Intralayer"] for lbl in labels], dtype=float)
vals_inter = np.array([means_avg[lbl]["Interlayer"] for lbl in labels], dtype=float)

err_total = np.array([sem_avg[lbl]["Total"] for lbl in labels], dtype=float)
err_intra = np.array([sem_avg[lbl]["Intralayer"] for lbl in labels], dtype=float)
err_inter = np.array([sem_avg[lbl]["Interlayer"] for lbl in labels], dtype=float)

x = np.arange(len(labels))
base_colors = ["#0b1166", "#364fa2", "#4983c4", "#3ab6e8"][: len(labels)]
edge_colors = [darken(c, 0.6) for c in base_colors]

width = 0.18
gap = 0.04
thick_cap = 1.75

fig, ax = plt.subplots(figsize=(6, 4.0))

for i in range(len(labels)):
    face_total = base_colors[i]
    face_intra = lighten(base_colors[i], 0.25)
    face_inter = lighten(base_colors[i], 0.45)

    x_total = x[i] - (width + gap)
    x_intra = x[i]
    x_inter = x[i] + (width + gap)

    ax.bar(x_total, vals_total[i], width, color=face_total, edgecolor=edge_colors[i], linewidth=1.5, zorder=-2)
    if not np.isclose(err_total[i], 0.0):
        eb = ax.errorbar(
            x_total, vals_total[i], yerr=err_total[i],
            fmt="none", ecolor="k",
            elinewidth=thick_cap, capsize=4, capthick=thick_cap,
            zorder=3,
        )
        for cap in eb[1]:
            cap.set_color("black")
            cap.set_markeredgewidth(thick_cap)
    ax.bar(x_total, vals_total[i], width, facecolor="none", edgecolor=edge_colors[i], hatch="--",
           linewidth=0, alpha=0.30, zorder=-1)

    ax.bar(x_intra, vals_intra[i], width, color=face_intra, edgecolor=edge_colors[i], linewidth=1.5, zorder=-2)
    if not np.isclose(err_intra[i], 0.0):
        eb = ax.errorbar(
            x_intra, vals_intra[i], yerr=err_intra[i],
            fmt="none", ecolor="k",
            elinewidth=thick_cap, capsize=4, capthick=thick_cap,
            zorder=3,
        )
        for cap in eb[1]:
            cap.set_color("black")
            cap.set_markeredgewidth(thick_cap)
    ax.bar(x_intra, vals_intra[i], width, facecolor="none", edgecolor=edge_colors[i], hatch="\\\\\\",
           linewidth=0, alpha=0.25, zorder=-1)

    ax.bar(x_inter, vals_inter[i], width, color=face_inter, edgecolor=edge_colors[i], linewidth=1.5, zorder=-2)
    if not np.isclose(err_inter[i], 0.0):
        eb = ax.errorbar(
            x_inter, vals_inter[i], yerr=err_inter[i],
            fmt="none", ecolor="k",
            elinewidth=thick_cap, capsize=4, capthick=thick_cap,
            zorder=3,
        )
        for cap in eb[1]:
            cap.set_color("black")
            cap.set_markeredgewidth(thick_cap)
    ax.bar(x_inter, vals_inter[i], width, facecolor="none", edgecolor=edge_colors[i], hatch="///",
           linewidth=0, alpha=0.30, zorder=-1)

x_labels = ["Pure H$_2$O", "0.5M\nNaCl", "1M\nNaCl", "2M\nNaCl"]
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.set_ylabel("Average H-Bonds")
ax.grid(axis="y", alpha=0.3)

bulk_val = 3.520167250451478
bulk_err = 0.006014099443621756
ax.axhline(bulk_val, color="k", linestyle="--", linewidth=2.0, zorder=-4, alpha=0.75)
ax.axhspan(bulk_val - bulk_err, bulk_val + bulk_err, color="gray", alpha=0.15, zorder=-5)

sample = 1 if len(base_colors) > 1 else 0
legend_handles = [
    mpatches.Patch(facecolor=base_colors[sample], edgecolor=edge_colors[sample], label="Total", hatch="--"),
    mpatches.Patch(facecolor=lighten(base_colors[sample], 0.25), edgecolor=edge_colors[sample], label="Intralayer", hatch="\\\\\\"),
    mpatches.Patch(facecolor=lighten(base_colors[sample], 0.45), edgecolor=edge_colors[sample], label="Interlayer", hatch="////"),
]
ax.legend(
    handles=legend_handles,
    loc="upper right",
    ncols=1,
    handletextpad=0.5,
    labelspacing=0.1,
    handlelength=1,
    edgecolor="k",
)

ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))

plt.tight_layout()
out_dir = "../plots-output"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "fig-5-hb-intra-inter-total-triplet.pdf")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved figure: {out_path}")
