"""Draw the implemented DualBranchMINet as a publication-style vector figure.

Run from any directory with the project's Python environment. The output PNG
and editable SVG are placed alongside this script. Filter responses and the
graph inset use the actual default model buffers and kernels.
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from model.bci2a import DualBranchMINet
from utils import load_config


COLORS = {
    "ink": "#192335",
    "navy": "#152450",
    "border": "#a5a9ad",
    "blue": "#dcebf8",
    "blue_edge": "#7f9eba",
    "green": "#e3f0d7",
    "green_edge": "#92af79",
    "orange": "#fae3d4",
    "orange_edge": "#d0a78d",
    "purple": "#e4ddf2",
    "purple_edge": "#a596bd",
    "yellow": "#fff0c6",
    "white": "#ffffff",
}
TEXT_ARTISTS = []


def text(ax, x, y, content, size=17, weight="normal", color=None, ha="center", **kwargs):
    artist = ax.text(
        x, y, content, transform=ax.transAxes, fontsize=size,
        weight=weight, color=color or COLORS["ink"], ha=ha, va="center",
        linespacing=1.15, zorder=5, **kwargs,
    )
    TEXT_ARTISTS.append((ax, artist))
    return artist


def box(ax, x, y, width, height, content, fill="white", size=17, edge=None, weight="normal"):
    patch = FancyBboxPatch(
        (x, y), width, height, transform=ax.transAxes,
        boxstyle="round,pad=0.003,rounding_size=0.009",
        facecolor=COLORS.get(fill, fill), edgecolor=edge or COLORS["ink"],
        linewidth=1.0, zorder=3 if content else 1,
    )
    ax.add_patch(patch)
    if content:
        text(ax, x + width / 2, y + height / 2, content, size=size, weight=weight)
    return patch


def arrow(ax, points, color=None, width=1.3, dashed=False, head=True):
    """Route an orthogonal connection, placing its arrowhead on the last segment."""
    color = color or COLORS["ink"]
    style = (0, (3, 2)) if dashed else "solid"
    if len(points) > 2:
        coordinates = np.asarray(points[:-1])
        ax.plot(
            coordinates[:, 0], coordinates[:, 1], transform=ax.transAxes,
            color=color, linewidth=width, linestyle=style,
            solid_capstyle="round", zorder=2,
        )
    connection = FancyArrowPatch(
        points[-2], points[-1], transform=ax.transAxes,
        arrowstyle="-|>" if head else "-", mutation_scale=12,
        linewidth=width, linestyle=style, color=color,
        shrinkA=0, shrinkB=1.5, zorder=2,
    )
    ax.add_patch(connection)


def operation(ax, x, y, symbol, radius=0.026, size=18):
    # Compensation for the non-square panel axes keeps operation nodes circular.
    position = ax.get_position()
    aspect = position.width * ax.figure.get_figwidth() / (position.height * ax.figure.get_figheight())
    circle = matplotlib.patches.Ellipse(
        (x, y), radius * 2, radius * 2 * aspect,
        transform=ax.transAxes, facecolor="white", edgecolor=COLORS["ink"],
        linewidth=1.1, zorder=3,
    )
    ax.add_patch(circle)
    text(ax, x, y, symbol, size=size)


def panel(fig, bounds, title, fill):
    ax = fig.add_axes(bounds)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    border = FancyBboxPatch(
        (0.006, 0.007), 0.988, 0.986, transform=ax.transAxes,
        boxstyle="round,pad=0.003,rounding_size=0.012",
        facecolor=fill, edgecolor=COLORS["border"],
        linewidth=1.5, linestyle=(0, (5, 3)), zorder=0,
    )
    ax.add_patch(border)
    text(ax, 0.5, 0.949, title, size=22, weight="bold", color=COLORS["navy"])
    return ax


def inset(ax, bounds):
    parent = ax.get_position()
    x, y, width, height = bounds
    child = ax.figure.add_axes([
        parent.x0 + x * parent.width, parent.y0 + y * parent.height,
        width * parent.width, height * parent.height,
    ])
    child.set_facecolor("none")
    return child


def overall(ax):
    box(ax, 0.018, 0.255, 0.081, 0.48, "", fill="#f8fafc", edge="#aab6c1")
    text(ax, 0.0585, 0.681, "EEG trial", size=19)
    waveform = inset(ax, (0.030, 0.365, 0.057, 0.22))
    time = np.linspace(0, 1, 180)
    for index in range(5):
        wave = 0.10 * np.sin(2 * np.pi * (7 + index) * time)
        wave += 0.035 * np.sin(2 * np.pi * 29 * time + index)
        waveform.plot(time, wave + index * 0.32, color="#666e78", linewidth=0.65)
    waveform.axis("off")
    text(ax, 0.0585, 0.302, "[B, 22, 1000]", size=15)

    box(ax, 0.140, 0.542, 0.412, 0.315, "", fill="#fff5dd", edge="#d6c8a4")
    box(ax, 0.140, 0.125, 0.412, 0.315, "", fill="#f0f6e8", edge="#b5c6a1")
    text(ax, 0.346, 0.811, "Frequency–time branch", size=18, weight="bold", color="#a57836")
    text(ax, 0.346, 0.392, "Spatial branch", size=18, weight="bold", color="#5f8551")

    boxes = [
        (0.152, 0.605, 0.125, 0.145, "Sinc filter bank\n+ log-power (b)", "orange"),
        (0.296, 0.605, 0.104, 0.145, "DCT-PSCA (c)", "yellow"),
        (0.419, 0.605, 0.120, 0.145, "Temporal encoder (d)", "blue"),
        (0.152, 0.190, 0.125, 0.145, "Shared electrode\nencoder (e)", "green"),
        (0.296, 0.190, 0.104, 0.145, "Spatial-prior\nGAT (f)", "green"),
        (0.419, 0.190, 0.120, 0.145, "Adaptive ROI +\nregion encoder (g)", "green"),
    ]
    for x, y, width, height, content, fill in boxes:
        box(ax, x, y, width, height, content, fill=fill, size=17)
    arrow(ax, [(0.099, 0.49), (0.119, 0.49), (0.119, 0.6775), (0.152, 0.6775)])
    arrow(ax, [(0.119, 0.49), (0.119, 0.2625), (0.152, 0.2625)])
    text(ax, 0.122, 0.768, "Raw input", size=13)
    text(ax, 0.119, 0.151, "50 Hz notch", size=13)
    for y in (0.6775, 0.2625):
        arrow(ax, [(0.277, y), (0.296, y)])
        arrow(ax, [(0.400, y), (0.419, y)])

    box(ax, 0.592, 0.229, 0.102, 0.486, "Bidirectional\nCross-Attention\n(h)", fill="purple", size=19)
    arrow(ax, [(0.539, 0.6775), (0.592, 0.6775)])
    arrow(ax, [(0.539, 0.2625), (0.592, 0.2625)])
    text(ax, 0.565, 0.785, "$H_T$\n[B, 16, 64]", size=15)
    text(ax, 0.565, 0.159, "$H_S$\n[B, 3, 64]", size=15)
    box(ax, 0.721, 0.603, 0.068, 0.148, "Attention\npooling", fill="blue", size=16)
    box(ax, 0.721, 0.188, 0.068, 0.148, "Attention\npooling", fill="green", size=16)
    arrow(ax, [(0.694, 0.6775), (0.721, 0.6775)])
    arrow(ax, [(0.694, 0.2625), (0.721, 0.2625)])
    text(ax, 0.706, 0.791, "$H'_T$", size=17)
    text(ax, 0.706, 0.148, "$H'_S$", size=17)

    box(ax, 0.820, 0.253, 0.073, 0.448, "Gated\nfeature\nfusion\n(i)", fill="purple", size=18)
    arrow(ax, [(0.789, 0.6775), (0.804, 0.6775), (0.804, 0.57), (0.820, 0.57)])
    arrow(ax, [(0.789, 0.2625), (0.804, 0.2625), (0.804, 0.38), (0.820, 0.38)])
    text(ax, 0.803, 0.806, "$h_T$ [B, 64]", size=14)
    text(ax, 0.803, 0.150, "$h_S$ [B, 64]", size=14)
    box(ax, 0.918, 0.318, 0.065, 0.313, "Classifier\n64 → 64 → 4", fill="yellow", size=17)
    arrow(ax, [(0.893, 0.475), (0.918, 0.475)])
    text(ax, 0.950, 0.752, "Output", size=18)
    text(ax, 0.950, 0.209, "4-class logits\n[B, 4]", size=17, weight="bold")


def sinc_and_power(ax, model):
    text(ax, 0.5, 0.865, "$X_F$ [B, 22, 1000 + 2c]", size=19)
    box(ax, 0.08, 0.690, 0.84, 0.110,
        "5 constrained, learnable Sinc FIRs\n251 taps · Hamming window · shared electrodes", fill="orange", size=17)
    arrow(ax, [(0.5, 0.836), (0.5, 0.802)])
    text(ax, 0.5, 0.635, "[B, 5, 22, 1000 + 2c]", size=16)

    response_ax = inset(ax, (0.11, 0.380, 0.40, 0.190))
    kernels = model.frequency_branch.filterbank.filter_kernels().detach().numpy()
    frequencies = np.fft.rfftfreq(4096, d=1 / 250)
    response = np.abs(np.fft.rfft(kernels, n=4096, axis=-1))
    palette = ["#7c92b5", "#8db5a2", "#c8b45e", "#d39875", "#ae8bb8"]
    for index, color in enumerate(palette):
        response_ax.plot(frequencies, response[index], color=color, linewidth=1.8)
    response_ax.set_xlim(0, 48)
    response_ax.set_ylim(0, 1.15)
    response_ax.set_xticks([4, 13, 30, 40])
    response_ax.set_yticks([])
    response_ax.tick_params(axis="x", labelsize=12, length=2, pad=2)
    response_ax.spines[["top", "left", "right"]].set_visible(False)
    response_ax.spines["bottom"].set_color("#a0a4a8")
    text(ax, 0.30, 0.315, "Initial filter responses (Hz)", size=14)

    box(ax, 0.58, 0.463, 0.35, 0.114, "Center crop\n[B, 5, 22, 1000]", fill="white", size=17)
    arrow(ax, [(0.76, 0.609), (0.76, 0.579)])
    box(ax, 0.58, 0.265, 0.35, 0.122, "Sliding log-power\nL = 250, stride = 50", fill="yellow", size=17)
    arrow(ax, [(0.755, 0.460), (0.755, 0.390)])
    text(ax, 0.29, 0.235, r"$\log(\operatorname{mean}(x^2)+\epsilon)$", size=18)
    box(ax, 0.08, 0.078, 0.84, 0.106,
        "[B, 5, 22, 16] → band-major flatten → [B, 110, 16]", fill="blue", size=16)
    arrow(ax, [(0.755, 0.263), (0.755, 0.219), (0.5, 0.219), (0.5, 0.186)])
    text(ax, 0.5, 0.028, "Shared edges initially [4, 8, 13, 20, 30, 40] Hz", size=14)


def dct_psca(ax):
    box(ax, 0.065, 0.766, 0.445, 0.103, "$U$ [B, 110, 16]", fill="blue", size=19)
    wave_ax = inset(ax, (0.61, 0.688, 0.32, 0.196))
    positions = np.arange(16) + 0.5
    for index, (order, count) in enumerate(((1, 37), (2, 37), (3, 36))):
        signal = np.cos(np.pi * order * positions / 16) * 0.25 + (2 - index)
        wave_ax.plot(positions, signal, color=["#7d9ebd", "#88aa70", "#ca986a"][index], linewidth=1.8)
        wave_ax.text(0, 2 - index + 0.4, f"k={order} · {count} channels", fontsize=12, color=COLORS["ink"])
    wave_ax.set_xlim(0, 16)
    wave_ax.set_ylim(-0.35, 2.8)
    wave_ax.axis("off")
    box(ax, 0.065, 0.570, 0.445, 0.111, "Grouped DCT-II\ninterleaved orders 1, 2, 3", fill="yellow", size=17)
    arrow(ax, [(0.2875, 0.764), (0.2875, 0.683)])
    text(ax, 0.2875, 0.517, "$z$ [B, 110]", size=17)
    box(ax, 0.065, 0.348, 0.445, 0.112, "FC 110 → 27* → ReLU\nFC 27 → 110 → Sigmoid", fill="purple", size=17)
    arrow(ax, [(0.2875, 0.490), (0.2875, 0.462)])
    box(ax, 0.065, 0.170, 0.445, 0.100, "$A$ [B, 110] → [B, 110, 1]", fill="green", size=16)
    arrow(ax, [(0.2875, 0.346), (0.2875, 0.272)])
    operation(ax, 0.79, 0.405, "×", radius=0.029, size=23)
    operation(ax, 0.79, 0.220, "+", radius=0.029, size=23)
    arrow(ax, [
        (0.510, 0.818), (0.550, 0.818), (0.550, 0.906),
        (0.965, 0.906), (0.965, 0.460), (0.790, 0.460), (0.790, 0.445),
    ])
    arrow(ax, [(0.510, 0.220), (0.598, 0.220), (0.598, 0.405), (0.759, 0.405)])
    arrow(ax, [(0.598, 0.220), (0.759, 0.220)])
    arrow(ax, [(0.790, 0.365), (0.790, 0.259)])
    text(ax, 0.89, 0.51, "Broadcast\nover time", size=14)
    box(ax, 0.56, 0.057, 0.38, 0.095, "$U_{PSCA}=U\odot A+A$\n[B, 110, 16]", fill="blue", size=17)
    arrow(ax, [(0.790, 0.181), (0.790, 0.154)])
    text(ax, 0.28, 0.071, "One DCT coefficient\nper joint channel", size=15)


def temporal_encoder(ax):
    text(ax, 0.5, 0.866, "$U_{PSCA}$ [B, 110, 16]", size=19)
    box(ax, 0.10, 0.703, 0.80, 0.105, "Transpose → Linear 110 → 64", fill="blue", size=18)
    arrow(ax, [(0.5, 0.838), (0.5, 0.811)])
    operation(ax, 0.5, 0.613, "+", radius=0.026, size=23)
    arrow(ax, [(0.5, 0.700), (0.5, 0.649)])
    box(ax, 0.672, 0.565, 0.246, 0.095, "Learnable\nposition encoding", fill="yellow", size=15)
    arrow(ax, [(0.670, 0.613), (0.528, 0.613)])
    text(ax, 0.285, 0.612, "[B, 16, 64]", size=17)
    box(ax, 0.08, 0.117, 0.84, 0.411, "", fill="#edf4fa", edge="#a3b8ca")
    text(ax, 0.5, 0.487, "Transformer encoder × 2", size=18, weight="bold")
    arrow(ax, [(0.5, 0.577), (0.5, 0.529)])
    rows = [
        (0.390, "4-head MHSA", "blue"),
        (0.310, "Residual + LayerNorm", "white"),
        (0.230, "FFN 64 → 128 → 64 · GELU", "yellow"),
        (0.150, "Residual + LayerNorm", "white"),
    ]
    for y, title, color in rows:
        box(ax, 0.19, y, 0.62, 0.060, title, fill=color, size=16)
    for y in (0.390, 0.310, 0.230):
        arrow(ax, [(0.5, y), (0.5, y - 0.020)])
    arrow(ax, [(0.5, 0.463), (0.131, 0.463), (0.131, 0.340), (0.19, 0.340)])
    arrow(ax, [(0.5, 0.300), (0.866, 0.300), (0.866, 0.180), (0.81, 0.180)])
    arrow(ax, [(0.5, 0.116), (0.5, 0.087)])
    text(ax, 0.5, 0.055, "$H_T$ [B, 16, 64]", size=20, weight="bold")


def electrode_encoder(ax):
    text(ax, 0.5, 0.866, "$X_S$ [B, 22, 1000]", size=19)
    box(ax, 0.09, 0.706, 0.82, 0.106, "Reshape → [B × 22, 1, 1000]", fill="white", size=18)
    arrow(ax, [(0.5, 0.838), (0.5, 0.815)])
    box(ax, 0.09, 0.524, 0.82, 0.115,
        "Conv1d 1 → 32 · k=25, s=5, p=12\nBatchNorm + GELU", fill="green", size=17)
    arrow(ax, [(0.5, 0.703), (0.5, 0.642)])
    text(ax, 0.5, 0.474, "[B × 22, 32, 200]", size=16)
    box(ax, 0.09, 0.310, 0.82, 0.115,
        "Conv1d 32 → 64 · k=15, s=2, p=7\nBatchNorm + GELU", fill="green", size=17)
    arrow(ax, [(0.5, 0.449), (0.5, 0.428)])
    text(ax, 0.5, 0.261, "[B × 22, 64, 100]", size=16)
    box(ax, 0.09, 0.111, 0.82, 0.104, "AdaptiveAvgPool1d(1) → reshape", fill="yellow", size=18)
    arrow(ax, [(0.5, 0.236), (0.5, 0.218)])
    text(ax, 0.5, 0.059, "$H_E$ [B, 22, 64]", size=20, weight="bold")
    text(ax, 0.5, 0.023, "Shared temporal convolution weights across all electrodes", size=13)


def spatial_gat(ax, model):
    graph_ax = inset(ax, (0.065, 0.576, 0.38, 0.303))
    coordinates = model.electrode_coordinates.detach().numpy()
    adjacency = model.gat.adjacency.detach().numpy()
    xy = coordinates[:, :2]
    xy = xy - xy.mean(axis=0)
    xy = xy / np.abs(xy).max() * 0.84
    graph_ax.add_patch(Circle((0, 0), 1, fill=False, edgecolor="#b4b9ba", linewidth=1.0))
    for source in range(22):
        for target in range(source + 1, 22):
            if adjacency[source, target]:
                graph_ax.plot(xy[[source, target], 0], xy[[source, target], 1],
                              color="#b0c0a4", linewidth=0.8, zorder=1)
    graph_ax.scatter(xy[:, 0], xy[:, 1], s=38, color="#a4c286", edgecolors="#4b6240", linewidths=0.7, zorder=2)
    graph_ax.set_aspect("equal")
    graph_ax.set_xlim(-1.08, 1.08)
    graph_ax.set_ylim(-1.08, 1.08)
    graph_ax.axis("off")
    mask_ax = inset(ax, (0.60, 0.613, 0.29, 0.236))
    mask_ax.pcolormesh(
        adjacency.astype(int),
        cmap=matplotlib.colors.ListedColormap(["#f8f9f8", "#8da97b"]),
        shading="flat", rasterized=False,
    )
    mask_ax.set_xlim(0, len(adjacency))
    mask_ax.set_ylim(len(adjacency), 0)
    mask_ax.set_aspect("equal")
    mask_ax.set_xticks([])
    mask_ax.set_yticks([])
    for spine in mask_ax.spines.values():
        spine.set_color("#8c9786")
    arrow(ax, [(0.45, 0.735), (0.566, 0.735)])
    arrow(ax, [(0.89, 0.735), (0.956, 0.735), (0.956, 0.282), (0.877, 0.282)])
    text(ax, 0.25, 0.544, "Template electrode graph", size=15)
    text(ax, 0.75, 0.565, "KNN attention mask", size=15)
    text(ax, 0.5, 0.493, "K=4 neighbors + self-loops · symmetric adjacency", size=15)
    text(ax, 0.5, 0.441, "$H_E$ [B, 22, 64]", size=18)
    box(ax, 0.075, 0.126, 0.85, 0.269, "", fill="#eef5e9", edge="#a9b99b")
    text(ax, 0.5, 0.355, "GAT layer × 2 · 4 heads × 16 features", size=18, weight="bold")
    box(ax, 0.125, 0.251, 0.75, 0.063, "Feature-dependent scores + KNN mask", fill="green", size=16)
    box(ax, 0.125, 0.154, 0.75, 0.063, "Neighbor softmax → weighted aggregation", fill="white", size=16)
    arrow(ax, [(0.5, 0.413), (0.5, 0.397)])
    arrow(ax, [(0.5, 0.249), (0.5, 0.220)])
    text(ax, 0.5, 0.090, "Head concat → GELU → Dropout → Residual + LN", size=15)
    text(ax, 0.5, 0.037, "$H_G$ [B, 22, 64]", size=20, weight="bold")


def adaptive_roi(ax):
    box(ax, 0.055, 0.767, 0.415, 0.108, "$H_G$ [B, 22, 64]\nGraph-encoded electrodes", fill="green", size=16)
    box(ax, 0.535, 0.767, 0.410, 0.108, "Gaussian spatial prior $P$\nC3 / Cz / C4 anchors", fill="yellow", size=16)
    box(ax, 0.055, 0.560, 0.89, 0.126,
        "$s_{cr}=(W_s h_c)^T q_r/\sqrt{64}$\n$+\lambda\log(P_{cr}+\epsilon)$", fill="purple", size=19)
    arrow(ax, [(0.2625, 0.765), (0.2625, 0.689)])
    arrow(ax, [(0.740, 0.765), (0.740, 0.689)])
    box(ax, 0.08, 0.379, 0.84, 0.107,
        "Softmax over 3 regions → $S$ [B, 22, 3]", fill="green", size=18)
    arrow(ax, [(0.5, 0.557), (0.5, 0.489)])
    box(ax, 0.08, 0.195, 0.84, 0.140,
        r"$Z_r=\frac{\sum_c S_{cr}h_c}{\sum_c S_{cr}+\epsilon}$     [B, 3, 64]", fill="white", size=19)
    arrow(ax, [(0.5, 0.376), (0.5, 0.338)])
    box(ax, 0.08, 0.070, 0.84, 0.100,
        "Region Transformer × 2\n4 heads · FFN 64 → 128 → 64", fill="green", size=17)
    arrow(ax, [(0.5, 0.192), (0.5, 0.173)])
    text(ax, 0.5, 0.028, "$H_S$ [B, 3, 64]", size=20, weight="bold")


def cross_attention(ax):
    text(ax, 0.5, 0.889, "Independent parameters · parallel directions", size=15)
    box(ax, 0.06, 0.766, 0.40, 0.085, "$H_T$ [B, 16, 64]", fill="blue", size=17)
    box(ax, 0.54, 0.766, 0.40, 0.085, "$H_S$ [B, 3, 64]", fill="green", size=17)
    box(ax, 0.06, 0.555, 0.40, 0.125, "4-head cross-MHA\n$Q=H_T,\ K,V=H_S$", fill="blue", size=17)
    box(ax, 0.54, 0.555, 0.40, 0.125, "4-head cross-MHA\n$Q=H_S,\ K,V=H_T$", fill="green", size=17)
    arrow(ax, [(0.26, 0.764), (0.26, 0.683)])
    arrow(ax, [(0.74, 0.764), (0.74, 0.683)])
    arrow(ax, [(0.46, 0.812), (0.506, 0.812), (0.506, 0.713), (0.84, 0.713), (0.84, 0.683)], dashed=True)
    arrow(ax, [(0.54, 0.790), (0.485, 0.790), (0.485, 0.700), (0.16, 0.700), (0.16, 0.683)], dashed=True)
    for x, fill in ((0.06, "blue"), (0.54, "green")):
        box(ax, x, 0.417, 0.40, 0.080, "Residual + LayerNorm", fill="white", size=15)
        box(ax, x, 0.266, 0.40, 0.091, "FFN 64 → 128 → 64\nGELU + Dropout", fill=fill, size=16)
        box(ax, x, 0.122, 0.40, 0.080, "Residual + LayerNorm", fill="white", size=15)
        middle = x + 0.20
        arrow(ax, [(middle, 0.552), (middle, 0.500)])
        arrow(ax, [(middle, 0.414), (middle, 0.360)])
        arrow(ax, [(middle, 0.263), (middle, 0.205)])
    arrow(ax, [(0.06, 0.806), (0.020, 0.806), (0.020, 0.457), (0.060, 0.457)])
    arrow(ax, [(0.94, 0.806), (0.980, 0.806), (0.980, 0.457), (0.940, 0.457)])
    arrow(ax, [(0.460, 0.457), (0.485, 0.457), (0.485, 0.162), (0.460, 0.162)])
    arrow(ax, [(0.540, 0.457), (0.515, 0.457), (0.515, 0.162), (0.540, 0.162)])
    text(ax, 0.26, 0.061, "$H'_T$ [B, 16, 64]", size=19, weight="bold")
    text(ax, 0.74, 0.061, "$H'_S$ [B, 3, 64]", size=19, weight="bold")


def fusion_classifier(ax):
    text(ax, 0.5, 0.884, "$a=\operatorname{softmax}(H'q/\sqrt{64}),\quad h=\sum_l a_l H'_l$", size=18)
    box(ax, 0.06, 0.735, 0.40, 0.090, "Temporal tokens\n[B, 16, 64]", fill="blue", size=17)
    box(ax, 0.54, 0.735, 0.40, 0.090, "Region tokens\n[B, 3, 64]", fill="green", size=17)
    box(ax, 0.06, 0.574, 0.40, 0.099, "Learnable-query\nattention pooling", fill="blue", size=17)
    box(ax, 0.54, 0.574, 0.40, 0.099, "Learnable-query\nattention pooling", fill="green", size=17)
    arrow(ax, [(0.26, 0.732), (0.26, 0.676)])
    arrow(ax, [(0.74, 0.732), (0.74, 0.676)])
    text(ax, 0.26, 0.531, "$h_T$ [B, 64]", size=17)
    text(ax, 0.74, 0.531, "$h_S$ [B, 64]", size=17)
    box(ax, 0.17, 0.373, 0.66, 0.100,
        "$g=\sigma(W_g[h_T;h_S]+b_g)$\n[B, 64]", fill="purple", size=19)
    arrow(ax, [(0.26, 0.503), (0.26, 0.476)])
    arrow(ax, [(0.74, 0.503), (0.74, 0.476)])
    box(ax, 0.06, 0.199, 0.88, 0.119,
        "$\operatorname{GELU}(\operatorname{Linear}([g\odot h_T;(1-g)\odot h_S]))$\nLayerNorm + Dropout → [B, 64]", fill="purple", size=17)
    arrow(ax, [(0.5, 0.370), (0.5, 0.321)])
    box(ax, 0.06, 0.049, 0.88, 0.100,
        "FC 64 → 64 → GELU → Dropout\nFC 64 → 4 → logits [B, 4]", fill="yellow", size=18)
    arrow(ax, [(0.5, 0.196), (0.5, 0.152)])


def verify_text_bounds(fig):
    """Catch labels that escape their panels before delivering the figure."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    failures = []
    for ax, artist in TEXT_ARTISTS:
        bounds = artist.get_window_extent(renderer=renderer)
        panel_bounds = ax.get_window_extent(renderer=renderer)
        if (bounds.x0 < panel_bounds.x0 or bounds.x1 > panel_bounds.x1
                or bounds.y0 < panel_bounds.y0 or bounds.y1 > panel_bounds.y1):
            failures.append(artist.get_text())
    if failures:
        raise RuntimeError(f"Text exceeds panel bounds: {failures}")


def main():
    matplotlib.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
        "savefig.facecolor": "white",
    })
    torch.set_num_threads(2)
    model = DualBranchMINet(load_config()).eval()
    figure = plt.figure(figsize=(32, 20), dpi=110, facecolor="white")
    upper = panel(figure, [0.011, 0.727, 0.978, 0.261],
                  "(a) Overall architecture of DualBranchMINet", "#fdf4ee")
    overall(upper)
    left = 0.011
    width = 0.2385
    gap = 0.008
    top_row_y, bottom_row_y, height = 0.380, 0.034, 0.333
    definitions = [
        ("(b) Learnable filter bank & power", "#fff6e0", lambda ax: sinc_and_power(ax, model)),
        ("(c) DCT-enhanced PSCA", "#f6f1fb", dct_psca),
        ("(d) Temporal token encoder", "#f1f6fb", temporal_encoder),
        ("(e) Shared electrode encoder", "#f3f7ec", electrode_encoder),
        ("(f) Spatial-prior graph attention", "#f1f6ec", lambda ax: spatial_gat(ax, model)),
        ("(g) Adaptive ROI & region encoder", "#f2f7ed", adaptive_roi),
        ("(h) Bidirectional Cross-Attention", "#f3eff9", cross_attention),
        ("(i) Pooling, fusion & classification", "#eee7f7", fusion_classifier),
    ]
    for index, (title, color, draw) in enumerate(definitions):
        column = index % 4
        y = top_row_y if index < 4 else bottom_row_y
        ax = panel(figure, [left + column * (width + gap), y, width, height], title, color)
        draw(ax)
    figure.text(
        0.012, 0.013,
        "Default BCI2a configuration  ·  B = batch size  ·  c = context samples per side (default 0)  ·  "
        "* PSCA hidden width 27 is provisional  ·  Dropout = 0.3",
        fontsize=15, fontfamily="Times New Roman", color="#59616a", va="center",
    )
    verify_text_bounds(figure)
    destination = Path(__file__).resolve().parent
    for suffix in ("svg", "png"):
        filename = destination / f"DualBranchMINet_architecture_v1.{suffix}"
        figure.savefig(filename, dpi=220, metadata={"Creator": "DualBranchMINet architecture figure"})
        print(filename)
    plt.close(figure)


if __name__ == "__main__":
    main()
