#!/usr/bin/env python3
"""Build 3 Frankenstein figures from REAL module npz data. Nice/low-thread."""
from __future__ import annotations
import os, sys
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[v] = "1"

import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

sys.path.insert(0, "/workspace/flybrain/regions/frankenstein")
sys.path.insert(0, "/workspace/flybrain/regions/empn")
sys.path.insert(0, "/workspace/flybrain/compress")
import frankenstein as F

OUT = Path("/workspace/flybrain/figs")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 0
BIN = 420  # heatmap bins


def load_counts():
    sizes = {}
    layers = {}
    signs = {}
    nnz = {}
    for name in ("mouse", "human", "fly_cx"):
        W, sign, layer, ct = F.load_mod(name)
        sizes[name] = int(W.shape[0])
        layers[name] = {str(k): int(v) for k, v in zip(*np.unique(layer.astype(str), return_counts=True))}
        signs[name] = {"exc": int((sign > 0).sum()), "inh": int((sign < 0).sum())}
        nnz[name] = int(W.nnz)
    return sizes, layers, signs, nnz


def bin_signed_density(W: sp.spmatrix, nbin: int):
    """Bin sparse W into nbin x nbin. Return density (frac nonzero) and mean signed weight."""
    W = W.tocoo()
    N = W.shape[0]
    # bin edges
    edges = np.linspace(0, N, nbin + 1)
    # digitize rows/cols into bins
    br = np.clip(np.digitize(W.row, edges) - 1, 0, nbin - 1)
    bc = np.clip(np.digitize(W.col, edges) - 1, 0, nbin - 1)
    dens = np.zeros((nbin, nbin), dtype=np.float64)
    signed = np.zeros((nbin, nbin), dtype=np.float64)
    # accumulate
    for r, c, d in zip(br, bc, W.data):
        dens[r, c] += 1.0
        signed[r, c] += float(d)
    # density = nonzero count / max cells per bin (~ (N/nbin)^2)
    cell = (N / nbin) ** 2
    dens /= max(cell, 1.0)
    # mean signed weight where present; else 0
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_signed = np.where(dens > 0, signed / (dens * cell), 0.0)
    return dens, mean_signed, N


def signed_display(dens, mean_signed):
    """Blue = excitatory mass, red = inhibitory. Magnitude = connection density."""
    # sign of mean weight; magnitude from dens (log-ish)
    mag = np.log1p(dens * 50.0)  # stretch small densities
    mag = mag / (mag.max() + 1e-12)
    signed_mag = np.sign(mean_signed) * mag
    # where dens==0, keep 0
    signed_mag = np.where(dens > 0, signed_mag, 0.0)
    return signed_mag


def fig_matrix(W_frank, W_er, sizes, offsets):
    dens_f, ms_f, N = bin_signed_density(W_frank, BIN)
    dens_e, ms_e, _ = bin_signed_density(W_er, BIN)
    disp_f = signed_display(dens_f, ms_f)
    disp_e = signed_display(dens_e, ms_e)
    vmax = max(np.abs(disp_f).max(), np.abs(disp_e).max(), 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8.2), facecolor="#0d1117")
    titles = [
        f"FRANKENSTEIN  N={N:,}  (real wiring + bridges σ={F.BRIDGE_SIGMA})",
        f"ER CONTROL  N={N:,}  (per-block nnz-matched random)",
    ]
    labels = {
        "mouse": "Mouse cortex",
        "human": "Human cortex",
        "fly_cx": "Fly central complex",
    }
    # block order in graft: mouse, human, fly_cx
    block_names = list(F.BLOCKS)
    for ax, disp, title in zip(axes, (disp_f, disp_e), titles):
        ax.set_facecolor("#0d1117")
        im = ax.imshow(disp, cmap="RdBu_r", norm=norm, origin="upper",
                       interpolation="nearest", aspect="equal")
        # block boundaries in bin coords
        for off in offsets[1:-1]:
            b = off / N * BIN
            ax.axhline(b - 0.5, color="#e6edf3", lw=1.2, alpha=0.85)
            ax.axvline(b - 0.5, color="#e6edf3", lw=1.2, alpha=0.85)
        # label blocks along diagonal
        for i, name in enumerate(block_names):
            a, b = offsets[i], offsets[i + 1]
            mid = (a + b) / 2 / N * BIN
            ax.text(mid, mid, labels[name], color="#f0f6fc", fontsize=11,
                    ha="center", va="center", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#161b22", ec="#58a6ff", alpha=0.85))
        ax.set_title(title, color="#f0f6fc", fontsize=13, pad=10, fontweight="bold")
        ax.set_xlabel("postsynaptic bin →", color="#8b949e", fontsize=11)
        ax.set_ylabel("presynaptic bin →", color="#8b949e", fontsize=11)
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.ax.yaxis.set_tick_params(color="#8b949e")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#8b949e")
        cbar.set_label("blue=excitatory  ·  red=inhibitory\n(magnitude ∝ binned density)",
                       color="#8b949e", fontsize=9)

    # counts strip
    ns = "  |  ".join(f"{labels[n]}: {sizes[n]:,}" for n in block_names)
    fig.suptitle(
        f"Frankenstein connection matrix  ·  blocks [{ns}]  ·  total N={N:,}",
        color="#58a6ff", fontsize=14, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = OUT / "franken_matrix.png"
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path, N, dens_f, dens_e


def fig_diagram(sizes, layers, signs, nnz, bridge_info, N):
    fig, ax = plt.subplots(figsize=(14, 9), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, title, lines, ec="#58a6ff", fc="#161b22"):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
                           linewidth=2, edgecolor=ec, facecolor=fc)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h - 0.35, title, ha="center", va="top",
                color="#f0f6fc", fontsize=12, fontweight="bold")
        for i, line in enumerate(lines):
            ax.text(x + w / 2, y + h - 0.85 - i * 0.38, line, ha="center", va="top",
                    color="#c9d1d9", fontsize=10)

    def arrow(x1, y1, x2, y2, color="#79c0ff", lw=2.2, style="-|>"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                    connectionstyle="arc3,rad=0"))

    # Title
    ax.text(7, 9.6, "How the Frankenstein brain is wired",
            ha="center", color="#58a6ff", fontsize=18, fontweight="bold")
    ax.text(7, 9.15,
            f"Real connectomes grafted as blocks  ·  hub = fly CX  ·  bridges σ={F.BRIDGE_SIGMA}  ·  global ρ={F.RHO}",
            ha="center", color="#8b949e", fontsize=11)

    # Input
    box(0.3, 4.2, 1.8, 1.6, "INPUT",
        ["drive = ∪ L4", "mouse + human", "+ fly L4-proxy"],
        ec="#3fb950", fc="#122117")

    # Mouse
    ml = layers["mouse"]
    box(3.0, 7.0, 3.4, 2.2, f"MOUSE cortex  N={sizes['mouse']:,}",
        [f"MICrONS proofread  ·  nnz={nnz['mouse']:,}",
         f"L4 in={ml.get('L4',0)}   L5 out={ml.get('L5',0)}",
         f"E={signs['mouse']['exc']}  I={signs['mouse']['inh']}",
         f"L2/3={ml.get('L2/3',0)}  L6={ml.get('L6',0)}"],
        ec="#f778ba")

    # Human
    hl = layers["human"]
    box(3.0, 0.6, 3.4, 2.2, f"HUMAN cortex  N={sizes['human']:,}",
        [f"H01 HF edges  ·  nnz={nnz['human']:,}",
         f"L4 in={hl.get('L4',0)}   L5 out={hl.get('L5',0)}",
         f"E={signs['human']['exc']}  I={signs['human']['inh']}",
         f"L2/3={hl.get('L2/3',0)}  L6={hl.get('L6',0)}"],
        ec="#d2a8ff")

    # Fly CX hub
    fl = layers["fly_cx"]
    box(7.6, 3.5, 3.8, 3.2, f"FLY central complex  N={sizes['fly_cx']:,}",
        ["EMPN CX subgraph (all typed CX)",
         "E-PG compass  ·  P-EN  ·  FB",
         f"(layer proxy: L4-like={fl.get('L4',0)}, L5-like={fl.get('L5',0)})",
         f"E={signs['fly_cx']['exc']}  I={signs['fly_cx']['inh']}  nnz={nnz['fly_cx']:,}",
         "HUB — no mouse↔human bridges"],
        ec="#58a6ff")

    # Readout
    box(12.0, 4.2, 1.7, 1.6, "READOUT",
        ["ridge / OLS", "on states", f"N={N:,} total"],
        ec="#f85149", fc="#2a1215")

    # Arrows: input -> blocks
    arrow(2.1, 5.3, 3.0, 7.6, color="#3fb950")
    arrow(2.1, 4.7, 3.0, 2.2, color="#3fb950")
    arrow(2.1, 5.0, 7.6, 5.0, color="#3fb950")

    # Bridges mouse L5 -> fly, fly -> mouse L4
    bi_m = bridge_info.get("mouse", {})
    bi_h = bridge_info.get("human", {})
    arrow(6.4, 7.6, 7.6, 6.2, color="#f778ba", lw=1.8)  # mouse L5 -> fly
    arrow(7.6, 5.8, 6.4, 7.2, color="#f778ba", lw=1.8)  # fly -> mouse L4
    ax.text(6.85, 7.0, f"bridges\nn_out={bi_m.get('n_out','?')}\nn_in={bi_m.get('n_in','?')}\nσ={F.BRIDGE_SIGMA}",
            color="#f778ba", fontsize=8, ha="center")

    arrow(6.4, 2.2, 7.6, 4.0, color="#d2a8ff", lw=1.8)
    arrow(7.6, 4.4, 6.4, 2.6, color="#d2a8ff", lw=1.8)
    ax.text(6.85, 3.0, f"bridges\nn_out={bi_h.get('n_out','?')}\nn_in={bi_h.get('n_in','?')}\nσ={F.BRIDGE_SIGMA}",
            color="#d2a8ff", fontsize=8, ha="center")

    # fly -> readout
    arrow(11.4, 5.0, 12.0, 5.0, color="#f85149", lw=2.5)

    # legend / totals
    ax.text(7, 0.25,
            f"TOTAL NEURONS = mouse {sizes['mouse']:,} + human {sizes['human']:,} + fly CX {sizes['fly_cx']:,}  =  {N:,}"
            f"    ·    block order in W: [mouse | human | fly_CX]    ·    each block ρ=1 before bridges, then global ρ={F.RHO}",
            ha="center", color="#8b949e", fontsize=10)

    path = OUT / "franken_diagram.png"
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


def spectrum(W, label):
    """Full eigvals of dense W; return scaled-to-radius-1 eigenvalues + stats."""
    Wd = W.toarray().astype(np.float64)
    ev = np.linalg.eigvals(Wd)
    r = np.abs(ev).max()
    evs = ev / (r + 1e-12)
    frac = float(np.mean(np.abs(evs) > 0.5))
    mean_abs = float(np.mean(np.abs(evs)))
    return evs, dict(label=label, N=W.shape[0], radius_raw=float(r),
                     frac_gt_0_5=frac, mean_abs=mean_abs, n_gt=int(np.sum(np.abs(evs) > 0.5)))


def fig_eigs(specs):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), facecolor="#0d1117")
    colors = {"FRANKEN": "#58a6ff", "ER": "#f778ba", "FLY_CX": "#3fb950"}
    for ax, (evs, st) in zip(axes, specs):
        ax.set_facecolor("#0d1117")
        lab = st["label"]
        ax.scatter(evs.real, evs.imag, s=3, alpha=0.35, c=colors[lab], linewidths=0)
        # unit circle
        th = np.linspace(0, 2 * np.pi, 400)
        ax.plot(np.cos(th), np.sin(th), color="#e6edf3", lw=1.2, alpha=0.9, label="|λ|=1")
        ax.plot(0.5 * np.cos(th), 0.5 * np.sin(th), color="#f85149", lw=1.4,
                ls="--", alpha=0.95, label="|λ|=0.5")
        pct = 100 * st["frac_gt_0_5"]
        ax.set_title(
            f"{lab}  N={st['N']:,}\n{pct:.2f}% with |λ|>0.5   (n={st['n_gt']:,})",
            color="#f0f6fc", fontsize=12, fontweight="bold",
        )
        ax.set_xlabel("Re(λ)", color="#8b949e")
        ax.set_ylabel("Im(λ)", color="#8b949e")
        ax.tick_params(colors="#8b949e")
        ax.set_aspect("equal")
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.grid(True, color="#21262d", lw=0.6)
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        leg = ax.legend(facecolor="#161b22", edgecolor="#30363d", fontsize=8,
                        labelcolor="#c9d1d9", loc="upper right")

    fig.suptitle(
        "Eigenvalues in the complex plane  (each matrix scaled to spectral radius = 1)\n"
        "Fat mid-spectrum (|λ|>0.5) → more independent echoes → higher noise-memory capacity",
        color="#58a6ff", fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    path = OUT / "franken_eigs.png"
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    print("loading modules + counts…", flush=True)
    sizes, layers, signs, nnz = load_counts()
    print("sizes", sizes, "layers", layers, flush=True)

    print("building FRANKEN (real blocks + bridges) seed", SEED, flush=True)
    blocks_real = []
    for name in F.BLOCKS:
        Wm, sign, layer, ct = F.make_block(name, "real", SEED)
        blocks_real.append((name, Wm, layer))
    W_frank, info = F.graft_blocks(blocks_real, SEED)
    offsets = info["offsets"]
    N = info["N"]
    bridge_info = info["bridge"]
    print("FRANKEN N", N, "nnz", W_frank.nnz, "offsets", offsets, "bridge", bridge_info, flush=True)

    print("building ER control…", flush=True)
    blocks_er = []
    for name in F.BLOCKS:
        Wm, sign, layer, ct = F.make_block(name, "er", SEED)
        blocks_er.append((name, Wm, layer))
    W_er, info_er = F.graft_blocks(blocks_er, SEED + 1000)  # different bridge RNG ok; structure random
    # Actually for fair ER viz, use same seed for bridges pattern randomness but er blocks
    # Rebuild with same seed so bridge layout comparable
    W_er, info_er = F.graft_blocks(blocks_er, SEED)
    print("ER N", W_er.shape[0], "nnz", W_er.nnz, flush=True)

    print("FLY_CX alone…", flush=True)
    W_fly, sign_f, layer_f, ct_f = F.make_block("fly_cx", "real", SEED)
    W_fly = F.scale(W_fly, F.RHO)

    print("fig 1: matrix…", flush=True)
    p1, N1, dens_f, dens_e = fig_matrix(W_frank, W_er, sizes, offsets)
    print("saved", p1, flush=True)

    print("fig 2: diagram…", flush=True)
    p2 = fig_diagram(sizes, layers, signs, nnz, bridge_info, N)
    print("saved", p2, flush=True)

    print("fig 3: eigenvalues (full spectrum, 1 thread)…", flush=True)
    specs = []
    for W, lab in ((W_frank, "FRANKEN"), (W_er, "ER"), (W_fly, "FLY_CX")):
        print("  eig", lab, "N", W.shape[0], flush=True)
        evs, st = spectrum(W, lab)
        print("   ", st, flush=True)
        specs.append((evs, st))
    # order panels FRANKEN | ER | FLY_CX
    p3 = fig_eigs(specs)
    print("saved", p3, flush=True)

    # meta json for parent
    meta = {
        "paths": {"matrix": str(p1), "diagram": str(p2), "eigs": str(p3)},
        "sizes": sizes,
        "layers": layers,
        "signs": signs,
        "nnz": nnz,
        "N_total": N,
        "offsets": offsets,
        "bridge": bridge_info,
        "bridge_sigma": F.BRIDGE_SIGMA,
        "rho": F.RHO,
        "eigs": [s for _, s in specs],
        "franken_nnz": int(W_frank.nnz),
        "er_nnz": int(W_er.nnz),
        "heatmap_bins": BIN,
        "dens_frank_max": float(dens_f.max()),
        "dens_er_max": float(dens_e.max()),
    }
    import json
    (OUT / "franken_figs_meta.json").write_text(json.dumps(meta, indent=2))
    print("DONE", json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
