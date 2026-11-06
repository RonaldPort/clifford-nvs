"""Figuras del artículo (PDF vectorial, dimensionadas para IEEEtran).

    fig_curvas.pdf         pérdida L1 de entrenamiento y PSNR de validación
    fig_psnr_angulo.pdf    PSNR de prueba frente al ángulo relativo (IC 95 %)
    fig_cualitativa.pdf    fuente, objetivo y predicción de cada modelo
    fig_error.pdf          mapas de error absoluto por modelo
    fig_peores_casos.pdf   pares con menor PSNR del modelo indicado en --focus
Cada figura se guarda también en PNG para las diapositivas.

Requiere haber ejecutado src.evaluate (lee por_par.csv.gz).

Uso:
    python -m src.figures --configs configs/A_*.yaml configs/B_*.yaml configs/C_*.yaml \
        --seeds 0 1 --data data/processed --work /content/drive/MyDrive/clifford_nvs --out results
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .data import load_split, to_float_images
from .evaluate import load_best
from .geometry import relative_pose
from .utils import load_config, read_json, resolve_device, run_dir

COL_W, FULL_W = 3.5, 7.16  # anchos de columna y de página en IEEEtran (pulgadas)
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, out, name, **kw):
    """PDF vectorial para el artículo y PNG para las diapositivas."""
    fig.savefig(os.path.join(out, f"{name}.pdf"), **kw)
    fig.savefig(os.path.join(out, f"{name}.png"), dpi=220, **kw)
    plt.close(fig)


def plot_curves(cfgs, seeds, work, out):
    fig, (ax_l, ax_v) = plt.subplots(1, 2, figsize=(FULL_W, 2.3))
    for cfg in cfgs:
        tr, va = [], []
        for s in seeds:
            p = os.path.join(run_dir(work, cfg["name"], s), "log.json")
            if os.path.exists(p):
                log = read_json(p)
                tr.append(pd.DataFrame(log["train"]).set_index("step")["l1"])
                va.append(pd.DataFrame(log["val"]).set_index("step")["psnr"])
        for ax, series in ((ax_l, tr), (ax_v, va)):
            if not series:
                continue
            df = pd.concat(series, axis=1)
            line, = ax.plot(df.index, df.mean(axis=1), label=cfg["label"], lw=1.2)
            if df.shape[1] > 1:
                ax.fill_between(df.index, df.min(axis=1), df.max(axis=1), color=line.get_color(),
                                alpha=0.2, lw=0)
    ax_l.set(xlabel="Paso de entrenamiento", ylabel="Pérdida L1 (entrenamiento)")
    ax_v.set(xlabel="Paso de entrenamiento", ylabel="PSNR validación (dB)")
    ax_l.set_yscale("log")
    ax_v.legend(frameon=False)
    fig.tight_layout()
    _save(fig, out, "fig_curvas")


def plot_angle(per_pair, out, bin_width=15):
    edges = np.arange(0, 180 + bin_width, bin_width)
    centers = (edges[:-1] + edges[1:]) / 2
    fig, ax = plt.subplots(figsize=(COL_W, 2.9))
    for label, g in per_pair.groupby("label", sort=False):
        g = g.groupby("pair").agg(psnr=("psnr", "mean"), angle=("angle", "first"))
        idx = np.clip(np.digitize(g["angle"], edges) - 1, 0, len(centers) - 1)
        stats = g.assign(b=idx).groupby("b")["psnr"].agg(["mean", "std", "count"])
        x = centers[stats.index]
        ci = 1.96 * stats["std"] / np.sqrt(stats["count"])
        style = dict(ls="--", color="gray") if label == "Copiar fuente" else {}
        line, = ax.plot(x, stats["mean"], marker="o", ms=2.5, lw=1.1, label=label, **style)
        ax.fill_between(x, stats["mean"] - ci, stats["mean"] + ci, color=line.get_color(), alpha=0.2, lw=0)
    ax.set(xlabel="Ángulo relativo fuente–objetivo (°)", ylabel="PSNR prueba (dB)", xlim=(0, 180))
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    _save(fig, out, "fig_psnr_angulo")


@torch.no_grad()
def _predict(model, test, src_idx, tgt_idx, device):
    src = to_float_images(test["images"][src_idx]).to(device)
    T = relative_pose(test["poses"][src_idx], test["poses"][tgt_idx]).to(device)
    return model(src, T).float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()


def _clear(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _grid(columns, rows, out, name, cmap=None, vmax=None, angles=None):
    fig, axes = plt.subplots(rows, len(columns), figsize=(0.95 * len(columns), 0.95 * rows), squeeze=False)
    im = None
    for j, (title, arr) in enumerate(columns):
        axes[0, j].set_title(title, fontsize=6)
        for i in range(rows):
            im = axes[i, j].imshow(arr[i], cmap=cmap, vmin=0, vmax=vmax) if cmap else axes[i, j].imshow(arr[i])
            _clear(axes[i, j])
            if j == 0 and angles is not None:
                axes[i, j].set_ylabel(f"{angles[i]:.0f}°", fontsize=6)
    if cmap:
        cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
        cb.set_label("Error absoluto por píxel (media RGB)", fontsize=6)
        cb.ax.tick_params(labelsize=5)
    else:
        fig.tight_layout(pad=0.2)
    _save(fig, out, name, bbox_inches="tight")


def plot_samples(cfgs, work, test, per_pair, out, device, n_rows=6):
    first = per_pair[per_pair["run"] == per_pair["run"].iloc[0]].drop_duplicates("pair")
    quantiles = np.linspace(0.05, 0.95, n_rows)
    chosen = [first.iloc[(first["angle"] - first["angle"].quantile(q)).abs().argsort().iloc[0]] for q in quantiles]
    pair_ids = [int(r["pair"]) for r in chosen]
    angles = [float(r["angle"]) for r in chosen]
    src_idx, tgt_idx = _pair_indices(per_pair, pair_ids)

    target = test["images"][tgt_idx].numpy()
    cols = [("Fuente", test["images"][src_idx].numpy()), ("Objetivo", target)]
    errs = []
    for cfg in cfgs:
        path = os.path.join(run_dir(work, cfg["name"], 0), "best.pt")
        if not os.path.exists(path):
            continue
        model, _ = load_best(path, device)
        pred = _predict(model, test, src_idx, tgt_idx, device)
        cols.append((cfg["label"], pred))
        errs.append((cfg["label"], np.abs(pred - target / 255.0).mean(-1)))
    _grid(cols, n_rows, out, "fig_cualitativa", angles=angles)
    if errs:
        vmax = max(e.max() for _, e in errs)
        _grid(errs, n_rows, out, "fig_error", cmap="magma", vmax=vmax, angles=angles)


def plot_worst(cfg, work, test, per_pair, out, device, n=6):
    path = os.path.join(run_dir(work, cfg["name"], 0), "best.pt")
    sub = per_pair[per_pair["run"] == cfg["name"]]
    if not os.path.exists(path) or sub.empty:
        return
    worst = sub.groupby("pair").agg(psnr=("psnr", "mean"), angle=("angle", "first")).nsmallest(n, "psnr")
    src_idx, tgt_idx = _pair_indices(per_pair, worst.index.tolist())
    model, _ = load_best(path, device)
    pred = _predict(model, test, src_idx, tgt_idx, device)
    fig, axes = plt.subplots(3, n, figsize=(FULL_W, 3.6), squeeze=False)
    rows = [("Fuente", test["images"][src_idx].numpy()), ("Objetivo", test["images"][tgt_idx].numpy()),
            (cfg["label"], pred)]
    for i, (name, arr) in enumerate(rows):
        for j in range(n):
            axes[i, j].imshow(arr[j])
            _clear(axes[i, j])
            if j == 0:
                axes[i, j].set_ylabel(name, fontsize=7)
            if i == 0:
                axes[i, j].set_title(f"{worst['angle'].iloc[j]:.0f}°, {worst['psnr'].iloc[j]:.1f} dB", fontsize=6)
    fig.tight_layout(pad=0.3)
    _save(fig, out, "fig_peores_casos")


def _pair_indices(per_pair, pair_ids):
    lookup = per_pair.drop_duplicates("pair").set_index("pair")
    return (torch.tensor(lookup.loc[pair_ids, "src_idx"].values),
            torch.tensor(lookup.loc[pair_ids, "tgt_idx"].values))


def main():
    ap = argparse.ArgumentParser(description="Genera las figuras del artículo.")
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--focus", default=None, help="Config del análisis de peores casos (por defecto, la última)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = resolve_device(args.device)
    cfgs = [load_config(p) for p in args.configs]
    res = cfgs[0]["data"]["res"]
    per_pair = pd.read_csv(os.path.join(args.out, "por_par.csv.gz"))
    test = load_split(os.path.join(args.data, f"cars_test_{res}.npz"), "cpu")

    plot_curves(cfgs, args.seeds, args.work, args.out)
    plot_angle(per_pair, args.out)
    plot_samples(cfgs, args.work, test, per_pair, args.out, device)
    focus = load_config(args.focus) if args.focus else cfgs[-1]
    plot_worst(focus, args.work, test, per_pair, args.out, device)
    print(f"Figuras escritas en {args.out}")


if __name__ == "__main__":
    main()
