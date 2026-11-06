"""Evaluación en la partición de prueba y generación de tablas.

Protocolo: vista 64 de cada objeto de prueba como fuente y `n_targets` vistas
objetivo fijas por objeto. Se evalúa el checkpoint best.pt de cada semilla,
elegido con la partición de validación; la partición de prueba no interviene
en ninguna decisión de entrenamiento.

Salidas en --out:
    por_par.csv.gz         métricas de cada par, run y semilla
    resultados.csv         media ± desviación entre semillas, parámetros, costo
    angulos.csv            PSNR por intervalo de ángulo relativo
    wilcoxon.csv           pruebas de rangos con signo emparejadas (Holm)
    tabla_principal.tex, tabla_angulos.tex, tabla_wilcoxon.tex

Uso:
    python -m src.evaluate --configs configs/A_*.yaml configs/B_*.yaml configs/C_*.yaml \
        --seeds 0 1 --data data/processed --work /content/drive/MyDrive/clifford_nvs --out results
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from .data import iterate_pairs, load_split, make_eval_pairs
from .metrics import LPIPS, psnr, ssim, throughput
from .models import build_model, count_params
from .utils import load_config, read_json, resolve_device, run_dir, write_json, write_text

ANGLE_BINS = [0, 30, 60, 90, 120, 150, 180]
BASELINE = {"name": "copia_fuente", "label": "Copiar fuente"}


@torch.no_grad()
def score_pairs(split, pairs, device, lpips_fn, model=None, batch=256, amp=True):
    """Métricas por par. Sin modelo, la predicción es la propia vista fuente."""
    out = {"psnr": [], "ssim": [], "lpips": []}
    if model is not None:
        model.eval()
    for src, tgt, T, _ in iterate_pairs(split, pairs, batch, device):
        if model is None:
            pred = src
        else:
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device == "cuda"):
                pred = model(src, T)
        pred = pred.float().clamp(0, 1)
        out["psnr"].append(psnr(pred, tgt).cpu())
        out["ssim"].append(ssim(pred, tgt).cpu())
        out["lpips"].append(lpips_fn(pred, tgt).cpu() if lpips_fn else torch.full((len(pred),), float("nan")))
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def load_best(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["cfg"]["model"]).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), ck


def display_label(run, label):
    """'A_conv_matrix' -> 'A' (variantes de la Tabla I); la línea base conserva su nombre."""
    return short_label(run, label)


def short_label(run, label):
    head = str(run).split("_")[0]
    return head if len(head) == 1 and head.isalpha() else label


def _fmt(mean, std, digits):
    return f"{mean:.{digits}f}" if np.isnan(std) else f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def _tex_escape(text):
    return str(text).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def write_main_table(df, path):
    rows = []
    for _, r in df.iterrows():
        params = "—" if np.isnan(r.params_M) else f"{r.params_M:.2f}"
        lp = "—" if np.isnan(r.LPIPS_mean) else _fmt(r.LPIPS_mean, r.LPIPS_std, 3)
        rows.append(f"{_tex_escape(display_label(r.run, r.label))} & {params} & {_fmt(r.PSNR_mean, r.PSNR_std, 2)} & "
                    f"{_fmt(r.SSIM_mean, r.SSIM_std, 3)} & {lp} \\\\")
    tex = "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{Síntesis de nuevas vistas en SRN Cars ($64\times64$). Modelos según la Tabla~\ref{tab:params}; media $\pm$ desviación estándar entre semillas y parámetros en millones.}",
        r"\label{tab:principal}", r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{lcccc}", r"\toprule",
        r"Modelo & Parám. & PSNR (dB) $\uparrow$ & SSIM $\uparrow$ & LPIPS $\downarrow$ \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write_text(tex + "\n", path)


def write_angle_table(ang, path):
    bins = list(ang.columns)
    header = " & ".join(_tex_escape(b) for b in bins)
    rows = [f"{_tex_escape(label)} & " + " & ".join("—" if np.isnan(v) else f"{v:.2f}" for v in ang.loc[label]) + r" \\"
            for label in ang.index]
    tex = "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{PSNR ({\upshape dB}) según el ángulo geodésico entre la cámara fuente y la objetivo, por intervalos de $30^\circ$.}",
        r"\label{tab:angulos}", r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{l" + "c" * len(bins) + "}", r"\toprule",
        f"Modelo & {header} \\\\", r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write_text(tex + "\n", path)


def _fmt_p(p):
    """Valor p en notación científica de LaTeX; por debajo de la precisión doble se acota."""
    if p < 1e-300:
        return r"$<10^{-300}$"
    if p >= 1e-3:
        return f"{p:.3f}"
    mant, exp = f"{p:.1e}".split("e")
    return rf"${mant}\times10^{{{int(exp)}}}$"


def write_wilcoxon_table(wx, path):
    rows = [f"{_tex_escape(r.a_short)} vs. {_tex_escape(r.b_short)} & {r.metric} & ${r.median_diff:+.3f}$ & "
            f"{_fmt_p(r.p_holm)} \\\\" for _, r in wx.iterrows()]
    tex = "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{Prueba de Wilcoxon emparejada por par de vistas sobre la media entre semillas; $p$ corregido por Holm. $\Delta$ es la diferencia mediana del primer modelo respecto del segundo.}",
        r"\label{tab:wilcoxon}", r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{llcc}", r"\toprule",
        r"Comparación & Métrica & $\Delta$ & $p$ \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write_text(tex + "\n", path)


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser(description="Evalúa los runs en la partición de prueba.")
    ap.add_argument("--configs", nargs="+", required=True, help="El primero es la referencia")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--n_targets", type=int, default=16)
    ap.add_argument("--n_objects", type=int, default=None, help="Por defecto, todos los objetos de prueba")
    ap.add_argument("--no_lpips", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = resolve_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    cfgs = [load_config(p) for p in args.configs]
    res = cfgs[0]["data"]["res"]

    test = load_split(os.path.join(args.data, f"cars_test_{res}.npz"), "cpu")
    pairs = make_eval_pairs(test, args.n_objects, args.n_targets, src_view=64, seed=0)
    print(f"Pares de prueba: {len(pairs['src'])} ({len(np.unique(pairs['obj']))} objetos)")
    lpips_fn = None if args.no_lpips else LPIPS("vgg", device)

    base_cols = {"pair": np.arange(len(pairs["src"])), "obj": pairs["obj"].numpy(),
                 "src_idx": pairs["src"].numpy(), "tgt_idx": pairs["tgt"].numpy(),
                 "angle": pairs["angle"].numpy()}
    frames, meta = [], []

    m = score_pairs(test, pairs, device, lpips_fn)
    frames.append(pd.DataFrame({**base_cols, **m, "run": BASELINE["name"], "label": BASELINE["label"], "seed": 0}))
    meta.append({"run": BASELINE["name"], "label": BASELINE["label"], "params_M": np.nan,
                 "img_s": np.nan, "gpu_hours": 0.0, "block": None, "pose": None})

    for cfg in cfgs:
        gpu_h, speed, params = [], np.nan, np.nan
        for seed in args.seeds:
            d = run_dir(args.work, cfg["name"], seed)
            path = os.path.join(d, "best.pt")
            if not os.path.exists(path):
                print(f"Falta {path}; se omite")
                continue
            model, ck = load_best(path, device)
            params = count_params(model) / 1e6
            if np.isnan(speed) and device == "cuda":
                speed = throughput(model, device, res=res)
            m = score_pairs(test, pairs, device, lpips_fn, model)
            frames.append(pd.DataFrame({**base_cols, **m, "run": cfg["name"], "label": cfg["label"], "seed": seed}))
            summ = os.path.join(d, "summary.json")
            if os.path.exists(summ):
                gpu_h.append(read_json(summ)["gpu_hours"])
            else:
                gpu_h.append(torch.load(os.path.join(d, "last.pt"), map_location="cpu",
                                        weights_only=False)["train_seconds"] / 3600)
            print(f"{cfg['name']} semilla {seed}: PSNR {m['psnr'].mean():.2f} dB  "
                  f"(best.pt del paso {ck['step']})")
        meta.append({"run": cfg["name"], "label": cfg["label"], "params_M": params, "img_s": speed,
                     "gpu_hours": float(np.sum(gpu_h)) if gpu_h else np.nan,
                     "block": cfg["model"]["block"], "pose": cfg["model"]["pose"]})

    per_pair = pd.concat(frames, ignore_index=True)
    per_pair.to_csv(os.path.join(args.out, "por_par.csv.gz"), index=False)

    # Media por semilla y luego media ± desviación entre semillas.
    seed_means = per_pair.groupby(["run", "seed"])[["psnr", "ssim", "lpips"]].mean()
    agg = seed_means.groupby("run").agg(["mean", "std"])
    agg.columns = [f"{m.upper()}_{s}" for m, s in agg.columns]
    agg["n_seeds"] = seed_means.groupby("run").size()
    meta_df = pd.DataFrame(meta).set_index("run")
    table = meta_df.join(agg, how="inner").loc[[r["run"] for r in meta if r["run"] in agg.index]]
    table.to_csv(os.path.join(args.out, "resultados.csv"))
    write_main_table(table.reset_index(), os.path.join(args.out, "tabla_principal.tex"))

    # Análisis por ángulo relativo.
    labels = [f"{a}–{b}°" for a, b in zip(ANGLE_BINS[:-1], ANGLE_BINS[1:])]
    per_pair["bin"] = pd.cut(per_pair["angle"], ANGLE_BINS, labels=labels, include_lowest=True)
    ang = per_pair.groupby(["label", "bin"], observed=False)["psnr"].mean().unstack()
    ang = ang.loc[table["label"]]
    ang.to_csv(os.path.join(args.out, "angulos.csv"))
    ang.index = [display_label(r, l) for r, l in zip(table.index, table["label"])]
    write_angle_table(ang, os.path.join(args.out, "tabla_angulos.tex"))

    # Wilcoxon emparejado sobre la media entre semillas de cada par.
    pair_mean = per_pair.groupby(["run", "pair"])[["psnr", "ssim", "lpips"]].mean()
    ref = cfgs[0]["name"]
    runs = [c["name"] for c in cfgs if c["name"] in agg.index]
    comps = [(r, ref) for r in runs if r != ref]
    for c in cfgs:
        if c["model"]["pose"] == "rotor":
            twin = next((o["name"] for o in cfgs if o["model"]["block"] == c["model"]["block"]
                         and o["model"]["pose"] == "matrix"), None)
            if twin and twin != ref and twin in runs and c["name"] in runs:
                comps.append((c["name"], twin))
    rows = []
    metrics = ["psnr", "ssim"] + ([] if args.no_lpips else ["lpips"])
    for a, b in comps:
        for met in metrics:
            da, db = pair_mean.loc[a][met].values, pair_mean.loc[b][met].values
            diff = da - db
            p = wilcoxon(da, db).pvalue if np.any(diff != 0) else 1.0
            rows.append({"a": meta_df.loc[a, "label"], "b": meta_df.loc[b, "label"],
                         "a_short": short_label(a, meta_df.loc[a, "label"]),
                         "b_short": short_label(b, meta_df.loc[b, "label"]), "metric": met.upper(),
                         "median_diff": float(np.median(diff)), "p": p})
    if rows:
        wx = pd.DataFrame(rows)
        wx["p_holm"] = holm(wx["p"])
        wx.to_csv(os.path.join(args.out, "wilcoxon.csv"), index=False)
        write_wilcoxon_table(wx, os.path.join(args.out, "tabla_wilcoxon.tex"))

    total_h = float(np.nansum(table["gpu_hours"]))
    write_json({"gpu_total_hours": total_h,
                "device": torch.cuda.get_device_name() if device == "cuda" else device,
                "n_test_pairs": int(len(pairs["src"]))}, os.path.join(args.out, "costo.json"))

    cols = ["label", "params_M", "PSNR_mean", "PSNR_std", "SSIM_mean", "LPIPS_mean", "img_s", "gpu_hours", "n_seeds"]
    print(table[cols].round(3).to_string())
    print(f"Horas de GPU totales: {total_h:.2f}")


if __name__ == "__main__":
    main()
