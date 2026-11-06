"""Análisis complementario de los resultados de prueba y de los registros de entrenamiento.

Calcula, a partir de por_par.csv.gz y de los log.json de cada run:
    variabilidad por semilla y diferencia entre modelos por semilla
    pruebas de Wilcoxon sobre el PSNR medio por objeto (unidad independiente)
    diferencias entre modelos por intervalo fino de ángulo, con su valor p
    concentración de los peores casos por objeto y por ángulo
    hitos de convergencia en validación

Salida: results/analisis.json, con los valores citados en las secciones IV y V.

Uso:
    python -m src.analysis --configs configs/A_*.yaml configs/B_*.yaml configs/C_*.yaml \
        --seeds 0 1 --work /content/drive/MyDrive/clifford_nvs --out results
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .utils import load_config, read_json, run_dir, write_json

FINE_BINS = [0, 30, 60, 90, 120, 150, 165, 175, 180]


def _wilcoxon(diff):
    diff = np.asarray(diff)
    return float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0


def seed_analysis(per_pair, names):
    out = {}
    means = per_pair.groupby(["run", "seed"])["psnr"].mean()
    for n in names:
        out[n] = {f"seed{s}": float(v) for s, v in means.loc[n].items()}
    return out


def paired_by_seed(per_pair, a, b):
    res = {}
    for s in sorted(per_pair["seed"].unique()):
        pa = per_pair[(per_pair.run == a) & (per_pair.seed == s)].set_index("pair")["psnr"]
        pb = per_pair[(per_pair.run == b) & (per_pair.seed == s)].set_index("pair")["psnr"]
        if len(pa) and len(pb):
            res[f"seed{s}"] = float((pa - pb).mean())
    return res


def object_level(per_pair, a, b):
    po = per_pair.groupby(["run", "obj"])["psnr"].mean().unstack(0)
    diff = po[a] - po[b]
    return {"n_objects": int(len(diff)), "mean": float(diff.mean()), "median": float(diff.median()),
            "fraction_positive": float((diff > 0).mean()), "p": _wilcoxon(diff)}


def angle_bins(per_pair, a, b):
    pm = per_pair.groupby(["run", "pair"]).agg(psnr=("psnr", "mean"), angle=("angle", "first")).reset_index()
    wide = pm.pivot(index="pair", columns="run", values="psnr")
    angle = pm.drop_duplicates("pair").set_index("pair")["angle"]
    rows = []
    for lo, hi in zip(FINE_BINS[:-1], FINE_BINS[1:]):
        m = (angle >= lo) & ((angle < hi) if hi < 180 else (angle <= hi))
        diff = wide.loc[m, a] - wide.loc[m, b]
        rows.append({"bin": f"{lo}-{hi}", "n_pairs": int(m.sum()), "mean": float(diff.mean()),
                     "median": float(diff.median()), "p": _wilcoxon(diff)})
    return rows


def worst_cases(per_pair, run, fraction=0.01, top=20):
    pm = per_pair[per_pair.run == run].groupby("pair").agg(
        psnr=("psnr", "mean"), angle=("angle", "first"), obj=("obj", "first"))
    worst = pm.nsmallest(max(1, int(len(pm) * fraction)), "psnr")
    top_k = pm.nsmallest(top, "psnr")
    return {"n_pairs": int(len(worst)), "n_objects": int(worst.obj.nunique()),
            "fraction_above_90deg": float((worst.angle > 90).mean()),
            "fraction_above_90deg_all": float((pm.angle > 90).mean()),
            f"top{top}_by_object": {int(k): int(v) for k, v in top_k.obj.value_counts().items()},
            "fraction_above_175deg_all": float((pm.angle > 175).mean())}


def convergence(cfgs, seeds, work, threshold=18.0):
    out = {}
    for cfg in cfgs:
        logs = [read_json(os.path.join(run_dir(work, cfg["name"], s), "log.json")) for s in seeds
                if os.path.exists(os.path.join(run_dir(work, cfg["name"], s), "log.json"))]
        if not logs:
            continue
        val = pd.concat([pd.DataFrame(l["val"]).set_index("step")["psnr"] for l in logs], axis=1).mean(axis=1)
        last = int(val.index.max())
        prev = int(val.index[val.index <= last - 1000].max()) if (val.index <= last - 1000).any() else last
        reached = val[val >= threshold]
        out[cfg["name"]] = {"first_step_above_threshold": int(reached.index.min()) if len(reached) else None,
                            "threshold_db": threshold, "final_val_psnr": float(val.loc[last]),
                            "gain_last_1000_steps": float(val.loc[last] - val.loc[prev])}
    return out


def main():
    ap = argparse.ArgumentParser(description="Análisis complementario de resultados.")
    ap.add_argument("--configs", nargs="+", required=True, help="El primero es la referencia")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cfgs = [load_config(p) for p in args.configs]
    names = [c["name"] for c in cfgs]
    per_pair = pd.read_csv(os.path.join(args.out, "por_par.csv.gz"))
    ref = names[0]

    comparisons = [(n, ref) for n in names[1:]]
    for c in cfgs:
        if c["model"]["pose"] == "rotor":
            twin = next((o["name"] for o in cfgs if o["model"]["block"] == c["model"]["block"]
                         and o["model"]["pose"] == "matrix"), None)
            if twin and twin != ref:
                comparisons.append((c["name"], twin))

    result = {"psnr_by_seed": seed_analysis(per_pair, names), "comparisons": {}}
    for a, b in comparisons:
        result["comparisons"][f"{a} - {b}"] = {
            "by_seed": paired_by_seed(per_pair, a, b),
            "object_level": object_level(per_pair, a, b),
            "angle_bins": angle_bins(per_pair, a, b),
        }
    result["worst_cases"] = worst_cases(per_pair, names[-1])
    result["convergence"] = convergence(cfgs, args.seeds, args.work)
    write_json(result, os.path.join(args.out, "analisis.json"))

    for key, comp in result["comparisons"].items():
        ol = comp["object_level"]
        print(f"{key}: por semilla {comp['by_seed']} | por objeto media {ol['mean']:+.3f}, "
              f"fracción>0 {ol['fraction_positive']:.3f}, p={ol['p']:.1e}")
    print("peores casos:", result["worst_cases"])
    print("convergencia:", result["convergence"])


if __name__ == "__main__":
    main()
