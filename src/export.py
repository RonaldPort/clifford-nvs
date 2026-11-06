"""Empaqueta los pesos publicables y una muestra de prueba para la demostración.

Escribe en --out:
    <nombre>_seed<k>.pt   pesos de best.pt y su configuración (sin optimizador)
    demo_sample.npz       primeros `n_objects` objetos de la partición de prueba

La carpeta resultante se comparte en Drive o Hugging Face y su enlace se
indica en el README y en notebooks/02_demo_inferencia.ipynb.

Uso:
    python -m src.export --configs configs/A_*.yaml configs/B_*.yaml configs/C_*.yaml \
        --seeds 0 --data data/processed --work /content/drive/MyDrive/clifford_nvs --out release
"""
import argparse
import os

import numpy as np
import torch

from .utils import load_config, run_dir


def main():
    ap = argparse.ArgumentParser(description="Exporta pesos y muestra de demostración.")
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default="release")
    ap.add_argument("--n_objects", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for path in args.configs:
        cfg = load_config(path)
        for seed in args.seeds:
            src = os.path.join(run_dir(args.work, cfg["name"], seed), "best.pt")
            if not os.path.exists(src):
                print(f"Falta {src}")
                continue
            ck = torch.load(src, map_location="cpu", weights_only=False)
            dst = os.path.join(args.out, f"{cfg['name']}_seed{seed}.pt")
            torch.save({"model": ck["model"], "cfg": ck["cfg"], "step": ck["step"],
                        "val_psnr": ck["val_psnr"]}, dst)
            print(f"{dst}  (paso {ck['step']}, validación {ck['val_psnr']:.2f} dB)")

    d = np.load(os.path.join(args.data, f"cars_test_{load_config(args.configs[0])['data']['res']}.npz"))
    off, cnt = d["obj_offset"][:args.n_objects], d["obj_count"][:args.n_objects]
    end = off[-1] + cnt[-1]
    np.savez_compressed(os.path.join(args.out, "demo_sample.npz"), images=d["images"][:end],
                        poses=d["poses"][:end], obj_offset=off, obj_count=cnt,
                        obj_names=d["obj_names"][:args.n_objects])
    print(f"demo_sample.npz: {args.n_objects} objetos, {end} vistas")


if __name__ == "__main__":
    main()
