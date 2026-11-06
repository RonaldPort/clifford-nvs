"""Preprocesado de ShapeNet-SRN Cars, muestreo de pares y protocolo de evaluación.

Cada partición se guarda como un .npz con:
    images     (N, H, W, 3) uint8
    poses      (N, 4, 4)    float32, cámara→mundo
    obj_offset (O,)         índice de la primera vista de cada objeto
    obj_count  (O,)         número de vistas de cada objeto
    obj_names  (O,)         identificador ShapeNet del objeto

Uso:
    python -m src.data --srn_root /content/srn --out data/processed
"""
import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np
import torch
from PIL import Image

from .geometry import relative_pose, rotation_angle_deg

SPLITS = ("cars_train", "cars_val", "cars_test")


# --------------------------------------------------------------------------
# Preprocesado
# --------------------------------------------------------------------------

def _load_image(args):
    path, res = args
    with Image.open(path) as im:
        im = im.convert("RGBA")
        canvas = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(canvas, im).convert("RGB")
        return np.asarray(im.resize((res, res), Image.LANCZOS), dtype=np.uint8)


def preprocess_split(split_dir, out_path, res=64, max_objects=None, workers=None):
    objects = sorted(d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d)))
    if max_objects:
        objects = objects[:max_objects]

    paths, poses, offsets, counts, names = [], [], [], [], []
    for name in objects:
        rgb = sorted(glob.glob(os.path.join(split_dir, name, "rgb", "*.png")))
        pose = sorted(glob.glob(os.path.join(split_dir, name, "pose", "*.txt")))
        n = min(len(rgb), len(pose))
        if n < 2:
            continue
        offsets.append(len(paths))
        counts.append(n)
        names.append(name)
        paths += rgb[:n]
        poses += [np.loadtxt(p, dtype=np.float32).reshape(4, 4) for p in pose[:n]]

    images = np.empty((len(paths), res, res, 3), dtype=np.uint8)
    with Pool(workers or os.cpu_count()) as pool:
        for i, img in enumerate(pool.imap(_load_image, [(p, res) for p in paths], chunksize=256)):
            images[i] = img

    np.savez(out_path, images=images, poses=np.stack(poses), obj_offset=np.array(offsets),
             obj_count=np.array(counts), obj_names=np.array(names))
    print(f"{os.path.basename(out_path)}: {len(names)} objetos, {len(paths)} vistas")


def find_split_root(srn_root):
    """Devuelve el directorio que contiene cars_train/, sin suponer cómo se descomprimió."""
    hits = glob.glob(os.path.join(srn_root, "**", "cars_train"), recursive=True)
    if not hits:
        raise FileNotFoundError(f"No se encontró cars_train dentro de {srn_root}")
    return os.path.dirname(hits[0])


# --------------------------------------------------------------------------
# Carga y muestreo
# --------------------------------------------------------------------------

def load_split(path, device="cpu"):
    d = np.load(path)
    return {
        "images": torch.from_numpy(d["images"]).to(device),
        "poses": torch.from_numpy(d["poses"]).to(device),
        "obj_offset": torch.from_numpy(d["obj_offset"]).long().to(device),
        "obj_count": torch.from_numpy(d["obj_count"]).long().to(device),
    }


def to_float_images(images_uint8):
    """(B, H, W, 3) uint8 → (B, 3, H, W) float en [0, 1]."""
    return images_uint8.permute(0, 3, 1, 2).float().div_(255.0)


class PairSampler:
    """Muestrea pares (fuente, objetivo) de un mismo objeto directamente en el
    dispositivo. Evita el DataLoader, que en Colab queda limitado por sus dos
    CPU. El generador propio permite reanudar con la misma secuencia."""

    def __init__(self, split, batch, seed, device):
        self.s, self.batch, self.device = split, batch, device
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)

    def __call__(self):
        B, g = self.batch, self.gen
        obj = torch.randint(len(self.s["obj_offset"]), (B,), generator=g, device=self.device)
        cnt = self.s["obj_count"][obj]
        a = (torch.rand(B, generator=g, device=self.device) * cnt).long()
        b = (a + 1 + (torch.rand(B, generator=g, device=self.device) * (cnt - 1)).long()) % cnt
        i, j = self.s["obj_offset"][obj] + a, self.s["obj_offset"][obj] + b
        src = to_float_images(self.s["images"][i])
        tgt = to_float_images(self.s["images"][j])
        return src, tgt, relative_pose(self.s["poses"][i], self.s["poses"][j])

    def state_dict(self):
        return {"gen": self.gen.get_state()}

    def load_state_dict(self, state):
        self.gen.set_state(state["gen"])


# --------------------------------------------------------------------------
# Protocolo de evaluación
# --------------------------------------------------------------------------

def make_eval_pairs(split, n_objects=None, n_targets=16, src_view=64, seed=0):
    """Pares fijos: la vista `src_view` de cada objeto como fuente (protocolo de
    pixelNeRF en SRN Cars) y `n_targets` vistas objetivo distintas, elegidas con
    una semilla fija. Devuelve índices globales, objeto y ángulo relativo."""
    rng = np.random.default_rng(seed)
    off = split["obj_offset"].cpu().numpy()
    cnt = split["obj_count"].cpu().numpy()
    objs = np.arange(len(off))
    if n_objects and n_objects < len(off):
        objs = np.sort(rng.choice(len(off), n_objects, replace=False))
    src, tgt, oid = [], [], []
    for o in objs:
        s = src_view if src_view < cnt[o] else 0
        cand = np.setdiff1d(np.arange(cnt[o]), [s])
        for t in rng.choice(cand, min(n_targets, len(cand)), replace=False):
            src.append(off[o] + s)
            tgt.append(off[o] + t)
            oid.append(o)
    src, tgt = torch.tensor(src), torch.tensor(tgt)
    poses = split["poses"].cpu()
    angle = rotation_angle_deg(relative_pose(poses[src], poses[tgt])[:, :3, :3])
    return {"src": src, "tgt": tgt, "obj": torch.tensor(oid), "angle": angle}


def iterate_pairs(split, pairs, batch, device):
    """Recorre los pares en lotes y entrega (fuente, objetivo, pose relativa, slice)."""
    n = len(pairs["src"])
    for k in range(0, n, batch):
        sl = slice(k, min(k + batch, n))
        si, ti = pairs["src"][sl], pairs["tgt"][sl]
        imgs, poses = split["images"], split["poses"]
        src = to_float_images(imgs[si.to(imgs.device)]).to(device, non_blocking=True)
        tgt = to_float_images(imgs[ti.to(imgs.device)]).to(device, non_blocking=True)
        T = relative_pose(poses[si.to(poses.device)], poses[ti.to(poses.device)]).to(device)
        yield src, tgt, T, sl


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Preprocesa SRN Cars a .npz de baja resolución.")
    ap.add_argument("--srn_root", required=True, help="Directorio donde se descomprimió srn_cars.zip")
    ap.add_argument("--out", required=True, help="Directorio de salida de los .npz")
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--val_objects", type=int, default=100,
                    help="Objetos de validación a conservar (la validación periódica usa 50)")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    root = find_split_root(args.srn_root)
    os.makedirs(args.out, exist_ok=True)
    limits = {"cars_train": None, "cars_val": args.val_objects, "cars_test": None}
    for split in SPLITS:
        out = os.path.join(args.out, f"{split}_{args.res}.npz")
        if os.path.exists(out):
            print(f"{out} ya existe; se omite")
            continue
        preprocess_split(os.path.join(root, split), out, args.res, limits[split], args.workers)


if __name__ == "__main__":
    main()
