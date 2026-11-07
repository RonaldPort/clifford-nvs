"""Síntesis de nuevas vistas a partir de una imagen arbitraria de un automóvil.

Los modelos se entrenaron con renders de ShapeNet-SRN Cars: objeto centrado,
fondo blanco, 64×64 y cámaras sobre una esfera alrededor del objeto. Una foto
cualquiera se lleva a ese formato en tres pasos:

    1. Segmentación del auto: canal alfa si existe, rembg si está instalado o,
       en su defecto, umbral respecto al color del borde de la imagen.
    2. Estimación del punto de vista: la silueta normalizada se compara por IoU
       con las siluetas de los objetos de la muestra en sus 251 vistas, con y
       sin reflejo horizontal. La mejor coincidencia fija la pose fuente.
    3. Encuadre: el auto se escala y centra como aparece un auto de SRN en esa
       vista y se compone sobre fondo blanco.

Las poses objetivo son las 251 vistas de la espiral de prueba del objeto de
referencia, de modo que T_rel = T_s^{-1} T_t pertenece a la distribución de
entrenamiento. La brecha de dominio (iluminación, texturas, fondo real) sigue
presente: la órbita resultante es una reconstrucción plausible, no exacta.

Uso:
    python -m src.infer_image --image foto.jpg --release release --out salida_imagen
    python -m src.infer_image --image foto.jpg --release release --view 64 --obj 0 --flip 0
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage

from .data import load_split, to_float_images
from .geometry import relative_pose, rotation_angle_deg
from .models import build_model

SIL = 32          # resolución de las siluetas normalizadas
WORK = 256        # resolución de trabajo antes de reducir a 64×64
WHITE_THR = 245   # un píxel de SRN es fondo si sus tres canales superan este valor
REMBG_MODEL = "isnet-general-use"   # 170 MB; el modelo por omisión de rembg pesa 1 GB
_REMBG = None


# --------------------------------------------------------------------------
# Modelos y muestra de referencia
# --------------------------------------------------------------------------

def load_models(release, device):
    """Carga los .pt publicados en `release`, indexados por su etiqueta."""
    models = {}
    for f in sorted(os.listdir(release)):
        if f.endswith(".pt"):
            ck = torch.load(os.path.join(release, f), map_location=device, weights_only=False)
            m = build_model(ck["cfg"]["model"]).to(device).eval()
            m.load_state_dict(ck["model"])
            models[ck["cfg"].get("label", f[:-3])] = m
    if not models:
        raise FileNotFoundError(f"No hay pesos .pt en {release}")
    return models


def _largest_component(mask):
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def _clean(mask):
    mask = ndimage.binary_opening(mask, iterations=1)
    mask = ndimage.binary_closing(mask, iterations=2)
    return ndimage.binary_fill_holes(_largest_component(mask))


def _bbox(mask):
    ys, xs = np.nonzero(mask)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def normalized_silhouette(mask):
    """Silueta recortada a su caja, centrada en un cuadrado y reducida a SIL×SIL."""
    x0, y0, x1, y1 = _bbox(mask)
    crop = mask[y0:y1, x0:x1].astype(np.uint8) * 255
    h, w = crop.shape
    side = max(h, w)
    sq = np.zeros((side, side), np.uint8)
    sq[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = crop
    return np.asarray(Image.fromarray(sq).resize((SIL, SIL), Image.Resampling.BILINEAR)) > 127


def reference_bank(sample):
    """Siluetas normalizadas y encuadre (lado mayor y centro de la caja, en
    fracción de la imagen) de cada vista de cada objeto de la muestra."""
    imgs = sample["images"].numpy()
    res = imgs.shape[1]
    sil, frame = [], []
    for img in imgs:
        mask = _clean((img < WHITE_THR).any(-1))
        x0, y0, x1, y1 = _bbox(mask)
        sil.append(normalized_silhouette(mask))
        frame.append((max(x1 - x0, y1 - y0) / res, (x0 + x1) / (2 * res), (y0 + y1) / (2 * res)))
    return np.stack(sil), np.array(frame)


# --------------------------------------------------------------------------
# Segmentación y encuadre de la imagen de entrada
# --------------------------------------------------------------------------

def foreground(img, method="auto"):
    """Devuelve (rgb float [0,1], alfa float [0,1], método usado)."""
    if method in ("auto", "alpha") and img.mode in ("RGBA", "LA"):
        a = np.asarray(img.getchannel("A"), np.float32) / 255
        if a.min() < 0.98:
            return np.asarray(img.convert("RGB"), np.float32) / 255, a, "alpha"
    rgb = np.asarray(img.convert("RGB"), np.float32) / 255
    if method in ("auto", "rembg"):
        try:
            from rembg import new_session, remove
            global _REMBG
            if _REMBG is None:
                _REMBG = new_session(REMBG_MODEL)
            a = np.asarray(remove(img.convert("RGB"), session=_REMBG).getchannel("A"), np.float32) / 255
            a = a * _clean(a > 0.5)
            return rgb, a, "rembg"
        except ImportError:
            if method == "rembg":
                raise
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    dist = np.linalg.norm(rgb - np.median(border, 0), axis=-1)
    thr = max(0.12, float(np.percentile(np.linalg.norm(border - np.median(border, 0), axis=-1), 99)))
    mask = _clean(dist > thr)
    return rgb, ndimage.gaussian_filter(mask.astype(np.float32), 0.7), "border"


def frame_like_srn(rgb, alpha, side_frac, cx, cy, res=64):
    """Compone el objeto sobre blanco con el lado mayor de su caja igual a
    `side_frac` de la imagen y su centro en (cx, cy), y reduce a res×res."""
    x0, y0, x1, y1 = _bbox(alpha > 0.5)
    scale = side_frac * WORK / max(x1 - x0, y1 - y0)
    rgba = np.concatenate([rgb, alpha[..., None]], -1)[y0:y1, x0:x1]
    obj = Image.fromarray((rgba * 255).astype(np.uint8), "RGBA")
    obj = obj.resize((max(1, round(obj.width * scale)), max(1, round(obj.height * scale))), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (WORK, WORK), (255, 255, 255, 255))
    canvas.alpha_composite(obj, (round(cx * WORK - obj.width / 2), round(cy * WORK - obj.height / 2)))
    return canvas.convert("RGB").resize((res, res), Image.Resampling.LANCZOS)


# --------------------------------------------------------------------------
# Punto de vista, órbita y figuras
# --------------------------------------------------------------------------

def rank_views(sil, bank, offsets, counts, top=6):
    """Ordena (objeto, vista, reflejo) por IoU entre siluetas normalizadas."""
    cands = []
    for flip in (0, 1):
        s = sil[:, ::-1] if flip else sil
        inter = (bank & s).sum((1, 2))
        iou = inter / np.maximum((bank | s).sum((1, 2)), 1)
        for g in np.argsort(-iou)[:top]:
            obj = int(np.searchsorted(offsets, g, side="right") - 1)
            cands.append({"obj": obj, "view": int(g - offsets[obj]), "flip": flip,
                          "iou": float(iou[g]), "index": int(g)})
    cands.sort(key=lambda c: -c["iou"])
    seen, out = set(), []
    for c in cands:
        if (c["obj"], c["view"], c["flip"]) not in seen:
            seen.add((c["obj"], c["view"], c["flip"]))
            out.append(c)
    return out[:top]


@torch.no_grad()
def orbit(models, src, poses_obj, src_view, device, batch=64):
    """Predice las vistas de todas las poses del objeto de referencia."""
    n = len(poses_obj)
    T = relative_pose(poses_obj[src_view].expand(n, 4, 4), poses_obj)
    x = to_float_images(torch.from_numpy(np.array(src))[None]).to(device)
    preds = {}
    for name, m in models.items():
        out = [m(x.expand(len(Tb), -1, -1, -1), Tb.to(device)).float().clamp(0, 1).cpu()
               for Tb in T.split(batch)]
        preds[name] = (torch.cat(out).permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    return preds, rotation_angle_deg(T[:, :3, :3]).numpy()


def _label(img, text):
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, img.width, 14), fill=(255, 255, 255))
    d.text((3, 1), text, fill=(0, 0, 0))
    return img


def candidates_sheet(entrada, sil, cands, sample, scale=3):
    """Entrada, su silueta y los renders de referencia de las mejores vistas."""
    cell = 64 * scale
    tiles = [_label(entrada.resize((cell, cell), Image.Resampling.NEAREST), "entrada"),
             _label(Image.fromarray(np.uint8(sil) * 255).convert("RGB").resize((cell, cell), Image.Resampling.NEAREST),
                    "silueta")]
    for c in cands:
        ref = Image.fromarray(sample["images"][c["index"]].numpy())
        ref = ImageOps.mirror(ref) if c["flip"] else ref
        tiles.append(_label(ref.resize((cell, cell), Image.Resampling.NEAREST),
                            f"o{c['obj']} v{c['view']}{' R' if c['flip'] else ''} IoU {c['iou']:.2f}"))
    sheet = Image.new("RGB", (cell * len(tiles), cell), "white")
    for k, t in enumerate(tiles):
        sheet.paste(t, (k * cell, 0))
    return sheet


def orbit_gif(path, entrada, preds, order, flip, scale=3, every=2, ms=60):
    """Animación: entrada | predicción de cada modelo, recorriendo la espiral."""
    src = np.asarray(entrada)
    frames = []
    for k in order[::every]:
        row = [src] + [np.asarray(ImageOps.mirror(Image.fromarray(p[k]))) if flip else p[k]
                       for p in preds.values()]
        im = Image.fromarray(np.concatenate(row, 1))
        frames.append(im.resize((im.width * scale, im.height * scale), Image.Resampling.NEAREST))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms, loop=0)


def turntable(path, preds, angles, flip, n=8, scale=2):
    """Rejilla modelo × ángulo con n vistas de ángulo relativo creciente."""
    picks = [int(np.argmin(np.abs(angles - a))) for a in np.linspace(0, angles.max(), n)]
    rows = []
    for name, p in preds.items():
        row = [np.asarray(ImageOps.mirror(Image.fromarray(p[k]))) if flip else p[k] for k in picks]
        rows.append(np.concatenate(row, 1))
    grid = Image.fromarray(np.concatenate(rows, 0))
    grid = grid.resize((grid.width * scale, grid.height * scale), Image.Resampling.NEAREST)
    pad = Image.new("RGB", (grid.width + 150, grid.height + 16), "white")
    pad.paste(grid, (150, 16))
    d = ImageDraw.Draw(pad)
    cell = 64 * scale
    for r, name in enumerate(preds):
        d.text((4, 16 + r * cell + cell // 2 - 6), name, fill=(0, 0, 0))
    for c, k in enumerate(picks):
        d.text((150 + c * cell + cell // 2 - 12, 2), f"{angles[k]:.0f}°", fill=(0, 0, 0))
    pad.save(path)
    return picks


# --------------------------------------------------------------------------
# Programa principal
# --------------------------------------------------------------------------

def run(image, release, out, view="auto", obj="auto", flip="auto", mask="auto", device="auto", top=6):
    os.makedirs(out, exist_ok=True)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    models = load_models(release, device)
    sample = load_split(os.path.join(release, "demo_sample.npz"))
    offsets, counts = sample["obj_offset"].numpy(), sample["obj_count"].numpy()
    bank, frames = reference_bank(sample)

    img = ImageOps.exif_transpose(Image.open(image))
    img.thumbnail((1024, 1024))
    rgb, alpha, used = foreground(img, mask)
    if (alpha > 0.5).sum() < 50:
        raise ValueError("No se encontró un objeto en primer plano; prueba con --mask rembg o una imagen PNG con transparencia.")
    sil = normalized_silhouette(alpha > 0.5)
    cands = rank_views(sil, bank, offsets, counts, top)

    best = dict(cands[0])
    if obj != "auto":
        best["obj"] = int(obj)
    if view != "auto":
        best["view"] = int(view)
    if flip != "auto":
        best["flip"] = int(flip)
    best["index"] = int(offsets[best["obj"]] + best["view"])
    s = sil[:, ::-1] if best["flip"] else sil
    ref = bank[best["index"]]
    best["iou"] = float((ref & s).sum() / max((ref | s).sum(), 1))

    # La imagen se refleja si su silueta coincide con la vista reflejada; las
    # predicciones se reflejan de vuelta para conservar la orientación original.
    rgb_in, a_in = (rgb[:, ::-1], alpha[:, ::-1]) if best["flip"] else (rgb, alpha)
    entrada = frame_like_srn(np.ascontiguousarray(rgb_in), np.ascontiguousarray(a_in), *frames[best["index"]])
    entrada_vista = ImageOps.mirror(entrada) if best["flip"] else entrada

    o, c = offsets[best["obj"]], counts[best["obj"]]
    preds, angles = orbit(models, entrada, sample["poses"][o:o + c], best["view"], device)

    entrada_vista.resize((192, 192), Image.Resampling.NEAREST).save(os.path.join(out, "entrada_64.png"))
    candidates_sheet(entrada_vista, sil, cands, sample).save(os.path.join(out, "candidatos.png"))
    orbit_gif(os.path.join(out, "orbita.gif"), entrada_vista, preds, np.arange(c), best["flip"])
    turntable(os.path.join(out, "vistas.png"), preds, angles, best["flip"])
    summary = {"imagen": os.path.basename(image), "segmentacion": used, "elegida": best,
               "candidatos": cands, "modelos": list(models), "n_vistas": int(c),
               "angulo_max": float(angles.max()),
               "aviso": None if best["iou"] >= 0.6 else
               "Silueta poco parecida a las de SRN: revisa la segmentación o elige la vista a mano."}
    with open(os.path.join(out, "resumen.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary, preds, angles


def main():
    ap = argparse.ArgumentParser(description="Órbita de nuevas vistas a partir de una imagen propia.")
    ap.add_argument("--image", required=True)
    ap.add_argument("--release", default="release", help="Carpeta con los .pt y demo_sample.npz")
    ap.add_argument("--out", default="salida_imagen")
    ap.add_argument("--view", default="auto", help="Vista fuente de la espiral (0-250) o auto")
    ap.add_argument("--obj", default="auto", help="Objeto de referencia de la muestra o auto")
    ap.add_argument("--flip", default="auto", choices=["auto", "0", "1"])
    ap.add_argument("--mask", default="auto", choices=["auto", "alpha", "rembg", "border"])
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    s, _, _ = run(args.image, args.release, args.out, args.view, args.obj, args.flip, args.mask, args.device)
    e = s["elegida"]
    print(f"Segmentación: {s['segmentacion']} | objeto {e['obj']}, vista {e['view']}, "
          f"reflejo {e['flip']}, IoU {e['iou']:.2f}")
    if s["aviso"]:
        print("Aviso:", s["aviso"])
    print(f"Salidas en {args.out}: entrada_64.png, candidatos.png, vistas.png, orbita.gif, resumen.json")


if __name__ == "__main__":
    main()
