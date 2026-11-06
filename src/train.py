"""Entrenamiento con reanudación automática.

En el directorio del run se escriben:
    last.pt       estado completo (modelo, optimizador, escalador AMP, generadores
                  aleatorios, registros y tiempo acumulado). Se guarda cada
                  `ckpt_every` pasos o cada `ckpt_minutes` minutos, lo que ocurra
                  antes, mediante escritura atómica.
    best.pt       solo pesos del modelo con mejor PSNR de validación.
    log.json      pérdidas de entrenamiento y PSNR de validación por paso.
    summary.json  resumen final: parámetros, pasos, horas de GPU, mejor validación.

Si el proceso se interrumpe (fin de sesión de Colab, cierre del navegador,
caída del equipo local), ejecutar el mismo comando reanuda desde last.pt con la
misma secuencia aleatoria.

Uso:
    python -m src.train --config configs/C_clifford_rotor.yaml --seed 0 \
        --data data/processed --work /content/drive/MyDrive/clifford_nvs
"""
import argparse
import math
import os
import time

import torch
import torch.nn.functional as F

from .data import PairSampler, iterate_pairs, load_split, make_eval_pairs
from .metrics import LPIPS, psnr
from .models import build_model, count_params
from .utils import atomic_save, load_config, resolve_device, run_dir, seed_everything, write_json


def lr_at(step, total, base_lr, warmup_frac):
    """Calentamiento lineal seguido de decaimiento coseno."""
    warm = max(1, int(warmup_frac * total))
    if step < warm:
        return base_lr * (step + 1) / warm
    progress = min(1.0, (step - warm) / max(1, total - warm))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def validate(model, split, pairs, device, amp, batch=256):
    model.eval()
    scores = []
    for src, tgt, T, _ in iterate_pairs(split, pairs, batch, device):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            pred = model(src, T)
        scores.append(psnr(pred.float().clamp(0, 1), tgt))
    model.train()
    return torch.cat(scores).mean().item()


def _rng_state(sampler):
    state = {"torch": torch.get_rng_state(), "sampler": sampler.state_dict()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state, sampler):
    # torch.load(map_location="cuda") también mueve estos estados a la GPU;
    # PyTorch los exige como ByteTensor en CPU.
    torch.set_rng_state(state["torch"].cpu())
    sampler.load_state_dict({"gen": state["sampler"]["gen"].cpu()})
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def train(cfg, data_dir, out_dir, seed, device="auto", time_budget_h=None):
    device = resolve_device(device)
    tcfg, vcfg = cfg["train"], cfg["val"]
    res = cfg["data"]["res"]
    amp = bool(tcfg["amp"]) and device == "cuda"
    os.makedirs(out_dir, exist_ok=True)
    last_path, best_path = os.path.join(out_dir, "last.pt"), os.path.join(out_dir, "best.pt")

    if os.path.exists(os.path.join(out_dir, "summary.json")):
        print(f"{out_dir}: entrenamiento ya completado")
        return

    seed_everything(seed)
    torch.backends.cudnn.benchmark = True

    train_split = load_split(os.path.join(data_dir, f"cars_train_{res}.npz"), device)
    val_split = load_split(os.path.join(data_dir, f"cars_val_{res}.npz"), device)
    val_pairs = make_eval_pairs(val_split, vcfg["n_objects"], vcfg["n_targets"], vcfg["src_view"], seed=0)
    sampler = PairSampler(train_split, tcfg["batch"], seed, device)

    model = build_model(cfg["model"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"],
                            betas=(0.9, 0.99))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    lpips_fn = LPIPS("alex", device) if tcfg["lpips_weight"] > 0 else None

    step, train_seconds, best_val = 0, 0.0, -float("inf")
    resumed_counts = (0, 0)
    log = {"train": [], "val": []}
    if os.path.exists(last_path):
        # Carga en CPU: load_state_dict copia pesos y estado del optimizador al dispositivo,
        # y los estados aleatorios deben permanecer en CPU.
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        _restore_rng(ck["rng"], sampler)
        step, train_seconds, best_val, log = ck["step"], ck["train_seconds"], ck["best_val"], ck["log"]
        resumed_counts = (ck.get("skipped", 0), ck.get("rollbacks", 0))
        print(f"Reanudando {out_dir} desde el paso {step} ({train_seconds / 3600:.2f} h acumuladas)")

    n_params = count_params(model)
    print(f"{cfg['name']} | semilla {seed} | {n_params / 1e6:.3f} M parámetros | {device}")

    def state_is_finite():
        return all(torch.isfinite(t).all() for t in model.state_dict().values() if t.is_floating_point())

    def snapshot():
        return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}

    good_state = snapshot()            # última copia sana de pesos y estadísticas de BatchNorm
    skipped, rollbacks = resumed_counts

    def save_last():
        """Solo se guarda un estado finito; si no lo es, se restaura la última copia sana."""
        nonlocal good_state
        if not state_is_finite():
            rollback()
            return
        good_state = snapshot()
        atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                     "rng": _rng_state(sampler), "step": step, "train_seconds": elapsed(),
                     "best_val": best_val, "log": log, "cfg": cfg, "seed": seed,
                     "skipped": skipped, "rollbacks": rollbacks}, last_path)
        write_json(log, os.path.join(out_dir, "log.json"))

    total = tcfg["steps"]
    session_start = time.time()
    base_seconds = train_seconds

    def elapsed():
        """Horas de GPU como tiempo de pared acumulado entre sesiones."""
        return base_seconds + time.time() - session_start

    def rollback():
        nonlocal rollbacks
        rollbacks += 1
        model.load_state_dict(good_state)
        print(f"           estado no finito: se restauró la última copia sana ({rollbacks}/{max_rollbacks})")
        if rollbacks > max_rollbacks:
            raise RuntimeError("El entrenamiento diverge de forma persistente; reduzca train.lr.")

    max_rollbacks = tcfg.get("max_rollbacks", 5)
    last_ckpt_time = time.time()
    window = {"l1": 0.0, "lpips": 0.0, "n": 0, "skipped": 0, "t0": time.time()}
    model.train()

    while step < total:
        for group in opt.param_groups:
            group["lr"] = lr_at(step, total, tcfg["lr"], tcfg["warmup_frac"])

        src, tgt, T = sampler()
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            pred = model(src, T)
        # Las pérdidas se calculan en float32: LPIPS normaliza características y es frágil en fp16.
        pred = pred.float()
        l1 = F.l1_loss(pred, tgt)
        lp = lpips_fn(pred, tgt).mean() if lpips_fn else torch.zeros((), device=device)
        loss = l1 + tcfg["lpips_weight"] * lp
        opt.zero_grad(set_to_none=True)
        step += 1

        if not torch.isfinite(loss):
            # Paso descartado. La pasada hacia adelante pudo contaminar las estadísticas de BatchNorm.
            skipped += 1
            window["skipped"] += 1
            if not state_is_finite():
                rollback()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            window["l1"] += l1.detach()
            window["lpips"] += lp.detach()
            window["n"] += 1

        if step % tcfg["log_every"] == 0 or step == total:
            n = max(window["n"], 1)
            steps_done = window["n"] + window["skipped"]
            rec = {"step": step, "l1": float(window["l1"]) / n, "lpips": float(window["lpips"]) / n,
                   "skipped": window["skipped"], "it_s": steps_done / (time.time() - window["t0"]),
                   "lr": opt.param_groups[0]["lr"]}
            log["train"].append(rec)
            window = {"l1": 0.0, "lpips": 0.0, "n": 0, "skipped": 0, "t0": time.time()}
            note = f"  omitidos {rec['skipped']}" if rec["skipped"] else ""
            print(f"[{step:>6}/{total}] L1 {rec['l1']:.4f}  LPIPS {rec['lpips']:.4f}  "
                  f"{rec['it_s']:.1f} it/s  {elapsed() / 3600:.2f} h{note}")

        if step % tcfg["val_every"] == 0 or step == total:
            if not state_is_finite():
                rollback()
            val_psnr = validate(model, val_split, val_pairs, device, amp)
            log["val"].append({"step": step, "psnr": val_psnr})
            if math.isfinite(val_psnr) and val_psnr > best_val:
                best_val = val_psnr
                atomic_save({"model": model.state_dict(), "cfg": cfg, "step": step, "val_psnr": val_psnr},
                            best_path)
            print(f"           validación PSNR {val_psnr:.2f} dB (mejor {best_val:.2f})")

        due = step % tcfg["ckpt_every"] == 0 or time.time() - last_ckpt_time > 60 * tcfg["ckpt_minutes"]
        if due:
            save_last()
            last_ckpt_time = time.time()

        if time_budget_h and time.time() - session_start > time_budget_h * 3600:
            save_last()
            print("Presupuesto de tiempo de la sesión agotado; el run se reanudará en la siguiente.")
            return

    save_last()
    summary = {"name": cfg["name"], "seed": seed, "params": n_params, "steps": step,
               "gpu_hours": elapsed() / 3600, "best_val_psnr": best_val,
               "skipped_steps": skipped, "rollbacks": rollbacks,
               "device": torch.cuda.get_device_name() if device == "cuda" else device}
    write_json(summary, os.path.join(out_dir, "summary.json"))
    print(f"Completado: {summary}")


def main():
    ap = argparse.ArgumentParser(description="Entrena una configuración de la ablación.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", required=True, help="Directorio con los .npz preprocesados")
    ap.add_argument("--work", required=True, help="Directorio persistente para runs (p. ej. en Drive)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--time_budget_h", type=float, default=None)
    ap.add_argument("--override", nargs="*", default=[], help="seccion.clave=valor")
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    train(cfg, args.data, run_dir(args.work, cfg["name"], args.seed), args.seed, args.device, args.time_budget_h)


if __name__ == "__main__":
    main()
