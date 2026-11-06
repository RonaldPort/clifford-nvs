"""Configuración, semillas y escritura segura de checkpoints."""
import copy
import json
import os
import random

import numpy as np
import torch
import yaml


def _deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in override.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _parse_value(text):
    return yaml.safe_load(text)


def load_config(path, overrides=()):
    """Lee un YAML; la clave `base` hereda de otro YAML del mismo directorio.
    `overrides` acepta cadenas `seccion.clave=valor` para pruebas rápidas."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    parent = cfg.pop("base", None)
    if parent:
        cfg = _deep_merge(load_config(os.path.join(os.path.dirname(path), parent)), cfg)
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        *path_keys, last = key.split(".")
        for k in path_keys:
            node = node.setdefault(k, {})
        node[last] = _parse_value(value)
    return cfg


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_save(obj, path):
    """Escribe en un temporal y lo renombra: un corte durante la escritura
    deja intacto el checkpoint anterior."""
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def run_dir(work, name, seed):
    return os.path.join(work, "runs", name, f"seed{seed}")


def resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def write_text(text, path):
    with open(path, "w") as f:
        f.write(text)
