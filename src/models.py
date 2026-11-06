"""U-Net condicionada por pose con bloques convolucionales o de interacción de Clifford.

Bloques:
    ConvBlock      dwconv 7×7 + MLP puntual (estilo ConvNeXt), con FFN.
    CliffordBlock  interacción geométrica dispersa de CliffordNet, sin FFN.

Condicionamiento de la pose relativa T_rel = [R | t]:
    matrix  FiLM con la matriz aplanada (R, t) ∈ R^12.
    rotor   FiLM con (q, t) ∈ R^7, q ∈ Spin(3), y transporte del latente del
            cuello de botella por el producto sándwich z_k' = q z_k q~.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import rotmat_to_rotor, rotor_to_rotmat


# --------------------------------------------------------------------------
# Componentes comunes
# --------------------------------------------------------------------------

class LayerNorm2d(nn.Module):
    """LayerNorm sobre canales; se calcula en float32 para estabilidad con AMP."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        mu = xf.mean(1, keepdim=True)
        var = (xf - mu).pow(2).mean(1, keepdim=True)
        y = (xf - mu) * torch.rsqrt(var + self.eps)
        return (self.weight[:, None, None] * y + self.bias[:, None, None]).to(x.dtype)


class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = x.new_empty(x.shape[0], 1, 1, 1).bernoulli_(1 - self.p)
        return x * keep / (1 - self.p)


def _apply_film(h, film):
    if film is None:
        return h
    gamma, beta = film
    return h * (1 + gamma) + beta


# --------------------------------------------------------------------------
# Bloques
# --------------------------------------------------------------------------

class CliffordInteraction(nn.Module):
    """Producto geométrico disperso entre estado z1 y contexto C = z2 - z1.

    Para cada desplazamiento de canal s:
        cuña      z1 ⊙ roll(C, s) - C ⊙ roll(z1, s)   (componente bivectorial)
        interior  SiLU(z1 ⊙ roll(C, s))                (componente escalar)
    Las 2·|S| respuestas se proyectan de vuelta a `dim` con una convolución 1×1.
    Reproduce `CliffordInteraction_PyTorch` (cli_mode='full', ctx_mode='diff')
    del repositorio oficial ParaMind2025/CAN.
    """

    def __init__(self, dim, shifts=(1, 2)):
        super().__init__()
        self.shifts = [s for s in shifts if s < dim]
        self.proj = nn.Conv2d(2 * dim * len(self.shifts), dim, 1)

    def forward(self, z1, z2):
        # Los productos bilineales pueden superar el rango de fp16 (65504): se calculan en float32.
        with torch.autocast(device_type=z1.device.type, enabled=False):
            z1, z2 = z1.float(), z2.float()
            ctx = z2 - z1
            feats = []
            for s in self.shifts:
                ctx_s = torch.roll(ctx, s, dims=1)
                feats.append(z1 * ctx_s - ctx * torch.roll(z1, s, dims=1))
                feats.append(F.silu(z1 * ctx_s))
            return self.proj(torch.cat(feats, 1))


class CliffordBlock(nn.Module):
    def __init__(self, dim, shifts=(1, 2), drop_path=0.0, ls_init=1e-2, **_):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.state = nn.Conv2d(dim, dim, 1)
        self.context = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
        )
        self.interaction = CliffordInteraction(dim, shifts)
        self.gate = nn.Conv2d(2 * dim, dim, 1)
        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), ls_init))
        self.drop_path = DropPath(drop_path)

    def forward(self, x, film=None):
        h = _apply_film(self.norm(x), film)
        g = self.interaction(self.state(h), self.context(h))
        gate = torch.sigmoid(self.gate(torch.cat([h, g], 1)))
        return x + self.drop_path(self.gamma * (F.silu(h) + gate * g))


class ConvBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=2.0, drop_path=0.0, ls_init=1e-2, **_):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.mlp = nn.Sequential(nn.Conv2d(dim, hidden, 1), nn.GELU(), nn.Conv2d(hidden, dim, 1))
        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), ls_init))
        self.drop_path = DropPath(drop_path)

    def forward(self, x, film=None):
        h = _apply_film(self.norm(self.dwconv(x)), film)
        return x + self.drop_path(self.gamma * self.mlp(h))


BLOCKS = {"conv": ConvBlock, "clifford": CliffordBlock}


# --------------------------------------------------------------------------
# Condicionamiento por pose
# --------------------------------------------------------------------------

class FourierEmbedding(nn.Module):
    def __init__(self, in_dim, n_freq=6):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(n_freq) * math.pi, persistent=False)
        self.out_dim = in_dim * (1 + 2 * n_freq)

    def forward(self, x):
        xf = x[..., None] * self.freqs
        return torch.cat([x, xf.sin().flatten(-2), xf.cos().flatten(-2)], -1)


class PoseEncoder(nn.Module):
    def __init__(self, in_dim, emb_dim):
        super().__init__()
        self.fourier = FourierEmbedding(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.fourier.out_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))

    def forward(self, x):
        return self.mlp(self.fourier(x))


class RotorTransport(nn.Module):
    """Agrupa los canales del latente en vectores de R^3 (grado 1 de Cl(3,0)) y
    les aplica el sándwich con el rotor relativo. Las proyecciones 1×1 permiten
    que la red elija qué subespacio se transporta; `post` empieza en cero para
    que el bloque arranque como identidad. Al existir estas proyecciones
    aprendidas, el bloque introduce un sesgo inductivo, no equivariancia exacta."""

    def __init__(self, dim):
        super().__init__()
        self.n = (dim // 3) * 3
        self.pre = nn.Conv2d(dim, dim, 1)
        self.post = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.post.weight)
        nn.init.zeros_(self.post.bias)

    def forward(self, z, R_q):
        B, C, H, W = z.shape
        u = self.pre(z)
        v = u[:, :self.n].reshape(B, self.n // 3, 3, H, W)
        v = torch.einsum("bij,bkjhw->bkihw", R_q.to(v.dtype), v).reshape(B, self.n, H, W)
        return z + self.post(torch.cat([v, u[:, self.n:]], 1))


# --------------------------------------------------------------------------
# U-Net
# --------------------------------------------------------------------------

class Stage(nn.Module):
    def __init__(self, block, dim, depth, emb_dim, drop_paths, **kw):
        super().__init__()
        self.blocks = nn.ModuleList([block(dim, drop_path=drop_paths[i], **kw) for i in range(depth)])
        self.film = nn.Linear(emb_dim, 2 * dim)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, emb):
        gamma, beta = self.film(emb).chunk(2, 1)
        film = (gamma[..., None, None], beta[..., None, None])
        for blk in self.blocks:
            x = blk(x, film)
        return x


class NVSUNet(nn.Module):
    def __init__(self, block="clifford", pose="rotor", dims=(48, 96, 144, 192),
                 depths=(2, 2, 3, 2), emb_dim=128, shifts=(1, 2, 4), mlp_ratio=2.0,
                 drop_path=0.05, ls_init=1e-2):
        super().__init__()
        if block not in BLOCKS or pose not in ("matrix", "rotor"):
            raise ValueError(f"Configuración no válida: block={block}, pose={pose}")
        Block, kw = BLOCKS[block], dict(shifts=shifts, mlp_ratio=mlp_ratio, ls_init=ls_init)
        self.pose_mode = pose
        self.pose = PoseEncoder(7 if pose == "rotor" else 12, emb_dim)

        n_stages = len(dims)
        total = 2 * sum(depths[:-1]) + depths[-1]
        dpr = torch.linspace(0, drop_path, total).tolist()
        k = 0

        self.stem = nn.Sequential(nn.Conv2d(3, dims[0], 3, padding=1), nn.BatchNorm2d(dims[0]))
        self.encoder, self.down = nn.ModuleList(), nn.ModuleList()
        for i in range(n_stages - 1):
            self.encoder.append(Stage(Block, dims[i], depths[i], emb_dim, dpr[k:k + depths[i]], **kw))
            self.down.append(nn.Sequential(LayerNorm2d(dims[i]), nn.Conv2d(dims[i], dims[i + 1], 2, stride=2)))
            k += depths[i]
        self.middle = Stage(Block, dims[-1], depths[-1], emb_dim, dpr[k:k + depths[-1]], **kw)
        k += depths[-1]
        self.transport = RotorTransport(dims[-1]) if pose == "rotor" else None

        self.up, self.fuse, self.decoder = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in reversed(range(n_stages - 1)):
            self.up.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(dims[i + 1], dims[i], 3, padding=1)))
            self.fuse.append(nn.Conv2d(2 * dims[i], dims[i], 1))
            self.decoder.append(Stage(Block, dims[i], depths[i], emb_dim, dpr[k:k + depths[i]], **kw))
            k += depths[i]
        self.head = nn.Sequential(LayerNorm2d(dims[0]), nn.Conv2d(dims[0], 3, 3, padding=1))

    def encode_pose(self, T):
        R, t = T[:, :3, :3].float(), T[:, :3, 3].float()
        if self.pose_mode == "rotor":
            q = rotmat_to_rotor(R)
            return self.pose(torch.cat([q, t], 1)), rotor_to_rotmat(q)
        return self.pose(torch.cat([R.flatten(1), t], 1)), None

    def forward(self, img, T_rel):
        emb, R_q = self.encode_pose(T_rel)
        x = self.stem(img * 2 - 1)
        skips = []
        for stage, down in zip(self.encoder, self.down):
            x = stage(x, emb)
            skips.append(x)
            x = down(x)
        x = self.middle(x, emb)
        if self.transport is not None:
            x = self.transport(x, R_q)
        for up, fuse, stage in zip(self.up, self.fuse, self.decoder):
            x = stage(fuse(torch.cat([up(x), skips.pop()], 1)), emb)
        return torch.sigmoid(self.head(x).float())


def build_model(model_cfg):
    cfg = dict(model_cfg)
    for key in ("dims", "depths", "shifts"):
        if key in cfg:
            cfg[key] = tuple(cfg[key])
    return NVSUNet(**cfg)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
