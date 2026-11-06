"""Métricas por imagen: PSNR, SSIM y LPIPS. Todas reciben tensores en [0, 1]."""
import time

import torch
import torch.nn.functional as F


def psnr(pred, target):
    mse = (pred - target).pow(2).flatten(1).mean(1).clamp_min(1e-10)
    return -10.0 * torch.log10(mse)


def _gaussian_window(size, sigma, channels, device, dtype):
    g = torch.exp(-(torch.arange(size, device=device, dtype=dtype) - size // 2) ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None]).expand(channels, 1, size, size).contiguous()


def ssim(pred, target, size=11, sigma=1.5, c1=0.01 ** 2, c2=0.03 ** 2):
    """SSIM con ventana gaussiana 11×11, σ = 1.5 (Wang et al., 2004)."""
    w = _gaussian_window(size, sigma, pred.shape[1], pred.device, pred.dtype)
    blur = lambda x: F.conv2d(x, w, groups=pred.shape[1])
    mu_p, mu_t = blur(pred), blur(target)
    var_p = blur(pred * pred) - mu_p ** 2
    var_t = blur(target * target) - mu_t ** 2
    cov = blur(pred * target) - mu_p * mu_t
    s = ((2 * mu_p * mu_t + c1) * (2 * cov + c2)) / ((mu_p ** 2 + mu_t ** 2 + c1) * (var_p + var_t + c2))
    return s.flatten(1).mean(1)


class LPIPS:
    """Envoltura de LPIPS (Zhang et al., 2018). Se entrena con la variante
    AlexNet y se evalúa con VGG para no medir con la misma red que guía la pérdida."""

    def __init__(self, net, device):
        import lpips
        self.model = lpips.LPIPS(net=net, verbose=False).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def __call__(self, pred, target):
        return self.model(pred * 2 - 1, target * 2 - 1).flatten()


@torch.no_grad()
def throughput(model, device, batch=64, res=64, iters=30, amp=True):
    """Imágenes por segundo en inferencia."""
    x = torch.rand(batch, 3, res, res, device=device)
    T = torch.eye(4, device=device).expand(batch, 4, 4).contiguous()
    use_amp = amp and device == "cuda"
    was_training = model.training
    model.eval()
    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        for _ in range(5):
            model(x, T)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x, T)
        if device == "cuda":
            torch.cuda.synchronize()
    model.train(was_training)
    return batch * iters / (time.perf_counter() - t0)
