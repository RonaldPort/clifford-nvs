"""Rotores de Cl(3,0) y poses de cámara.

La rotación de ángulo θ alrededor del eje unitario n corresponde al rotor
R = exp(-I n θ/2) = q0 - q1 e23 - q2 e31 - q3 e12, con
(q0, q1, q2, q3) = (cos θ/2, n sin θ/2). Los rotores unitarios forman Spin(3),
recubrimiento doble de SO(3): q y -q inducen la misma rotación. El vector
q = (q0, q1, q2, q3) usado en el código coincide con el cuaternión unitario
de la misma rotación, y la acción sándwich v ↦ R v R~ sobre el grado 1 tiene
la matriz que devuelve `rotor_to_rotmat`.
"""
import torch
import torch.nn.functional as F


def relative_pose(T_src, T_tgt):
    """Pose de la cámara objetivo en el marco de la cámara fuente: T_s^{-1} T_t.
    Las poses de SRN son matrices cámara→mundo de 4×4."""
    return torch.linalg.inv(T_src) @ T_tgt


def rotmat_to_rotor(R):
    """(..., 3, 3) → (..., 4). Método de Shepperd vectorizado: se elige la
    componente dominante para evitar divisiones por valores cercanos a cero.
    Se fija el representante canónico q0 ≥ 0."""
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    tr = m00 + m11 + m22
    cands = torch.stack([
        torch.stack([1 + tr, m21 - m12, m02 - m20, m10 - m01], -1),
        torch.stack([m21 - m12, 1 + m00 - m11 - m22, m01 + m10, m02 + m20], -1),
        torch.stack([m02 - m20, m01 + m10, 1 - m00 + m11 - m22, m12 + m21], -1),
        torch.stack([m10 - m01, m02 + m20, m12 + m21, 1 - m00 - m11 + m22], -1),
    ], -2)
    k = torch.stack([tr, m00, m11, m22], -1).argmax(-1)
    q = cands.gather(-2, k[..., None, None].expand(*k.shape, 1, 4)).squeeze(-2)
    q = F.normalize(q, dim=-1)
    return torch.where(q[..., :1] < 0, -q, q)


def rotor_to_rotmat(q):
    """Matriz de la aplicación lineal v ↦ q v q~ sobre el grado 1."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], -1).reshape(*q.shape[:-1], 3, 3)


def rotation_angle_deg(R):
    """Ángulo geodésico de la rotación, θ = 2 arccos(q0), en grados."""
    q0 = rotmat_to_rotor(R)[..., 0].clamp(-1.0, 1.0)
    return torch.rad2deg(2 * torch.arccos(q0))
