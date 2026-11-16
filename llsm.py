import numpy as np
from scipy import ndimage as ndi
def llsmooth(Hin, sigma, mask=None):
    """Local-linear (Gaussian-weighted plane fit) smoothing of a height field with holes. sigma in cells. Unbiased for tilted sheets and at edges."""
    v = np.isfinite(Hin) if mask is None else (mask & np.isfinite(Hin)); h, w = Hin.shape
    Y, X = np.mgrid[0:h, 0:w].astype(np.float64); Z = np.where(v, Hin, 0).astype(np.float64); W = v.astype(np.float64)
    g = lambda a: ndi.gaussian_filter(a, sigma, mode='constant')
    S0, Sy, Sx, Syy, Sxy, Sxx = g(W), g(W * Y), g(W * X), g(W * Y * Y), g(W * X * Y), g(W * X * X)
    Sz, Syz, Sxz = g(Z), g(Z * Y), g(Z * X)
    my, mx = Sy / np.maximum(S0, 1e-9), Sx / np.maximum(S0, 1e-9)                      # centre the regression locally
    cyy = Syy - S0 * my * my; cxy = Sxy - S0 * mx * my; cxx = Sxx - S0 * mx * mx
    cyz = Syz - my * Sz; cxz = Sxz - mx * Sz
    det = cyy * cxx - cxy * cxy; ok = det > 1e-6 * np.maximum(cyy * cxx, 1e-9)
    by = np.where(ok, (cyz * cxx - cxz * cxy) / np.where(ok, det, 1), 0); bx = np.where(ok, (cxz * cyy - cyz * cxy) / np.where(ok, det, 1), 0)
    out = Sz / np.maximum(S0, 1e-9) + by * (Y - my) + bx * (X - mx)
    return np.where(S0 > 0.05, out, np.nan).astype(np.float32)
