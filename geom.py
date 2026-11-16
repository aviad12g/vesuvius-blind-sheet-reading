import numpy as np, tifffile
from scipy import ndimage as ndi
STEP = 20
def geometry(d, cols=None, rows=None):
    """tifxyz mesh dir -> P (H,W,3 zyx), N (H,W,3 unit normal), valid (H,W) at full resolution (20 px per node)"""
    g = {c: tifffile.imread(f'{d}/{c}.tif').astype(np.float64) for c in 'xyz'}
    if cols: g = {c: v[:, cols[0]:cols[1]] for c, v in g.items()}
    if rows: g = {c: v[rows[0]:rows[1]] for c, v in g.items()}
    gm = np.ones(g['x'].shape, bool)
    for c in 'xyz': gm &= np.isfinite(g[c]) & (g[c] != -1)
    return from_grids(g['z'], g['y'], g['x'], gm) + (gm,)

def read_grids(d, cols=None, rows=None):
    g = {c: tifffile.imread(f'{d}/{c}.tif').astype(np.float64) for c in 'xyz'}
    if cols: g = {c: v[:, cols[0]:cols[1]] for c, v in g.items()}
    if rows: g = {c: v[rows[0]:rows[1]] for c, v in g.items()}
    gm = np.ones(g['x'].shape, bool)
    for c in 'xyz': gm &= np.isfinite(g[c]) & (g[c] != -1)
    return g['z'], g['y'], g['x'], gm

def node_normals(gz, gy, gx):
    dzr, dzc = np.gradient(gz); dyr, dyc = np.gradient(gy); dxr, dxc = np.gradient(gx)
    n = np.cross(np.stack([dzc, dyc, dxc], -1), np.stack([dzr, dyr, dxr], -1)); return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12)

def from_grids(gz, gy, gx, gm):
    """node grids (zyx coordinate arrays + validity) -> P, N, valid at full resolution"""
    _, (ii, jj) = ndi.distance_transform_edt(~gm, return_indices=True)
    gx, gy, gz = gx[ii, jj], gy[ii, jj], gz[ii, jj]
    NH, NW = gx.shape; H, W = NH * STEP, NW * STEP
    RR, CC = np.meshgrid(np.clip(np.arange(H) / STEP, 0, NH - 1), np.clip(np.arange(W) / STEP, 0, NW - 1), indexing='ij')
    up = lambda a: ndi.map_coordinates(a, [RR, CC], order=1, mode='nearest').astype(np.float32)
    P = np.stack([up(gz), up(gy), up(gx)], -1)
    valid = ndi.map_coordinates(gm.astype(np.float32), [RR, CC], order=0, mode='nearest') > 0
    n = node_normals(gz, gy, gx)
    N = np.stack([up(n[..., k]) for k in range(3)], -1); N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-12
    return P, N.astype(np.float32), valid
