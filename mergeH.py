# merge fragments that the tracker filed under other sheets into one sheet: a cell joins if it lies within tol voxels of the sheet's smooth extension
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from llsm import llsmooth
src, dst, ci, tol = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
cz = np.load(src); H = cz['H'].astype(np.float32); merged = H[ci].copy(); n0 = int(np.isfinite(merged).sum())
for it in range(3):
    Hs = llsmooth(merged, 16); d = np.abs(H - Hs[None]); d[ci] = np.inf; d = np.where(np.isfinite(d), d, np.inf)
    j = d.argmin(0); best = np.take_along_axis(d, j[None], 0)[0]; take = ~np.isfinite(merged) & (best < tol)
    if take.sum() == 0: break
    merged[take] = np.take_along_axis(H, j[None], 0)[0][take]; print('pass', it, 'added', int(take.sum()), 'cells from sheets', np.unique(j[take]).tolist())
out = {k: cz[k] for k in cz.files}; Hn = cz['H'].copy(); Hn[ci] = merged.astype(Hn.dtype); out['H'] = Hn; np.savez(dst, **out)
print('sheet %d: %d -> %d cells (%.2f -> %.2f cm2)' % (ci, n0, np.isfinite(merged).sum(), n0 * (4 * 9.362e-4) ** 2, np.isfinite(merged).sum() * (4 * 9.362e-4) ** 2))
