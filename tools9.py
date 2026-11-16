# remesh: turn part of a tracked sheet (one canvas of a base mesh) into its own evenly spaced tifxyz mesh (arclength along rows, orthogonal anchor curve).
# cmp: compare maps read on a mesh with the winner's published map (nearest-point registration).
import sys, os, glob, numpy as np, tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import geom
from llsm import llsmooth
cmd = sys.argv[1]
def bil(A, r, c):
    r0 = np.clip(np.floor(r).astype(int), 0, A.shape[0] - 2); c0 = np.clip(np.floor(c).astype(int), 0, A.shape[1] - 2); fr = (r - r0)[..., None] if A.ndim == 3 else r - r0; fc = (c - c0)[..., None] if A.ndim == 3 else c - c0
    return (A[r0, c0] * (1 - fc) + A[r0, c0 + 1] * fc) * (1 - fr) + (A[r0 + 1, c0] * (1 - fc) + A[r0 + 1, c0 + 1] * fc) * fr
if cmd == 'remesh':
    meshdir, canvnpz, ci, outdir = sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
    r0p, c0p, hwr, hwc = [float(x) for x in sys.argv[6:10]]                 # anchor (canvas px) and half-sizes of the region (px)
    sig_px = float(sys.argv[10]) if len(sys.argv) > 10 else 10.0; koff = float(sys.argv[11]) if len(sys.argv) > 11 else 0.0
    cz = np.load(canvnpz); H = cz['H'].astype(np.float32)[ci]; D = int(cz['D']); LS = int(cz['ls'])
    P, N, valid, _ = geom.geometry(meshdir); Pl, Nl = P[::LS, ::LS][:H.shape[0], :H.shape[1]].astype(np.float64), N[::LS, ::LS][:H.shape[0], :H.shape[1]].astype(np.float64)
    ra, rb = max(0, int((r0p - hwr) / LS)), min(H.shape[0], int((r0p + hwr) / LS) + 1); ca, cb = max(0, int((c0p - hwc) / LS)), min(H.shape[1], int((c0p + hwc) / LS) + 1)
    Hc = H[ra:rb, ca:cb]; Pc, Nc = Pl[ra:rb, ca:cb], Nl[ra:rb, ca:cb]; v = np.isfinite(Hc)
    lab, nl = ndi.label(v); a0r, a0c = int(r0p / LS) - ra, int(c0p / LS) - ca
    if nl > 1:                                                                # keep the connected piece nearest the anchor
        d = ndi.distance_transform_edt(~v, return_indices=True)[1]; keep = lab[d[0][a0r, a0c], d[1][a0r, a0c]]; v = lab == keep; Hc = np.where(v, Hc, np.nan)
    Hs = llsmooth(Hc, sig_px / LS); Hs[~v] = np.nan; X = Pc - (Hs - D + koff)[..., None] * Nc
    Hm = llsmooth(Hc, 32.0 / LS); Hm[~v] = np.nan; Xm = Pc - (Hm - D + koff)[..., None] * Nc           # smoother copy, used only to measure lengths
    h, w = Hc.shape; cols = np.arange(w, dtype=np.float64); rows = np.arange(h, dtype=np.float64)
    # rows of the new mesh are cross-sections of the sheet at constant z (scroll axis); columns are arclength along each cross-section
    zc = X[..., 0]; anc = X[a0r, a0c] if np.isfinite(X[a0r, a0c]).all() else np.nanmean(X.reshape(-1, 3), 0)
    zmin, zmax = np.nanpercentile(zc, 1), np.nanpercentile(zc, 99); i0 = int(np.ceil((zmin - anc[0]) / 20)); i1 = int(np.floor((zmax - anc[0]) / 20)); curves = []
    for i in range(i0, i1 + 1):
        zt = anc[0] + 20.0 * i; Q = np.full((w, 3), np.nan); Qm = np.full((w, 3), np.nan)
        for c in range(w):
            col = zc[:, c]; g = np.isfinite(col)
            if g.sum() < 2: continue
            rr_ = rows[g]; zz = col[g]; o = np.argsort(zz)
            if zt < zz[o[0]] or zt > zz[o[-1]]: continue
            rf = np.interp(zt, zz[o], rr_[o]); rA = int(np.clip(np.floor(rf), 0, h - 2)); fr = rf - rA
            if not (v[rA, c] and v[rA + 1, c]): continue
            Q[c] = X[rA, c] * (1 - fr) + X[rA + 1, c] * fr; Qm[c] = Xm[rA, c] * (1 - fr) + Xm[rA + 1, c] * fr
        g = np.isfinite(Q[:, 0]) & np.isfinite(Qm[:, 0])
        if g.sum() < 3: curves.append(None); continue
        cg = cols[g]; seg = np.linalg.norm(np.diff(Qm[g], axis=0), axis=1); sarc = np.concatenate([[0], np.cumsum(seg)])
        gap = np.diff(cg) > 12                                                           # do not bridge big holes
        Qa = Qm[g][:, 1:]; k0 = int(np.argmin(np.linalg.norm(Qa - anc[1:], axis=1))); ka, kb = max(k0 - 2, 0), min(k0 + 2, len(Qa) - 1)
        tau = Qa[kb] - Qa[ka]; tau = tau / max(np.linalg.norm(tau), 1e-9)                   # curve tangent near the anchor (points towards increasing arclength)
        sarc = sarc - (sarc[k0] + float(((anc[1:] - Qa[k0]) * tau).sum()))                # first-order foot of the anchor; also right when the curve stops short of it
        curves.append((cg, sarc, Q[g], gap))
    allS = np.concatenate([c_[1] for c_ in curves if c_ is not None]); j0 = int(np.ceil(allS.min() / 20)); j1 = int(np.floor(allS.max() / 20)); ni, nj = i1 - i0 + 1, j1 - j0 + 1
    Gx = np.full((3, ni, nj), np.nan, np.float32); tg = (j0 + np.arange(nj)) * 20.0
    def row(i):
        c_ = curves[i]; out = np.full((3, nj), np.nan)
        if c_ is None: return out
        cg, sarc, Qg, gap = c_
        for a in range(3): out[a] = np.interp(tg, sarc, Qg[:, a], left=np.nan, right=np.nan)
        for kk in np.nonzero(gap)[0]: out[:, (tg > sarc[kk]) & (tg < sarc[kk + 1])] = np.nan
        return out
    def slip(prev, cur):                                                                 # median sideways displacement of a row against its neighbour, along the neighbour's direction
        a_, b_ = prev[1:], cur[1:]; tan = np.full_like(a_, np.nan); tan[:, 1:-1] = a_[:, 2:] - a_[:, :-2]; dj = ((b_ - a_) * tan).sum(0) / np.maximum(np.linalg.norm(tan, axis=0), 1e-9)
        return float(np.nanmedian(dj)) if np.isfinite(dj).sum() > 5 else None
    ia = int(np.clip(-i0, 0, ni - 1)); Gx[:, ia] = row(ia); nfix = 0
    for order in (range(ia + 1, ni), range(ia - 1, -1, -1)):                             # walk outwards; each row is lined up square to the one before it
        last = ia
        for i in order:
            if curves[i] is None: continue
            for it in range(3):
                cur = row(i); m_ = slip(Gx[:, last], cur) if np.isfinite(Gx[:, last]).any() else None
                if m_ is None or abs(m_) < 0.05: break
                c_ = curves[i]; curves[i] = (c_[0], c_[1] + m_, c_[2], c_[3]); nfix += 1
            Gx[:, i] = row(i)
            if np.isfinite(Gx[0, i]).sum() > 5: last = i
    print('region cells', (h, w), 'valid %.0f%%' % (v.mean() * 100), '| row alignments applied:', nfix)
    bad = ~np.isfinite(Gx).all(0); Gx[:, bad] = -1; ok = ~bad
    os.makedirs(outdir, exist_ok=True)
    for a, nm in enumerate('zyx'): tifffile.imwrite(f'{outdir}/{nm}.tif', Gx[a].astype(np.float32))
    import json
    vv = Gx[0] > 0; name = os.path.basename(os.path.normpath(outdir))
    json.dump(dict(uuid=name, name=name, type='seg', format='tifxyz', scale=[0.05, 0.05],
                   bbox=[[float(Gx[a][vv].min()) for a in (2, 1, 0)], [float(Gx[a][vv].max()) for a in (2, 1, 0)]],
                   area_vx2=float(vv.sum()) * 400.0, source='tools9.py remesh (automatic, no hand refinement)'),
              open(f'{outdir}/meta.json', 'w'), indent=1)
    dc = (Gx[:, :, 1:] - Gx[:, :, :-1]); dr = (Gx[:, 1:] - Gx[:, :-1]); mc = ok[:, 1:] & ok[:, :-1]; mr = ok[1:] & ok[:-1]
    stc = np.linalg.norm(dc, axis=0)[mc]; str_ = np.linalg.norm(dr, axis=0)[mr]; m4 = mc[:-1] & mr[:, :-1]
    cosang = ((dc[:, :-1] * dr[:, :, :-1]).sum(0) / (np.linalg.norm(dc[:, :-1], axis=0) * np.linalg.norm(dr[:, :, :-1], axis=0) + 1e-9))[m4]
    print('new mesh nodes', (ni, nj), 'valid', int(ok.sum()), '= %.2f cm2' % (ok.sum() * (20 * 9.362e-4) ** 2), '| node step along rows: median %.2f (p5 %.2f p95 %.2f) | down columns: median %.2f (p5 %.2f p95 %.2f) | axes angle off 90deg: median %.1f, p95 %.1f deg' % (np.median(stc), np.percentile(stc, 5), np.percentile(stc, 95), np.median(str_), np.percentile(str_, 5), np.percentile(str_, 95), np.median(np.abs(np.degrees(np.arcsin(np.clip(cosang, -1, 1))))), np.percentile(np.abs(np.degrees(np.arcsin(np.clip(cosang, -1, 1)))), 95)))
elif cmd == 'cmp':
    newmesh, mapglob, wdir, prob, outpng = sys.argv[2:7]
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    Pn, Nn, vn, _ = geom.geometry(newmesh); Pw, Nw, vw, _ = geom.geometry(wdir)
    ref = np.nan_to_num(tifffile.imread(prob)).astype(np.float32)
    yy, xx = np.nonzero(vn); tree = cKDTree(Pn[vn].astype(np.float64))
    wy, wx = np.nonzero(vw[::4, ::4]); Q = Pw[::4, ::4][vw[::4, ::4]].astype(np.float64)
    dist, i = tree.query(Q, k=4, distance_upper_bound=8); hit = np.isfinite(dist).all(1); i = np.where(np.isfinite(dist), i, 0); wgt = 1.0 / (dist[hit] + 0.3) ** 2; wgt /= wgt.sum(1, keepdims=True)
    print('his surface cells with the new mesh within 8 voxels: %.0f%% | distance median %.2f' % (hit.mean() * 100, np.median(dist[hit][:, 0])))
    h4, w4 = vw[::4, ::4].shape; MY = np.full((h4, w4), np.nan, np.float32); MX = MY.copy(); MY[wy[hit], wx[hit]] = (yy[i[hit]] * wgt).sum(1); MX[wy[hit], wx[hit]] = (xx[i[hit]] * wgt).sum(1)
    val = np.isfinite(MY); _, (a, b) = ndi.distance_transform_edt(~val, return_indices=True); MYs = ndi.gaussian_filter(MY[a, b], 1.5); MXs = ndi.gaussian_filter(MX[a, b], 1.5)
    FH, FW = ref.shape; gy, gx = np.meshgrid(np.arange(FH) / 4.0, np.arange(FW) / 4.0, indexing='ij')
    py = ndi.map_coordinates(MYs, [gy, gx], order=1, mode='nearest'); px = ndi.map_coordinates(MXs, [gy, gx], order=1, mode='nearest')
    vm = ndi.map_coordinates(ndi.binary_erosion(val, iterations=3).astype(np.float32), [gy, gx], order=1, mode='constant') > 0.99; best = None
    LR = (np.arange(FH)[:, None] > 560) & (np.arange(FH)[:, None] < 1160) & (np.arange(FW)[None] > 3900) & (np.arange(FW)[None] < 5440)
    for f in sorted(glob.glob(mapglob)):
        A = np.asarray(Image.open(f)).astype(np.float32)
        sA = ndi.map_coordinates(A, [py, px], order=1, mode='constant'); on = ndi.map_coordinates((A > 0).astype(np.float32), [py, px], order=1, mode='constant') > 0.99
        M = vm & on & (ref > 0); p = np.where(M, (sA - 1) / 254, 0)
        if M.sum() < 1000: print(os.path.basename(f), 'no overlap'); continue
        c0 = np.corrcoef(p[M], ref[M])[0, 1]; L = M & LR; cl = np.corrcoef(p[L], ref[L])[0, 1] if L.sum() > 1000 else float('nan')
        print('%-30s corr with his: all %.3f (%.2f cm2) | clearest-letter region %.3f (%.2f cm2) | dark%% ours %.1f his %.1f' % (os.path.basename(f)[:-4], c0, M.sum() * (9.362e-4) ** 2, cl, L.sum() * (9.362e-4) ** 2, (p[M] > 0.5).mean() * 100, (ref[M] > 0.5).mean() * 100))
        sc = cl if cl == cl else c0
        if best is None or sc > best[0]: best = (sc, p, M)
    if best:
        _, p, M = best; M2 = M & LR if (M & LR).sum() > 1000 else M; ys, xs = np.nonzero(M2); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        top = 255 - np.clip(np.where(M, ref, 0)[y0:y1, x0:x1] / 0.95, 0, 1) * 255; bot = 255 - np.clip(p[y0:y1, x0:x1], 0, 1) * 255
        Image.fromarray(np.vstack([top, np.full((24, x1 - x0), 255), bot]).astype(np.uint8)).save(outpng); print('saved', outpng)
