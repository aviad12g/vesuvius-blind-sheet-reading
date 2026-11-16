"""slabtrack.py : find every papyrus sheet inside a thick slab around a base mesh.

Frame: base mesh point P(i,j), unit normal N(i,j); slab voxel (k,i,j) is the scroll point P - k*N. In this frame the sheets
are near-horizontal layers. The organisers' m7 surface prediction (binary, thin) sampled into the slab gives, per column,
runs along k; run centres are linked sideways into single-valued height fields k = h(i,j) ("patches"), which are then
packed into non-overlapping canvases (one canvas ~ one sheet) for rendering and ink inference.
"""
import numpy as np
from scipy import ndimage as ndi

def run_centres(M, max_thick=14):
    """M bool (K,h,w) -> cols (n,), centre (n,) float32, thick (n,) for every run along k (runs thicker than max_thick dropped)"""
    K, h, w = M.shape; M2 = M.reshape(K, -1).T
    st = M2.copy(); st[:, 1:] &= ~M2[:, :-1]; en = M2.copy(); en[:, :-1] &= ~M2[:, 1:]
    cs, ks = np.nonzero(st); ce, ke = np.nonzero(en)
    assert len(ks) == len(ke)
    thick = ke - ks + 1; ok = thick <= max_thick
    return cs[ok], ((ks + ke) / 2.0).astype(np.float32)[ok], thick[ok]

def _edges(cols, cen, h, w, tau2):
    """mutually-nearest run pairs in sideways-neighbouring columns, at most tau2 voxels apart. cols/cen sorted by (col, cen)."""
    BIG = 8192.0; key = cols.astype(np.float64) * BIG + cen; n = len(cols); out = []
    def nearest(shift):                      # for every run: index of the nearest run in column (col + shift), or -1
        q = (cols + shift).astype(np.float64) * BIG + cen; pos = np.searchsorted(key, q)
        lo = np.clip(pos - 1, 0, n - 1); hi = np.clip(pos, 0, n - 1)
        dlo = np.where(cols[lo] == cols + shift, np.abs(cen[lo] - cen), np.inf); dhi = np.where(cols[hi] == cols + shift, np.abs(cen[hi] - cen), np.inf)
        b = np.where(dlo <= dhi, lo, hi); d = np.minimum(dlo, dhi); b[~np.isfinite(d)] = -1
        return b, d
    for dc in (1, w):
        okc = (cols % w < w - 1) if dc == 1 else (cols // w < h - 1)
        fwd, d = nearest(dc); back, _ = nearest(-dc)
        a = np.flatnonzero(okc & (fwd >= 0) & (d <= tau2)); b = fwd[a]
        m = back[b] == a; out.append((a[m], b[m]))
    return np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out])

def link(cols, cen, shape, tau=2.5, tau2=3.0, min_cells=3000, merge_min=8):
    """group run centres into single-valued patches. shape = (K,h,w). -> list of float32 (h,w) height maps (nan outside).

    1. Labels spread column to column over a best-first spanning tree: a column's runs take the labels of the mutually
       nearest runs (within tau voxels) of an already labelled neighbour column (the tree parent first, then any other
       labelled neighbour); unmatched runs get fresh labels. A label never appears twice in one column, whatever bridges
       or merges the sheets have.
    2. Each label is split into pieces that are continuous sideways.
    3. Pieces of different labels that touch along a seam (>= merge_min continuous run pairs) and share no column are
       stitched back together, largest contact first."""
    import heapq
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    K, h, w = shape; ncol = h * w
    o = np.lexsort((cen, cols)); cols, cen = cols[o], cen[o]
    start = np.searchsorted(cols, np.arange(ncol)); end = np.searchsorted(cols, np.arange(ncol), 'right'); nrun = end - start
    lab = np.full(len(cols), -1, np.int64); nxt = 0; done = nrun == 0

    def match(pc, cc):
        kp = cen[start[pc]:end[pc]]; kc = cen[start[cc]:end[cc]]
        if len(kp) == 1: ip = np.zeros(len(kc), np.int64)
        else:
            ip = np.clip(np.searchsorted(kp, kc), 1, len(kp) - 1); ip = np.where(np.abs(kc - kp[ip - 1]) <= np.abs(kc - kp[ip]), ip - 1, ip)
        if len(kc) == 1: ic = np.zeros(len(kp), np.int64)
        else:
            ic = np.clip(np.searchsorted(kc, kp), 1, len(kc) - 1); ic = np.where(np.abs(kp - kc[ic - 1]) <= np.abs(kp - kc[ic]), ic - 1, ic)
        ok = (np.abs(kc - kp[ip]) <= tau) & (ic[ip] == np.arange(len(kc)))
        return ip, ok

    def nbrs(c):
        i, j = divmod(c, w); r = []
        if i > 0: r.append(c - w)
        if i < h - 1: r.append(c + w)
        if j > 0: r.append(c - 1)
        if j < w - 1: r.append(c + 1)
        return r

    remaining = np.argsort(-nrun); rp = 0; heap = []; cnt = 0
    while True:
        while rp < len(remaining) and done[remaining[rp]]: rp += 1
        if rp >= len(remaining): break
        s = remaining[rp]; lab[start[s]:end[s]] = np.arange(nxt, nxt + nrun[s]); nxt += nrun[s]; done[s] = True
        for q in nbrs(s):
            if not done[q]:
                ip, ok = match(s, q); cnt += 1; heapq.heappush(heap, (-int(ok.sum()), cnt, q, s))
        while heap:
            _, _, c, p = heapq.heappop(heap)
            if done[c]: continue
            ip, ok = match(p, c); l = np.where(ok, lab[start[p]:end[p]][ip], -1)
            if not ok.all():
                for q in nbrs(c):
                    if q != p and done[q] and nrun[q]:
                        ip2, ok2 = match(q, c); cand = lab[start[q]:end[q]][ip2]
                        take = (l < 0) & ok2 & ~np.isin(cand, l[l >= 0])
                        l[take] = cand[take]
                        if (l >= 0).all(): break
            nb = int((l < 0).sum()); l[l < 0] = np.arange(nxt, nxt + nb); nxt += nb; lab[start[c]:end[c]] = l; done[c] = True
            for q in nbrs(c):
                if not done[q]:
                    ip2, ok2 = match(c, q); cnt += 1; heapq.heappush(heap, (-int(ok2.sum()), cnt, q, c))
    ea, eb = _edges(cols, cen, h, w, tau2)
    same = lab[ea] == lab[eb]
    n, comp = connected_components(coo_matrix((np.ones(int(same.sum()), bool), (ea[same], eb[same])), shape=(len(cols), len(cols))), directed=False)
    sizes = np.bincount(comp, minlength=n)
    # 3. stitch pieces across label seams
    ca, cb = comp[ea[~same]], comp[eb[~same]]; big = (sizes[ca] >= 50) & (sizes[cb] >= 50) & (ca != cb); ca, cb = ca[big], cb[big]
    lo, hi = np.minimum(ca, cb), np.maximum(ca, cb); pk, pc = np.unique(lo.astype(np.int64) * n + hi, return_counts=True)
    root = np.arange(n)
    def find(x):
        while root[x] != x:
            root[x] = root[root[x]]; x = root[x]
        return x
    order = np.argsort(comp, kind='stable'); bounds = np.searchsorted(comp[order], np.arange(n + 1))
    colset = {}
    def cols_of(r):
        if r not in colset: colset[r] = np.sort(cols[order[bounds[r]:bounds[r + 1]]])
        return colset[r]
    for q in np.argsort(-pc):
        if pc[q] < merge_min: break
        a, b = find(int(pk[q] // n)), find(int(pk[q] % n))
        if a == b: continue
        A, B = cols_of(a), cols_of(b)
        if len(np.intersect1d(A, B, assume_unique=True)): continue
        root[b] = a; colset[a] = np.sort(np.concatenate([A, B])); colset.pop(b, None)
    comp = np.array([find(int(c)) for c in range(n)])[comp]
    sizes = np.bincount(comp, minlength=n); keep = np.flatnonzero(sizes >= min_cells); out = []
    for q in keep[np.argsort(-sizes[keep])]:
        m = comp == q; hm = np.full(ncol, np.nan, np.float32); hm[cols[m]] = cen[m]; out.append(hm.reshape(h, w))
    return out

def smooth_fill(hm, sigma=1.5, grow=2):
    """normalised gaussian smoothing of a nan-holed height map; fills holes/edges up to ~grow cells. -> (h, mask)"""
    v = np.isfinite(hm); a = np.where(v, hm, 0).astype(np.float32)
    num = ndi.gaussian_filter(a, sigma); den = ndi.gaussian_filter(v.astype(np.float32), sigma)
    m = ndi.binary_closing(v, iterations=grow) | v
    m = ndi.binary_fill_holes(m) & (den > 0.15)
    out = np.where(m, num / np.maximum(den, 1e-6), np.nan).astype(np.float32)
    return out, m

def pack(patches, gap=1):
    """first-fit packing of (h, mask) patches into canvases with disjoint footprints. -> list of (H float32 nan-outside, n_patches)"""
    order = sorted(range(len(patches)), key=lambda q: np.nanmedian(patches[q][0]))
    canv = []
    for q in order:
        hm, m = patches[q]; md = ndi.binary_dilation(m, iterations=gap)
        for c in canv:
            if not (c['occ'] & md).any():
                c['H'][m] = hm[m]; c['occ'] |= m; c['n'] += 1; break
        else:
            H = np.full(hm.shape, np.nan, np.float32); H[m] = hm[m]; canv.append(dict(H=H, occ=m.copy(), n=1))
    return canv
