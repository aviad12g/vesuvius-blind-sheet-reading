"""linefind.py : my own implementation of a single-line detector, following the published description of the
PHerc0306B team's method (marks -> chains of letter slots along a line -> isolation contrast). Sizes in mm.
Parameters as published: z>=2 marks, link 0.15 mm, area>=0.06 mm2, extent>=0.4; band +-1 mm, mark height<=4, length<=8,
slot gap 1.2, pitch max 6.5, >=4 slots, 8..60 mm; flanks 1.75..3.5 mm; angles -15..15 step 2.5.
score = log2(contrast) + log2(n_slots/4) - 0.5*min(crowd,2) - 2*max(0,0.85-onsheet)/0.85   (the glyph term is absent)."""
import numpy as np, cv2
Z_HI = 2.0; LINK, MIN_A, MIN_E = 0.15, 0.06, 0.4
BAND, MAX_H, MIN_H, MAX_L = 1.0, 4.0, 0.6, 8.0
SLOT, PITCH_MAX, MIN_SLOTS = 1.2, 6.5, 4
LEN_MIN, LEN_MAX = 8.0, 60.0; FL = (1.75, 3.5)
ANGLES = np.arange(-15.0, 15.01, 2.5)

def marks_of(z, valid, um):
    px = 1000.0 / um
    hot = (z >= Z_HI) & valid
    r = max(1, int(round(LINK * px / 2)))
    grp = cv2.dilate(hot.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    n, lab = cv2.connectedComponents(grp, connectivity=8); lab = np.where(hot, lab, 0)
    ys, xs = np.nonzero(lab); ids = lab[ys, xs]; o = np.argsort(ids, kind='stable'); ys, xs, ids = ys[o], xs[o], ids[o]
    cuts = np.flatnonzero(np.diff(ids)) + 1; out = []
    for yy, xx in zip(np.split(ys, cuts), np.split(xs, cuts)):
        if len(xx) == 0: continue
        area = len(xx) / px ** 2; ext = max(np.ptp(xx) + 1, np.ptp(yy) + 1) / px
        if area < MIN_A or ext < MIN_E: continue
        w = np.clip(z[yy, xx] - Z_HI, 0.05, None)
        out.append(dict(cx=np.average(xx, weights=w) / px, cy=np.average(yy, weights=w) / px, area=area,
                        xs=(xx / px).astype(np.float32), ys=(yy / px).astype(np.float32)))
    return out, lab > 0

def extents(marks, th):
    c, s = np.cos(th), np.sin(th); e = np.zeros((len(marks), 4), np.float32)
    for i, m in enumerate(marks):
        u = m['xs'] * c + m['ys'] * s; v = -m['xs'] * s + m['ys'] * c
        e[i] = (np.percentile(u, 2), np.percentile(u, 98), np.percentile(v, 3), np.percentile(v, 97))
    return e

def slots_of(idx, uc, e):
    sl = [[idx[0]]]
    for a, b in zip(idx[:-1], idx[1:]):
        if uc[b] - uc[sl[-1][-1]] < SLOT or e[b, 0] <= max(e[k, 1] for k in sl[-1]): sl[-1].append(b)
        else: sl.append([b])
    return sl

def chains_of(marks):
    if len(marks) < MIN_SLOTS: return []
    cx = np.array([m['cx'] for m in marks]); cy = np.array([m['cy'] for m in marks]); area = np.array([m['area'] for m in marks])
    found = {}
    for a in ANGLES:
        th = np.deg2rad(a); c, s = np.cos(th), np.sin(th); e = extents(marks, th)
        uc = cx * c + cy * s; vc = -cx * s + cy * c; h = e[:, 3] - e[:, 2]; l = e[:, 1] - e[:, 0]
        cand = np.flatnonzero((h <= MAX_H) & (l <= MAX_L) & (np.maximum(h, l) >= MIN_H))
        if len(cand) < MIN_SLOTS: continue
        for v0 in np.arange(vc[cand].min(), vc[cand].max() + 0.25, 0.25):
            sel = cand[np.abs(vc[cand] - v0) <= BAND]
            if len(sel) < MIN_SLOTS: continue
            sel = sel[np.argsort(uc[sel])]; sl = slots_of(list(sel), uc, e)
            sc = np.array([np.average(uc[q], weights=area[q]) for q in sl]); run = [0]
            for k in range(1, len(sl) + 1):
                if k == len(sl) or sc[k] - sc[k - 1] > PITCH_MAX:
                    if len(run) >= MIN_SLOTS:
                        ms = [i for j in run for i in sl[j]]
                        if e[ms, 1].max() - e[ms, 0].min() >= LEN_MIN:
                            key = frozenset(ms)
                            if key not in found or found[key][0] < len(run): found[key] = (len(run), a)
                    run = [k]
                else: run.append(k)
    items = sorted(found.items(), key=lambda kv: (-kv[1][0], -area[list(kv[0])].sum())); kept, used = [], []
    for key, (ns, a) in items:
        if any(len(key & u) >= 0.5 * min(len(key), len(u)) for u in used): continue
        kept.append((sorted(key), a)); used.append(key)
    return kept

def feats_of(ms, a0, marks, hot, valid, um):
    px = 1000.0 / um
    x = np.array([marks[i]['cx'] for i in ms]); y = np.array([marks[i]['cy'] for i in ms]); w = np.array([marks[i]['area'] for i in ms])
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    ev, evec = np.linalg.eigh(np.cov(np.vstack([x - mx, y - my]), aweights=w)); u = evec[:, 1]; u = -u if u[0] < 0 else u
    ang = float(np.degrees(np.arctan2(u[1], u[0])))
    if abs(ang - a0) > 3.0: ang = a0; u = np.array([np.cos(np.deg2rad(a0)), np.sin(np.deg2rad(a0))])
    vd = np.array([-u[1], u[0]]); e = extents([marks[i] for i in ms], np.deg2rad(ang)); off = mx * u[0] + my * u[1]
    uc = (x - mx) * u[0] + (y - my) * u[1]; rel = np.c_[e[:, 0] - off, e[:, 1] - off]
    sl = slots_of(list(np.argsort(uc)), uc, rel); sc = np.array([np.average(uc[q], weights=w[q]) for q in sl])
    u0, u1 = float(rel[:, 0].min()), float(rel[:, 1].max()); L = u1 - u0
    H, Wd = hot.shape
    cor = np.array([[mx + u[0] * uu + vd[0] * t, my + u[1] * uu + vd[1] * t] for uu in (u0, u1) for t in (-FL[1], FL[1])]) * px
    x0, x1 = int(max(0, cor[:, 0].min() - 2)), int(min(Wd, cor[:, 0].max() + 3)); y0, y1 = int(max(0, cor[:, 1].min() - 2)), int(min(H, cor[:, 1].max() + 3))
    yy, xx = np.mgrid[y0:y1, x0:x1]; pu = (xx / px - mx) * u[0] + (yy / px - my) * u[1]; pv = (xx / px - mx) * vd[0] + (yy / px - my) * vd[1]
    inl = (pu >= u0) & (pu <= u1); vv = valid[y0:y1, x0:x1]; hh = hot[y0:y1, x0:x1]
    band = inl & (np.abs(pv) <= BAND) & vv; fu = inl & (pv <= -FL[0]) & (pv >= -FL[1]) & vv; fd = inl & (pv >= FL[0]) & (pv <= FL[1]) & vv
    allm = inl & (np.abs(pv) <= FL[1])            # includes off-sheet px: onsheet penalises lines at the mesh edge
    hb = hh[band].mean() if band.any() else 0.0
    hu = hh[fu].mean() if fu.sum() > 20 else np.nan; hd = hh[fd].mean() if fd.sum() > 20 else np.nan
    hf = np.nanmean([hu, hd]) if not (np.isnan(hu) and np.isnan(hd)) else 0.0
    crowd = sum(1 for m in marks if u0 <= (m['cx'] - mx) * u[0] + (m['cy'] - my) * u[1] <= u1 and FL[0] <= abs((m['cx'] - mx) * vd[0] + (m['cy'] - my) * vd[1]) <= FL[1] and m['area'] >= 0.3)
    gaps = np.diff(sc); hs = e[:, 3] - e[:, 2]
    f = dict(n_slots=len(sl), len_mm=round(L, 2), angle=round(ang, 1), pitch_med=round(float(np.median(gaps)), 2) if len(gaps) else None,
             h_med=round(float(np.median(hs)), 2), contrast=round(float((hb + 0.01) / (hf + 0.01)), 3), band_hot=round(float(hb), 4),
             flank_hot=round(float(hf), 4), flanks=int(np.isfinite(hu)) + int(np.isfinite(hd)), crowd=round(crowd / len(sl), 3),
             onsheet=round(float(vv[allm].mean()) if allm.any() else 0.0, 3), cx_mm=round(float(mx), 2), cy_mm=round(float(my), 2),
             u0=round(u0, 2), u1=round(u1, 2), ux=float(u[0]), uy=float(u[1]))
    s = np.log2(max(f['contrast'], 0.25)) + np.log2(f['n_slots'] / 4.0) - 0.5 * min(f['crowd'], 2.0) - 2.0 * max(0.0, 0.85 - f['onsheet']) / 0.85
    if L > LEN_MAX: s -= 1.0
    f['score'] = round(float(s), 3)
    return f

def detect(p, um):
    """p: ink map, 0 = off-sheet. -> list of chain feature dicts, sorted by score"""
    v = p > 0
    if v.sum() < 500: return [], 0.0
    z = np.zeros_like(p, np.float32); z[v] = (p[v] - p[v].mean()) / (p[v].std() + 1e-6)
    marks, hot = marks_of(z, v, um)
    out = [feats_of(ms, a, marks, hot, v, um) for ms, a in chains_of(marks)]
    return sorted(out, key=lambda f: -f['score']), float(v.sum()) * (um / 1e4) ** 2
