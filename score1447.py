"""Score reads of PHerc1447 against the public labels of windings w058 / w060 (YoussefMoNader/ink-8um-pherc1447-surfaces).
usage: score1447.py <out root> <hf dir> <meshdir> [remesh_dir remesh_out]
  <out root>/A_{a,b}/t1447_w058{a,b}/maps/c00_{model}_{face}.png     team surface through our code (base-only)
  <out root>/B_{a,b}/t1447_w058{a,b}/{canv.npz, maps/cNN_v3_{face}.png}  blind tracking, every sheet in the slab
  <out root>/C_{a,b}/...                                              v8 on selected canvases (optional)
Every map is at 1/4 resolution (0 = off-sheet, 1..255 = probability).  Writes <out root>/score.json, sel.json and figures."""
import numpy as np, tifffile, sys, os, json, glob
from PIL import Image
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom
Image.MAX_IMAGE_PIXELS = None
root, hf, meshdir = sys.argv[1:4]; RM = sys.argv[4:6] if len(sys.argv) > 5 else None
COLS = (33, 222); BANDS = {'a': (35, 128), 'b': (128, 220)}; DS = 4; UM = 8.64; R8 = 8.0
fig = f'{root}/fig'; os.makedirs(fig, exist_ok=True)

def auc(p, y):
    r = rankdata(p); P = y.sum(); Nn = len(y) - P
    return float((r[y == 1].sum() - P * (P + 1) / 2) / (P * Nn)) if P > 0 and Nn > 0 else float('nan')
def ap(p, y):
    if y.sum() == 0: return float('nan')
    vals, inv = np.unique(p, return_inverse=True); tpv = np.bincount(inv, weights=y, minlength=len(vals))[::-1]; nv = np.bincount(inv, minlength=len(vals))[::-1]
    tp = np.cumsum(tpv); n = np.cumsum(nv); prec = tp / n; rec = tp / tp[-1]; return float((np.diff(np.concatenate([[0], rec])) * prec).sum())
def blk(a, f=DS, fn=np.mean):
    h, w = a.shape[0] // f * f, a.shape[1] // f * f; return fn(fn(a[:h, :w].reshape(h // f, f, w // f, f), axis=3), axis=1)
def img(p): return np.asarray(Image.open(p).convert('L'))

# ---------------------------------------------------------------- team windings (targets)
T = {}
for w in ('w058', 'w060', 'w062'):
    g = np.stack([tifffile.imread(f'{hf}/{w}/tifxyz/{c}.tif') for c in 'zyx'], -1).astype(np.float32); ok = np.isfinite(g).all(-1) & (g > 0).all(-1)
    v8 = blk(img(f'{hf}/{w}/predictions/v8in.png').astype(np.float32)); h, wd = min(g.shape[0], v8.shape[0]), min(g.shape[1], v8.shape[1])
    d = dict(xyz=g[:h, :wd], ok=ok[:h, :wd], v8=v8[:h, :wd])
    if os.path.exists(f'{hf}/{w}/labels/inklabels.png'):
        d['lab'] = blk(img(f'{hf}/{w}/labels/inklabels.png').astype(np.float32))[:h, :wd]; d['msk'] = blk((img(f'{hf}/{w}/labels/mask.png') > 0).astype(np.float32), fn=np.min)[:h, :wd] > 0
    T[w] = d
_tp = [T[w]['xyz'][T[w]['ok']][::3] for w in T]; team_tree = cKDTree(np.concatenate(_tp)); team_id = np.concatenate([np.full(len(p_), i) for i, p_ in enumerate(_tp)])

# ---------------------------------------------------------------- our samples
S = []      # dicts: tag, P (n,3), idx (n,2) map px, maps {name: (h,w) uint8}, shape
G = {}
for b, (r0, r1) in BANDS.items():
    md = f'{meshdir}/t1447_w058{b}'
    if not os.path.exists(f'{md}/x.tif'): continue
    P, N, valid = geom.from_grids(*geom.read_grids(md)); G[b] = (P[2::DS, 2::DS], N[2::DS, 2::DS], valid[2::DS, 2::DS])
def add(tag, Pm, mapdir, ci, models):
    mp = {}
    for f in sorted(glob.glob(f'{mapdir}/c{ci:02d}_*.png')):
        k = os.path.basename(f)[4:-4]
        if k.split('_')[0] in models: mp[k] = img(f)
    if not mp: return
    on = np.zeros(next(iter(mp.values())).shape, bool)
    for a in mp.values(): on |= a > 0
    h, w = min(on.shape[0], Pm.shape[0]), min(on.shape[1], Pm.shape[1]); on = on[:h, :w]
    if on.sum() < 500: return
    yy, xx = np.nonzero(on); S.append(dict(tag=tag, P=Pm[:h, :w][on].astype(np.float32), idx=np.stack([yy, xx], 1).astype(np.int32), maps=mp, n=int(on.sum())))
INFO = {}
for b in G:
    Pb, Nb, vb = G[b]
    add(f'A_{b}', Pb, f'{root}/A_{b}/t1447_w058{b}/maps', 0, ('v3', 'v8', 'studio'))
    cz = f'{root}/B_{b}/t1447_w058{b}/canv.npz'
    if not os.path.exists(cz): continue
    z = np.load(cz); H = z['H'].astype(np.float32); D = int(z['D'])
    cz2 = f'{root}/C_{b}/t1447_w058{b}/canv.npz'
    if os.path.exists(cz2):
        H2 = np.load(cz2)['H'].astype(np.float32); same = H2.shape == H.shape and np.nanmax(np.abs(np.nan_to_num(H2) - np.nan_to_num(H))) < 0.01; print('band', b, 'tracking reproduced in the v8 pass:', bool(same))
    for ci in range(len(H)):
        Hc = H[ci]; occ = np.isfinite(Hc)
        if occ.sum() < 50: continue
        _, (ii, jj) = ndi.distance_transform_edt(~occ, return_indices=True); Hf = Hc[ii, jj]; h, w = min(Hf.shape[0], Pb.shape[0]), min(Hf.shape[1], Pb.shape[1])
        Pm = Pb[:h, :w] - (Hf[:h, :w] - D)[..., None] * Nb[:h, :w]
        n0 = len(S); add(f'B_{b}_{ci:02d}', Pm, f'{root}/B_{b}/t1447_w058{b}/maps', ci, ('v3',))
        if len(S) > n0:
            for f in sorted(glob.glob(f'{root}/C_{b}/t1447_w058{b}/maps/c{ci:02d}_v8_*.png')):
                a = img(f)
                if (a > 0).sum() > 500: S[-1]['maps'][os.path.basename(f)[4:-4]] = a
            INFO[S[-1]['tag']] = dict(median_k=float(np.nanmedian(Hc) - D), area_cm2=round(S[-1]['n'] * (DS * UM * 1e-4) ** 2, 2))
if RM:
    Pn, Nn, vn, _ = geom.geometry(RM[0]); add('R', Pn[2::DS, 2::DS], f'{RM[1]}/{os.path.basename(RM[0].rstrip("/"))}/maps', 0, ('v3', 'v8', 'studio'))
print('sample sets:', [(s['tag'], s['n'], sorted(s['maps'])) for s in S][:60])

def sample(s, key, k):                        # value of map `key` of set s at its sample k (nan when that map is off there)
    a = s['maps'].get(key)
    if a is None: return None
    y, x = s['idx'][k, 0], s['idx'][k, 1]; ok = (y < a.shape[0]) & (x < a.shape[1]); v = np.full(len(k), np.nan, np.float32)
    vv = a[y[ok], x[ok]].astype(np.float32); v[ok] = np.where(vv > 0, (vv - 1) / 254, np.nan); return v

def evaluate(name, sets, out):
    """for every labelled winding: nearest sample (within 8 voxels) among `sets` for each label pixel -> coverage, AUC, AP per map"""
    if not sets: return
    off = np.cumsum([0] + [len(s['P']) for s in sets]); tree = cKDTree(np.concatenate([s['P'] for s in sets]))
    for w in ('w058', 'w060', 'w062'):
        t = T[w]; M = (t['msk'] if 'msk' in t else (t['v8'] > 0)) & t['ok']
        if w == 'w062': M &= (np.add.outer(np.arange(M.shape[0]), np.arange(M.shape[1])) % 3 == 0)
        ty, tx = np.nonzero(M); d, i = tree.query(t['xyz'][M], k=1, distance_upper_bound=R8, workers=-1); hit = np.isfinite(d)
        res = dict(mask_px=int(M.sum()), covered=round(float(hit.mean()), 3), dist_med=round(float(np.median(d[hit])), 2) if hit.any() else None, maps={}, sets={})
        if hit.sum() < 2000: out[f'{name}/{w}'] = res; print(name, w, 'coverage %.3f - too little to score' % hit.mean()); continue
        si = np.searchsorted(off, i[hit], side='right') - 1; kk = i[hit] - off[si]
        for j, s in enumerate(sets):
            if (si == j).mean() >= 0.03: res['sets'][s['tag']] = round(float((si == j).mean()), 3)
        y = (t['lab'][M][hit] > 127).astype(np.float64) if 'lab' in t else None; ref = t['v8'][M][hit] / 255.0
        keys = sorted({k for s in sets for k in s['maps']}); canv = {}
        for key in keys:
            v = np.full(int(hit.sum()), np.nan, np.float32)
            for j, s in enumerate(sets):
                m = si == j
                if m.any():
                    sv = sample(s, key, kk[m])
                    if sv is not None: v[m] = sv
            g = np.isfinite(v)
            if g.sum() < 2000: continue
            r = dict(px=int(g.sum()), share_of_mask=round(float(g.sum() / M.sum()), 3), dark_pct=round(float((v[g] > 0.5).mean() * 100), 1), corr_team_v8=round(float(np.corrcoef(v[g], ref[g])[0, 1]), 3))
            if y is not None and 0 < y[g].sum() < g.sum():
                r.update(auc=round(auc(v[g], y[g]), 3), ap=round(ap(v[g], y[g]), 3), team_v8_auc=round(auc(ref[g], y[g]), 3), team_v8_ap=round(ap(ref[g], y[g]), 3), ink_frac=round(float(y[g].mean()), 3))
            res['maps'][key] = r; canv[key] = (v, g)
        out[f'{name}/{w}'] = res
        print(name, w, 'covered %.3f dist med %s | sets %s' % (res['covered'], res['dist_med'], res['sets']))
        for key, r in res['maps'].items(): print('    %-16s' % key, {k: r[k] for k in r if k != 'px'})
        # figure in the winding's own frame (1/4 res): label | team v8-in | our maps
        if canv:
            ys, xs = ty[hit], tx[hit]; y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1; panes = []
            if 'lab' in t: panes.append(255 - t['lab'][y0:y1, x0:x1])
            panes.append(255 - t['v8'][y0:y1, x0:x1])
            order = sorted(canv, key=lambda k: -(res['maps'][k].get('ap') or res['maps'][k]['corr_team_v8']))[:4]
            for key in order:
                v, g = canv[key]; a = np.full((y1 - y0, x1 - x0), 255, np.float32); a[ys[g] - y0, xs[g] - x0] = 255 - v[g] * 255; panes.append(a)
            sep = np.full((12, x1 - x0), 128, np.float32); st = []
            for p_ in panes: st += [p_, sep]
            Image.fromarray(np.clip(np.vstack(st[:-1]), 0, 255).astype(np.uint8)).save(f'{fig}/{name}_{w}.jpg', quality=80)
            res['figure_rows'] = (['label'] if 'lab' in t else []) + ['team v8-in (published)'] + order

OUT = {}
evaluate('team_surface_our_code', [s for s in S if s['tag'].startswith('A_')], OUT)
Bs = [s for s in S if s['tag'].startswith('B_')]
evaluate('blind_sweep', Bs, OUT)
evaluate('remesh', [s for s in S if s['tag'] == 'R'], OUT)

# ---------------------------------------------------------------- which sheets are the team's, which are not; text-row periodicity
sheets = {}
for s in Bs:
    d, i = team_tree.query(s['P'][::16], k=1, distance_upper_bound=R8, workers=-1); near = np.isfinite(d); wid = np.where(near, team_id[np.minimum(i, len(team_id) - 1)], -1)
    r = dict(INFO.get(s['tag'], {})); r['near_team'] = round(float(near.mean()), 3); r['near_by_winding'] = {w: round(float((wid == j).mean()), 3) for j, w in enumerate(T)}
    for key, a in s['maps'].items():
        on = a > 0; p = np.where(on, (a.astype(np.float32) - 1) / 254, np.nan); r[f'dark_{key}'] = round(float((p[on] > 0.5).mean() * 100), 1)
        cnt = on.sum(1); prof = np.where(cnt > 150, np.nansum(np.where(on, p, 0), 1) / np.maximum(cnt, 1), np.nan); g = np.isfinite(prof); best = 0.0; lag = 0
        if g.sum() > 400:
            q = np.where(g, prof - np.nanmean(prof), 0.0); q = q - ndi.uniform_filter1d(q, 301) * 0
            for L in range(130, 240):                                   # 4.5 - 8.3 mm at 34.56 um per map px
                m = g[:-L] & g[L:]
                if m.sum() < 300: continue
                c = np.corrcoef(q[:-L][m], q[L:][m])[0, 1]
                if c > best: best, lag = float(c), L
        r[f'rowacf_{key}'] = [round(best, 2), round(lag * DS * UM / 1000, 2)]
    sheets[s['tag']] = r
new = sorted([t for t, r in sheets.items() if r['near_team'] < 0.1 and r.get('area_cm2', 0) >= 1.0], key=lambda t: -sheets[t]['area_cm2'])
print('sheets tracked:', len(sheets), '| not on any published winding and >= 1 cm2:', len(new))
for t in sorted(sheets, key=lambda t: (t[:3], sheets[t].get('median_k', 0))): print('  ', t, sheets[t])
for t in new[:14]:                                                      # contact images of unpublished sheets (both faces, V3; v8 when present)
    s = next(x for x in Bs if x['tag'] == t); ks = sorted(s['maps']); a0 = s['maps'][ks[0]]; ys, xs = np.nonzero(a0 > 0); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    rows = [np.where(s['maps'][k][y0:y1, x0:x1] > 0, 255 - s['maps'][k][y0:y1, x0:x1], 255) for k in ks]; sep = np.full((10, x1 - x0), 128, np.uint8); st = []
    for r_ in rows: st += [r_.astype(np.uint8), sep]
    Image.fromarray(np.vstack(st[:-1])).save(f'{fig}/new_{t}.jpg', quality=75); sheets[t]['figure_rows'] = ks

# ---------------------------------------------------------------- selection for the v8 pass and the re-mesh
sel = {}
A_ = {k: v for k, v in OUT.items() if k.startswith('team_surface_our_code/w058')}
face = {}
for mdl in ('v3', 'v8', 'studio'):
    sc = {f: (A_.get('team_surface_our_code/w058', {}).get('maps', {}).get(f'{mdl}_{f}', {}).get('ap') or 0) for f in ('forward', 'reverse')}; face[mdl] = max(sc, key=sc.get) if max(sc.values()) > 0 else 'reverse'
sel['face'] = face; sel['canv'] = {}; sel['remesh'] = None
for b in BANDS:
    want = set()
    for w in ('w058', 'w060', 'w062'):
        for tag, share in OUT.get(f'blind_sweep/{w}', {}).get('sets', {}).items():
            if tag.startswith(f'B_{b}_') and share >= 0.05: want.add(int(tag[-2:]))
    for t in new[:6]:
        if t.startswith(f'B_{b}_'): want.add(int(t[-2:]))
    sel['canv'][b] = sorted(want)
s58 = OUT.get('blind_sweep/w058', {}).get('sets', {}); cb = {t: v for t, v in s58.items() if t.startswith('B_b_')}
if cb: sel['remesh'] = dict(band='b', canvas=int(max(cb, key=cb.get)[-2:]), share=max(cb.values()))
json.dump(sel, open(f'{root}/sel.json', 'w')); json.dump(dict(score=OUT, sheets=sheets, new=new, sel=sel), open(f'{root}/score.json', 'w'), indent=1)
print('SEL', json.dumps(sel)); print('SCORE_DONE')
