"""sweep.py : read EVERY papyrus sheet around each base mesh of a scroll with TRACE v3B (both faces).

For each base mesh (a spiral-fit winding, tifxyz, 20 voxels per node):
  1. frame     P(i,j), N(i,j) from the mesh; slab voxel (k,i,j) = P - k*N, k in [-D, D].
  2. sheets    the organisers' m7 surface prediction is sampled into the slab (every LS px sideways, every voxel in depth);
               slabtrack links its runs into single-valued height fields and packs them into canvases (one ~ one sheet).
  3. read      per canvas: 17 CT layers (-8..+8 voxels about the tracked surface, along the frame normal), TRACE v3B
               (306B zoomlib.villa_map, overlap 0.5, no TTA, d = 0) forward and reverse, in column tiles with margins.
  4. save      1/4-resolution maps (uint8, 0 = off-sheet), the surface CT at 1/4 resolution, the height fields, and the
               line-detector chains of every map.
Restartable (done.json per mesh).  --shard i/n splits the mesh list across pods.  --control306 runs the positive control:
the PHerc0306B winners' surface moved ~3 sheets away and warped, which must come back with their line on top.
"""
import os, sys, json, time, argparse, glob
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--root', default='/workspace/s3'); ap.add_argument('--scroll', default='PHerc0800'); ap.add_argument('--meshdir', default='')
ap.add_argument('--only', default=''); ap.add_argument('--shard', default='0/1'); ap.add_argument('--out', default='')
ap.add_argument('--D', type=int, default=260); ap.add_argument('--ls', type=int, default=4)
ap.add_argument('--tile', type=int, default=768); ap.add_argument('--marg', type=int, default=96)
ap.add_argument('--max-minutes', type=float, default=600); ap.add_argument('--dry', action='store_true')
ap.add_argument('--studio-faces', default='forward,reverse'); ap.add_argument('--trace-faces', default='forward,reverse'); ap.add_argument('--claim', action='store_true'); ap.add_argument('--desc', action='store_true')
ap.add_argument('--ds', type=int, default=4); ap.add_argument('--canv', default=''); ap.add_argument('--xrange', default='')
ap.add_argument('--flatten', action='store_true'); ap.add_argument('--koffs', default='0'); ap.add_argument('--fl-s', type=int, default=6)
ap.add_argument('--base-only', action='store_true'); ap.add_argument('--rows', default=''); ap.add_argument('--cols', default=''); ap.add_argument('--min-mm2', type=float, default=8.0)
ap.add_argument('--models', default='trace,studio'); ap.add_argument('--studio-stride', type=int, default=96); ap.add_argument('--batch', type=int, default=32)
ap.add_argument('--ccols', default='115:235'); ap.add_argument('--control306', action='store_true'); ap.add_argument('--device', default='cuda'); ap.add_argument('--threads', type=int, default=48)
A = ap.parse_args(); R = A.root; T0 = time.time()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn.functional as F
from scipy import ndimage as ndi
from PIL import Image
import vol, geom, slabtrack as st, linefind as lf

DEV = A.device if torch.cuda.is_available() else 'cpu'
LS = A.ls; D = A.D; TILE = A.tile; MARG = A.marg; LAYERS = np.arange(-12, 13); C = 12; DS = A.ds; CANV = [int(v) for v in A.canv.split(',')] if A.canv else []; XR = [int(v) for v in A.xrange.split(':')] if A.xrange else None
assert TILE % (DS * LS) == 0 and MARG % DS == 0
def _sc(name, scan, um, kev): return dict(ct=f'{name}/volumes/{scan}-{um:.3f}um-1.2m-{kev}keV-masked.zarr', m7=f'{name}/representations/predictions/surfaces/{scan}-surface-20260413222639-surface-m7-L0-th0.2.zarr', um=um)
SCROLLS = {'PHerc0268': _sc('PHerc0268', '20251110183117', 8.64, 116), 'PHerc0257': _sc('PHerc0257', '20250821151750', 9.362, 113), 'PHerc0125': _sc('PHerc0125', '20250821151825', 9.362, 113), 'PHerc0826': _sc('PHerc0826', '20250821151701', 9.362, 113), 'PHerc1447': _sc('PHerc1447', '20250521151220', 8.64, 116), 'PHerc0800': _sc('PHerc0800', '20250521135224', 8.64, 116), 'PHerc0306B': _sc('PHerc0306B', '20250521133212', 8.64, 116),
           'PHerc0813': _sc('PHerc0813', '20250821151723', 9.362, 113), 'PHerc0358': _sc('PHerc0358', '20250821151737', 9.362, 113),
           'PHerc0211': _sc('PHerc0211', '20250821151803', 9.362, 113)}
UM = 8.64 if A.control306 else SCROLLS[A.scroll]['um']; UM4 = UM * DS
MODELS = A.models.split(','); STAT = dict(trace_s=0.0, studio_s=0.0, studio_win=0, trace_px=0)
KOFFS = [int(v) for v in A.koffs.split(',')]; FTAGS = [''] if len(KOFFS) == 1 else ['_k' + ('m' if k < 0 else 'p') + str(abs(k)) for k in KOFFS]
def log(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)

# ---------------------------------------------------------------- volume tiles on the GPU
def needed(Pt, Nt, vt, k0, k1, ch, stride=8, kstep=24, margin=3):
    Pv, Nv = Pt[::stride, ::stride][vt[::stride, ::stride]], Nt[::stride, ::stride][vt[::stride, ::stride]]
    out = set()
    if len(Pv) == 0: return out
    for k in list(np.arange(k0, k1, kstep)) + [k1]:
        q = Pv - float(k) * Nv
        for dz in (-margin, margin):
            for dy in (-margin, margin):
                for dx in (-margin, margin):
                    c = np.floor((q + np.array([dz, dy, dx], np.float32)) / ch).astype(np.int64)
                    out.update(map(tuple, np.unique(c, axis=0).tolist()))
    return {k for k in out if min(k) >= 0}

def tile_tensor(V, keys):
    V.fetch(keys, threads=A.threads)
    ks = np.array(sorted(keys)); lo = ks.min(0); hi = ks.max(0) + 1; ch = V.ch
    a = np.zeros(tuple((hi - lo) * ch), np.uint8)
    for k in keys:
        c = V.c.get(tuple(k))
        if c is not None:
            o = (np.array(k) - lo) * ch; a[o[0]:o[0] + ch, o[1]:o[1] + ch, o[2]:o[2] + ch] = c
    nb = sum(0 if V.c.get(tuple(k)) is None else 1 for k in keys); V.c.clear()
    t = torch.from_numpy(a).to(DEV)[None, None].float()          # float32: half-precision grid coordinates would jitter samples
    off = torch.tensor(lo * ch, dtype=torch.float32, device=DEV); size = torch.tensor(a.shape, dtype=torch.float32, device=DEV)
    return t, off, size, nb

def gs(t, off, size, q, mode):
    g = ((q - off) / (size - 1) * 2 - 1)[..., [2, 1, 0]][None, None]
    return F.grid_sample(t, g, mode=mode, padding_mode='zeros', align_corners=True)[0, 0, 0]

# ---------------------------------------------------------------- TRACE
_Z = None
def trace(stack, name='trace'):
    """stack (17,h,w) uint8 -> probability map (h,w) float32, layers in the given order"""
    global _Z
    if A.dry:
        g = ndi.gaussian_filter(stack[C].astype(np.float32), 3); return (g - g.min()) / max(float(np.ptp(g)), 1e-6)
    if _Z is None:
        Cd = f'{R}/r/submission/code'; os.environ['FL_ROOT'] = Cd; os.environ['VILLA_SRC'] = f'{R}/villa/vesuvius/src'
        sys.path.insert(0, f'{Cd}/x_zoom'); import zoomlib as Z; _Z = Z
        Z.CK['v3'] = f'{Cd}/models/community/reader-v3/reader-v3.pth'
    return _Z.villa_map(name, stack, C, 0, 'forward', overlap=0.5, tta8=False, batch=A.batch).astype(np.float32)

_V8 = None
def v8(stack):
    """stack (25,h,w) uint8 -> v8-in probability map; 24 layers C-12..C+11 in the given order, clipped at 200, stride 21"""
    global _V8
    if A.dry:
        g = ndi.gaussian_filter(stack[C].astype(np.float32), 3); return (g - g.min()) / max(float(np.ptp(g)), 1e-6)
    if _V8 is None:
        Cd = f'{R}/r/submission/code'; os.environ['FL_ROOT'] = Cd; os.environ['VILLA_SRC'] = f'{R}/villa/vesuvius/src'
        sys.path.insert(0, f'{Cd}/x_research2/speed'); import fastink as FI; _V8 = FI.V8Fast(amp='fp16')
    st = np.ascontiguousarray(np.minimum(stack[C - 12:C + 12], 200)).astype(np.uint8)
    out, n = _V8.predict(st, reverse=False, stride=21, batch=16); return out.astype(np.float32)

_RERR = set()
def reader(mdl, sf):
    try:
        if mdl == 'studio': return studio(np.ascontiguousarray(sf[C + 8:C - 8:-1]))
        if mdl == 'v8': return v8(sf)
        return trace(sf, mdl)
    except Exception as e:
        if mdl not in _RERR:
            import traceback; traceback.print_exc(); log('READER_FAILED', mdl, repr(e)[:300]); _RERR.add(mdl)
        return np.zeros(sf.shape[1:], np.float32)

# ---------------------------------------------------------------- Studio soft-loss (the PHerc0211 First Letters ink model)
_S = None; _SB = [0]
def studio(p16):
    """p16 (16,h,w) uint8 in the model's depth order -> probability map (h,w) float32.
    Same tiling as the model's published inference.py: 128 px windows, central 96 px, Hann-weighted overlap average,
    reflect padding at the borders, clip(raw,0,200)/255, bf16. Windows whose central 96 px hold no CT are skipped."""
    global _S
    if A.dry:
        g = ndi.gaussian_filter(p16[8].astype(np.float32), 3); return (g - g.min()) / max(float(np.ptp(g)), 1e-6)
    if _S is None:
        sys.path.insert(0, f'{R}/studio/studio-multiscroll-cpt-20261023/hf')
        from model import DeepResidualUNet
        from safetensors.torch import load_file
        m = DeepResidualUNet(deeper=False); m.load_state_dict(load_file(f'{R}/weights/studio-softloss-20261023-ema.safetensors', device='cpu'), strict=True)
        k = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(96) + 0.5) / 96)
        _S = (m.eval().to(DEV), torch.from_numpy((k[:, None] * k[None, :]).astype(np.float32)).to(DEV)); _SB[0] = A.batch
    model, kern = _S; h, w = p16.shape[1:]; sd = A.studio_stride
    ys = sorted(set(list(range(0, h, sd)) + [max(h - 96, 0)])); xs = sorted(set(list(range(0, w, sd)) + [max(w - 96, 0)]))
    occ = (p16 != 0).any(0)
    coords = [(y, x) for y in ys for x in xs if occ[y:y + 96, x:x + 96].any()]
    with torch.inference_mode():
        v = torch.from_numpy(p16).to(DEV).float()[None]
        vp = F.pad(v, (16, max(0, xs[-1] + 112 - w), 16, max(0, ys[-1] + 112 - h)), mode='reflect')[0]
        res = torch.zeros((h, w), device=DEV); wt = torch.zeros((h, w), device=DEV); i = 0
        while i < len(coords):
            bc = coords[i:i + _SB[0]]
            try:
                x = torch.stack([vp[:, y:y + 128, x_:x_ + 128] for y, x_ in bc])[:, None].clamp(0, 200) / 255
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=DEV == 'cuda'):
                    lg = model(x)
                pr = lg[:, 0, 16:112, 16:112].float().sigmoid()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); _SB[0] = max(1, _SB[0] // 2); log('studio batch ->', _SB[0]); continue
            for (y, x_), p in zip(bc, pr):
                hh, ww = min(96, h - y), min(96, w - x_)
                res[y:y + hh, x_:x_ + ww] += p[:hh, :ww] * kern[:hh, :ww]; wt[y:y + hh, x_:x_ + ww] += kern[:hh, :ww]
            i += len(bc)
        out = torch.where(wt > 0, res / wt.clamp_min(1e-12), torch.zeros_like(res)).cpu().numpy()
    out[~occ] = 0; STAT['studio_win'] += len(coords)
    return out.astype(np.float32)


# ---------------------------------------------------------------- surface refit: make the cut follow the papyrus layering
def _blur(x, sg):
    r = int(3 * sg); g = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=x.device, dtype=torch.float32) / sg) ** 2); g /= g.sum()
    x = F.conv2d(x[:, None], g[None, None, None, :], padding=(0, r)); x = F.conv2d(x, g[None, None, :, None], padding=(r, 0)); return x[:, 0]

def flatten_prof(prof, m, S=6, srefs=(48, 32, 24, 16, 12)):
    """prof (2R+1,H,W) CT along the normal about a rough surface, m (H,W) float mask -> d (H,W): depth shift (in layers) that
    aligns every column's depth profile with the local mean profile of its (already aligned) neighbours."""
    R = (prof.shape[0] - 1) // 2; Q = R - S
    kk = torch.arange(-Q, Q + 1, device=prof.device, dtype=torch.float32)
    d = torch.zeros_like(m); wm4 = _blur(m[None], 4.0)[0].clamp_min(1e-3); wm3 = _blur(m[None], 3.0)[0].clamp_min(1e-3)
    for sref in srefs:
        idx = (kk[:, None, None] + d[None] + R).clamp(0, 2 * R - 1e-3); i0 = idx.floor().long(); f = idx - i0
        A0 = (torch.gather(prof, 0, i0) * (1 - f) + torch.gather(prof, 0, (i0 + 1).clamp(max=2 * R)) * f) * m[None]
        ref = _blur(A0, float(sref)) / _blur(m[None], float(sref)).clamp_min(1e-3)
        ref0 = ref - ref.mean(0, keepdim=True); rn = ref0.norm(dim=0) + 1e-6
        cost = []
        for sh in range(-S, S + 1):
            seg = prof[S + sh:S + sh + 2 * Q + 1]; seg0 = seg - seg.mean(0, keepdim=True)
            c = -(seg0 * ref0).sum(0) / (seg0.norm(dim=0) * rn + 1e-6)
            cost.append(_blur((c * m)[None], 4.0)[0] / wm4)
        d = torch.stack(cost).argmin(0).float() - S
        d = _blur((d * m)[None], 3.0)[0] / wm3
    return d * m

def flatten(t, off, size, P2, Ng, mu, R=13):
    prof = torch.stack([gs(t, off, size, P2 - float(l) * Ng, 'bilinear') for l in range(-R, R + 1)])
    return flatten_prof(prof, mu.float(), S=A.fl_s)

def down(a, f=DS):
    h, w = a.shape[0] // f * f, a.shape[1] // f * f
    return a[:h, :w].reshape(h // f, f, w // f, f).mean((1, 3))

# ---------------------------------------------------------------- one base mesh
def process(name, grids, cfg, od, D=D):
    t0 = time.time(); os.makedirs(od, exist_ok=True)
    P, N, valid = geom.from_grids(*grids)
    H, W = valid.shape
    if not valid.any(): json.dump(dict(skipped='no valid nodes'), open(f'{od}/done.json', 'w')); return None
    CT = vol.open_zarr(cfg['ct']); M7 = vol.open_zarr(cfg['m7'])
    # 2. sheets: m7 slab at LS px, tracked on the whole mesh
    Pl, Nl, vl = P[::LS, ::LS], N[::LS, ::LS], valid[::LS, ::LS]; h, w = vl.shape
    Mb = np.zeros((2 * D + 1, h, w), bool); kk = torch.arange(-D, D + 1, dtype=torch.float32, device=DEV); TL = TILE // LS; nm7 = 0
    for a in range(0, w, TL):
        Pt, Nt, vt = Pl[:, a:a + TL], Nl[:, a:a + TL], vl[:, a:a + TL]
        keys = needed(Pt, Nt, vt, -D, D, M7.ch, stride=2)
        if not keys: continue
        t, off, size, nb = tile_tensor(M7, keys); nm7 += nb
        Pg, Ng = torch.from_numpy(np.ascontiguousarray(Pt)).to(DEV), torch.from_numpy(np.ascontiguousarray(Nt)).to(DEV)
        for b in range(0, len(kk), 32):
            q = Pg[None] - kk[b:b + 32, None, None, None] * Ng[None]                    # (32,h,tw,3)
            g = ((q - off) / (size - 1) * 2 - 1)[..., [2, 1, 0]][None]
            s = F.grid_sample(t, g, mode='nearest', padding_mode='zeros', align_corners=True)[0, 0]
            Mb[b:b + 32, :, a:a + TL] = (s > 0).cpu().numpy()
        del t
        if DEV == 'cuda': torch.cuda.empty_cache()
    Mb &= vl[None]; t1 = time.time()
    cols, cen, thick = st.run_centres(Mb); del Mb
    cell_mm2 = (LS * UM / 1000) ** 2
    pat = st.link(cols, cen, (2 * D + 1, h, w), tau=1.5 + LS / 4, tau2=2.0 + LS / 4, min_cells=int(A.min_mm2 / cell_mm2))
    canv = st.pack([st.smooth_fill(p, sigma=6.0 / LS + 0.25) for p in pat]); canv.sort(key=lambda c: np.nanmedian(c['H']))
    if A.base_only: canv = [dict(H=np.where(vl, float(D), np.nan).astype(np.float32), occ=vl.copy(), n=1)]
    t2 = time.time(); nc = len(canv)
    if nc == 0: json.dump(dict(skipped='no sheets tracked', name=name), open(f'{od}/done.json', 'w')); return None
    for c in canv:                                                                         # fill outside the footprint with the nearest height
        _, (ii, jj) = ndi.distance_transform_edt(~c['occ'], return_indices=True); c['Hf'] = c['H'][ii, jj].astype(np.float32)
    np.savez_compressed(f'{od}/canv.npz', H=np.stack([c['H'] for c in canv]).astype(np.float16), D=D, ls=LS, shape=np.array([H, W]))
    # 3. read: per tile, per canvas
    H4, W4 = H // DS, W // DS
    FACES = {'studio': A.studio_faces.split(','), 'trace': A.trace_faces.split(','), 'v3': A.trace_faces.split(','), 'v8': A.trace_faces.split(',')}
    maps = {(mdl, f): np.zeros((nc, H4, W4), np.uint8) for mdl in MODELS for f0 in ('forward', 'reverse') if f0 in FACES[mdl] for f in [f0 + ft for ft in FTAGS]}; ctq = np.zeros((nc, H4, W4), np.uint8); nct = 0; calls = 0
    Hf_t = torch.from_numpy(np.stack([c['Hf'] for c in canv])).to(DEV); occ_t = torch.from_numpy(np.stack([c['occ'] for c in canv]).astype(np.float32)).to(DEV)
    for x0 in range(0, W, TILE):
        xa, xb = max(0, x0 - MARG), min(W, x0 + TILE + MARG); x1 = min(W, x0 + TILE)
        Pt, Nt, vt = P[:, xa:xb], N[:, xa:xb], valid[:, xa:xb]
        if not vt[:, x0 - xa:x1 - xa].any(): continue
        if XR and (x1 <= XR[0] or x0 >= XR[1]): continue
        keys = needed(Pt, Nt, vt, -D - 13, D + 13, CT.ch)
        t, off, size, nb = tile_tensor(CT, keys); nct += nb
        Pg, Ng = torch.from_numpy(np.ascontiguousarray(Pt)).to(DEV), torch.from_numpy(np.ascontiguousarray(Nt)).to(DEV); vg = torch.from_numpy(np.ascontiguousarray(vt)).to(DEV)
        yy = torch.arange(H, device=DEV, dtype=torch.float32) / LS / max(h - 1, 1) * 2 - 1
        xx = torch.arange(xa, xb, device=DEV, dtype=torch.float32) / LS / max(w - 1, 1) * 2 - 1
        g2 = torch.stack(torch.meshgrid(xx, yy, indexing='xy'), -1)[None]                 # (1,H,Wt,2) as x,y
        for ci in range(nc):
            if CANV and ci not in CANV: continue
            mu = (F.grid_sample(occ_t[ci][None, None], g2, mode='bilinear', padding_mode='zeros', align_corners=True)[0, 0] > 0.99) & vg
            if int(mu[:, x0 - xa:x1 - xa].sum()) < 2000: continue
            hu = F.grid_sample(Hf_t[ci][None, None], g2, mode='bilinear', padding_mode='border', align_corners=True)[0, 0] - D
            P2 = Pg - hu[..., None] * Ng
            if A.flatten:
                dfl = flatten(t, off, size, P2, Ng, mu); P2 = P2 - dfl[..., None] * Ng
                log('   flatten c', ci, 'x0', x0, 'mean|d|', round(float(dfl.abs()[mu].mean()), 2), 'max', round(float(dfl.abs()[mu].max()), 2))
            P2b = P2; mu0 = mu
            for ko, ftag in zip(KOFFS, FTAGS):
                P2 = P2b - float(ko) * Ng; mu = mu0.clone()
                stack = torch.stack([gs(t, off, size, P2 - float(l) * Ng, 'bilinear') for l in LAYERS])   # layer index increases along -N
                stack = torch.round(stack).clamp(0, 255).to(torch.uint8); stack[:, ~mu] = 0
                mu &= stack[C] > 0; s = stack.cpu().numpy(); m = mu.cpu().numpy()
                core = np.s_[:H4 * DS, x0 - xa:x1 - xa]; m4 = down(m[core].astype(np.float32)) > 0.9; wq = m4.shape[1]
                for face in ('forward', 'reverse'):
                    sf = s if face == 'forward' else np.ascontiguousarray(s[::-1])
                    for mdl in MODELS:                                                        # studio order: +8 .. -7 about the surface
                        if face not in FACES[mdl]: continue
                        tm = time.time(); p = reader(mdl, sf)
                        STAT[mdl + '_s'] = STAT.get(mdl + '_s', 0.0) + time.time() - tm; calls += 1
                        if mdl == 'trace': STAT['trace_px'] += int(m.sum())
                        p[~m] = 0; p4 = down(p[core]); p4[~m4] = 0
                        maps[(mdl, face + ftag)][ci, :, x0 // DS:x0 // DS + wq] = np.clip(np.round(p4 * 255), 1, 255).astype(np.uint8) * m4
                if ko == 0: c4 = down(s[C][core].astype(np.float32)); ctq[ci, :, x0 // DS:x0 // DS + wq] = np.clip(c4, 1, 255).astype(np.uint8) * m4
        del t
        if DEV == 'cuda': torch.cuda.empty_cache()
        log('  tile x0', x0, 'of', W, '| ct chunks', nct, '|', {k: round(v, 1) if isinstance(v, float) else v for k, v in STAT.items()})
    t3 = time.time()
    # 4. save + line detector
    os.makedirs(f'{od}/maps', exist_ok=True); os.makedirs(f'{od}/ct', exist_ok=True); chains = []
    cov = []; dens = {}
    for ci in range(nc):
        cov.append(round(float((ctq[ci] > 0).mean()), 3))
        if CANV and ci not in CANV: continue
        Image.fromarray(ctq[ci]).save(f'{od}/ct/c{ci:02d}.png')
        for (mdl, face), mp in maps.items():
            Image.fromarray(mp[ci]).save(f'{od}/maps/c{ci:02d}_{mdl}_{face}.png')
            cs, area = lf.detect(mp[ci].astype(np.float32) / 255, UM4)
            dens.setdefault(f'{mdl}_{face}', {})[ci] = [int((mp[ci] > 0).sum()), int((mp[ci] > 127).sum())]
            for c in cs[:15]: c.update(mesh=name, canvas=ci, face=face, model=mdl); chains.append(c)
    chains.sort(key=lambda c: -c['score']); json.dump(chains, open(f'{od}/lines.json', 'w'))
    res = dict(name=name, shape=[H, W], canvases=nc, patches=len(pat), cover=cov, median_k=[int(np.nanmedian(c['H']) - D) for c in canv],
               area_cm2=round(float(sum(cov)) * H4 * W4 * (UM4 / 1e4) ** 2, 1), m7_chunks=nm7, ct_chunks=nct, trace_calls=calls,
               secs=dict(m7=round(t1 - t0, 1), track=round(t2 - t1, 1), read=round(t3 - t2, 1), total=round(time.time() - t0, 1)),
               stat={k: round(v, 1) if isinstance(v, float) else v for k, v in STAT.items()}, dens=dens, um=UM,
               best=[{k: c[k] for k in ('score', 'n_slots', 'len_mm', 'contrast', 'canvas', 'face', 'model')} for c in chains[:3]])
    json.dump(res, open(f'{od}/done.json', 'w'))
    log(name, {k: res[k] for k in ('shape', 'canvases', 'area_cm2', 'ct_chunks', 'trace_calls', 'secs')}, 'best', res['best'][:1])
    return dict(canv=canv, maps=maps, chains=chains, res=res, valid=valid)

# ---------------------------------------------------------------- positive control
def control306():
    mesh = f'{R}/r/submission/mesh/PHerc0306B_20250521133212_w074own_touchup_V5Rc'; r0, r1 = 62, 157; c0, c1 = [int(v) for v in A.ccols.split(':')]
    gz, gy, gx, gm = geom.read_grids(mesh, cols=(c0, c1), rows=(r0, r1)); n = geom.node_normals(gz, gy, gx)
    ii, jj = np.meshgrid(np.arange(gz.shape[0]), np.arange(gz.shape[1]), indexing='ij'); Dn = 70 + 15 * np.sin(ii / 14.0) * np.cos(jj / 19.0)
    od = A.out or f'{R}/control306'
    out = process('control306', (gz + Dn * n[..., 0], gy + Dn * n[..., 1], gx + Dn * n[..., 2], gm), SCROLLS['PHerc0306B'], od, D=150)
    h, w = out['canv'][0]['H'].shape
    RR, CC = np.meshgrid(np.arange(h) * LS / geom.STEP, np.arange(w) * LS / geom.STEP, indexing='ij')
    truth = ndi.map_coordinates(Dn, [RR, CC], order=1, mode='nearest') + 150
    by0, by1, bx0, bx1 = [(v - o) for v, o in zip((1639, 2745, 2431, 6829), (r0 * 20, r0 * 20, c0 * 20, c0 * 20))]     # claim box, px in this crop
    box = np.zeros((h, w), bool); box[by0 // LS:by1 // LS, max(0, bx0) // LS:bx1 // LS] = True
    frac = [float(((np.abs(c['H'] - truth) <= 3) & c['occ'] & box).sum() / box.sum()) for c in out['canv']]
    tc = int(np.argmax(frac)); log('truth canvas', tc, 'holds', round(frac[tc], 2), 'of the claim box within 3 voxels; all canvases', np.round(frac, 2).tolist())
    mm = UM / 1000
    def in_box(c): return by0 * mm <= c['cy_mm'] <= by1 * mm and bx0 * mm <= c['cx_mm'] <= bx1 * mm
    allhits = {}; ok = True
    for mdl in MODELS:
        ranked = [c for c in out['chains'] if c['model'] == mdl]; hits = [(i + 1, c) for i, c in enumerate(ranked) if c['canvas'] == tc and in_box(c)]
        log(mdl, 'chains', len(ranked), '| top 5:', [(c['canvas'], c['face'], c['score'], c['n_slots'], c['len_mm']) for c in ranked[:5]])
        allhits[mdl] = [(r, c) for r, c in hits[:5]]
        if hits:
            r, c = hits[0]; log('CONTROL', mdl, ': best chain on the text sheet inside the claim box ranks', r, 'of', len(ranked), {k: c[k] for k in ('face', 'score', 'n_slots', 'len_mm', 'contrast')})
            log('CONTROL_PASS' if r <= 3 and c['n_slots'] >= 5 else 'CONTROL_WEAK', mdl)
        else: log('CONTROL_FAIL', mdl, ': no chain on the text sheet inside the claim box')
    hits = allhits
    json.dump(dict(truth_canvas=tc, frac=frac, hits=hits), open(f'{od}/control.json', 'w'))

def main():
    log('device', DEV, '| D', D, 'LS', LS, 'tile', TILE, '+', MARG, '| dry' if A.dry else '')
    if A.control306: control306(); log('SWEEP_DONE'); return
    md = A.meshdir or f'{R}/pmesh/meshes/{A.scroll}'
    names = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f'{md}/*/x.tif'))
    if A.only: names = [n for n in names if n in A.only.split(',')]
    names.sort(key=lambda n: -os.path.getsize(f'{md}/{n}/x.tif') if False else (n.split('_')[1], n))    # by winding then z: shards get equal mixes
    i, n = [int(v) for v in A.shard.split('/')]; names = names[i::n]
    if A.desc: names = names[::-1]
    OUT = A.out or f'{R}/sweep/{A.scroll}'; log(len(names), 'meshes in this shard'); log('faces studio', A.studio_faces, '| trace', A.trace_faces)
    for name in names:
        od = f'{OUT}/{name}'
        if os.path.exists(f'{od}/done.json'): continue
        if A.claim:
            try: os.makedirs(od)
            except FileExistsError: continue
        if (time.time() - T0) / 60 > A.max_minutes: log('TIME CAP before', name); break
        rows = tuple(int(v) for v in A.rows.split(':')) if A.rows else None; cols = tuple(int(v) for v in A.cols.split(':')) if A.cols else None
        try:
            process(name, geom.read_grids(f'{md}/{name}', cols=cols, rows=rows), SCROLLS[A.scroll], od)
        except Exception as e:
            import traceback; traceback.print_exc(); log('FAILED', name, repr(e)[:300])
            if DEV == 'cuda': torch.cuda.empty_cache()
    log('SWEEP_DONE')

if __name__ == '__main__':
    main()
