"""meshcheck.py <tifxyz dir> [out.png] : is a surface stretched, sheared, or stepping between sheets?

CPU only, numpy + tifffile (+ pillow for the picture). Works on any tifxyz (x.tif, y.tif, z.tif; invalid nodes <= 0 or -1).

For every valid node it measures, in scan voxels:
  step_u, step_v   distance to the next node along a row / down a column (a clean 20-voxel mesh gives 20, 20)
  shear            angle between the two grid directions, as degrees away from 90
  bend             how far the node sits off the straight line through its two neighbours, measured along the surface
                   normal (second difference / 2). Gentle curvature gives about a voxel. Tight folds and steps between
                   papyrus sheets both give several voxels; geometry alone cannot tell those two apart.

Reading the numbers:
  steps far from the nominal spacing, or shear far from 0, with ordinary bend -> the 3D surface is fine, the flattening /
                                                                                 parameterisation is distorted
  lines of large bend in the picture                                          -> the 3D surface itself folds sharply or
                                                                                 steps between layers; check those lines
                                                                                 against the CT

Reference values on published surfaces (20-voxel meshes, 9.4 / 8.6 um scans):
  PHerc. 211 First Letters mesh (hand-refined):  steps 20.0 / 20.1, shear p95 1.7 deg,  bend p50 0.7  p95 2.4  p99 6.2
  PHerc. 343 First Letters mesh w047:            steps 19.8 / 19.8, shear p95 5.7 deg,  bend p50 1.7  p95 6.4  p99 10.5
  a spiral-fit band (pscamillo z7920_w020):      steps 20.1 / 20.0, shear p95 3.8 deg,  bend p50 0.8  p95 3.0  p99 9.3

The picture (if out.png is given) shows four maps side by side: step_u, step_v, shear, bend.
"""
import sys, json, os
import numpy as np, tifffile


def load(d):
    g = [tifffile.imread(os.path.join(d, c + '.tif')).astype(np.float64) for c in 'xyz']
    P = np.stack(g, -1); valid = (P[..., 2] > 0) & np.isfinite(P).all(-1)
    return P, valid


def measure(P, valid):
    H, W = valid.shape; nan = np.full((H, W), np.nan)
    du = P[:, 1:] - P[:, :-1]; vu = valid[:, 1:] & valid[:, :-1]
    dv = P[1:] - P[:-1]; vv = valid[1:] & valid[:-1]
    su = nan.copy(); su[:, :-1] = np.where(vu, np.linalg.norm(du, axis=-1), np.nan)
    sv = nan.copy(); sv[:-1] = np.where(vv, np.linalg.norm(dv, axis=-1), np.nan)
    # shear and normal from central differences
    tu = np.full(P.shape, np.nan); tv = np.full(P.shape, np.nan)
    cu = valid[:, 2:] & valid[:, :-2]; tu[:, 1:-1][cu] = (P[:, 2:] - P[:, :-2])[cu]
    cv = valid[2:] & valid[:-2]; tv[1:-1][cv] = (P[2:] - P[:-2])[cv]
    n = np.cross(tu, tv); nn = np.linalg.norm(n, axis=-1, keepdims=True); n = n / np.where(nn > 0, nn, np.nan)
    cosang = (tu * tv).sum(-1) / (np.linalg.norm(tu, axis=-1) * np.linalg.norm(tv, axis=-1))
    shear = np.degrees(np.arcsin(np.clip(cosang, -1, 1)))                      # 0 = square grid
    bu = np.full((H, W), np.nan); m = valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2]
    d2 = P[:, 2:] - 2 * P[:, 1:-1] + P[:, :-2]; bu[:, 1:-1] = np.where(m, np.abs((d2 * n[:, 1:-1]).sum(-1)) / 2, np.nan)
    bv = np.full((H, W), np.nan); m = valid[2:] & valid[1:-1] & valid[:-2]
    d2 = P[2:] - 2 * P[1:-1] + P[:-2]; bv[1:-1] = np.where(m, np.abs((d2 * n[1:-1]).sum(-1)) / 2, np.nan)
    bend = np.fmax(bu, bv)
    return su, sv, shear, bend


def q(a, ps=(5, 50, 95, 99)):
    a = a[np.isfinite(a)]
    return [round(float(np.percentile(a, p)), 2) for p in ps] if a.size else [None] * len(ps)


def main():
    d = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else None
    P, valid = load(d); su, sv, shear, bend = measure(P, valid)
    nom = float(np.nanmedian(np.concatenate([su[np.isfinite(su)], sv[np.isfinite(sv)]])))
    far = lambda s: float(np.nanmean(np.abs(s[np.isfinite(s)] / nom - 1) > 0.25))
    hop = float(np.nanmean(bend[np.isfinite(bend)] > 4)); hop8 = float(np.nanmean(bend[np.isfinite(bend)] > 8))
    rep = dict(mesh=os.path.basename(os.path.normpath(d)), nodes=[int(v) for v in valid.shape], valid_nodes=int(valid.sum()),
               area_cm2_at_9p362um=round(float(np.nansum(su * sv)) * (9.362e-4) ** 2, 2), nominal_step=round(nom, 2),
               step_u_p5_p50_p95_p99=q(su), step_v_p5_p50_p95_p99=q(sv), abs_shear_deg_p50_p95_p99=q(np.abs(shear), (50, 95, 99)),
               bend_vox_p50_p95_p99=q(bend, (50, 95, 99)),
               share_step_u_off_by_25pct=round(far(su), 3), share_step_v_off_by_25pct=round(far(sv), 3), share_bend_over_4_vox=round(hop, 4), share_bend_over_8_vox=round(hop8, 4))
    v = []
    if max(rep['share_step_u_off_by_25pct'], rep['share_step_v_off_by_25pct']) > 0.05 or (rep['abs_shear_deg_p50_p95_p99'][1] or 0) > 15:
        v.append('parameterisation is distorted (stretch or shear) on part of the surface')
    if hop8 > 0.02: v.append('sharp bends (more than 8 voxels) on %.1f%% of nodes: tight folds or steps between sheets; look at the bend map' % (100 * hop8))
    rep['reading'] = v or ['even spacing, near-square grid, bend in the range of published surfaces']
    print(json.dumps(rep, indent=1))
    if out:
        from PIL import Image
        def im(a, lo, hi):
            x = np.clip((a - lo) / (hi - lo), 0, 1); x = np.where(np.isfinite(a), x, 0)
            rgb = np.stack([x, 1 - np.abs(2 * x - 1), 1 - x], -1) * np.isfinite(a)[..., None]       # blue low, green mid, red high
            return (rgb * 255).astype(np.uint8)
        k = max(1, int(round(600 / max(valid.shape))))
        tiles = [im(su, 0.5 * nom, 1.5 * nom), im(sv, 0.5 * nom, 1.5 * nom), im(np.abs(shear), 0, 30), im(bend, 0, 8)]
        gap = np.zeros((valid.shape[0], 2, 3), np.uint8) + 255
        row = np.concatenate([t for x in tiles for t in (x, gap)][:-1], 1)
        Image.fromarray(row).resize((row.shape[1] * k, row.shape[0] * k), Image.NEAREST).save(out)


if __name__ == '__main__':
    main()
