"""HF tifxyz of PHerc1447 w058 (scale 0.25 = 4 voxels per node) -> two base meshes in our layout (20 voxels per node, invalid = -1).
usage: mk1447.py <hf dir> <out meshdir>.  Bands are node ranges of the every-5th-node grid: render px = node * 20."""
import numpy as np, tifffile, sys, os, json
hf, out = sys.argv[1:3]
COLS = (33, 222); BANDS = {'a': (35, 128), 'b': (128, 220)}
g = {c: tifffile.imread(f'{hf}/w058/tifxyz/{c}.tif').astype(np.float32) for c in 'xyz'}
ok = np.ones(g['x'].shape, bool)
for c in 'xyz': ok &= np.isfinite(g[c]) & (g[c] > 0)
for b, (r0, r1) in BANDS.items():
    d = f'{out}/t1447_w058{b}'; os.makedirs(d, exist_ok=True)
    for c in 'xyz':
        a = np.where(ok, g[c], -1.0).astype(np.float32)[::5, ::5][r0:r1, COLS[0]:COLS[1]]; tifffile.imwrite(f'{d}/{c}.tif', a)
    json.dump(dict(uuid=f't1447_w058{b}', type='seg', format='tifxyz', scale=[0.05, 0.05], rows=[r0, r1], cols=list(COLS),
                   source='YoussefMoNader/ink-8um-pherc1447-surfaces w058, every 5th node'), open(f'{d}/meta.json', 'w'))
    print(b, a.shape, 'valid %.2f' % (a != -1).mean(), '= render px rows %d:%d cols %d:%d' % (r0 * 20, r1 * 20, COLS[0] * 20, COLS[1] * 20))
