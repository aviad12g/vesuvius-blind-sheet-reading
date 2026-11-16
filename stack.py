import numpy as np, glob, sys, re
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
d, out = sys.argv[1], sys.argv[2]; box = [int(v) for v in sys.argv[3].split(',')] if len(sys.argv) > 3 else None   # y0,y1,x0,x1 of the mark
fs = glob.glob(d + '/c00_studio_forward_k*.png'); kof = lambda f: (-1 if 'km' in f else 1) * int(re.findall(r'k[mp](\d+)', f)[0]); fs = sorted(fs, key=kof)
M = {kof(f): np.asarray(Image.open(f)) for f in fs}; ks = sorted(M); m = M[0] > 0; P = {k: np.where(m & (M[k] > 0), (M[k].astype(np.float32) - 1) / 254, 0) for k in ks}
def marks(p):
    lab, n = ndi.label(p > 0.5)
    if n == 0: return 0
    ar = ndi.sum(np.ones(p.shape, np.float32), lab, np.arange(1, n + 1)); return int(((ar >= 2400) & (ar <= 56000)).sum())      # letter-sized at full resolution
print('depth | dark%% whole patch | letter-sized marks | match with depth 0' + (' || in the mark box: dark%% | match with depth 0' if box else ''))
for k in ks:
    p = P[k]; s = '%+4d | %5.1f | %3d | %5.2f' % (k, (p[m] > 0.5).mean() * 100, marks(p), np.corrcoef(p[m], P[0][m])[0, 1])
    if box:
        y0, y1, x0, x1 = box; mb = m[y0:y1, x0:x1]; pb = p[y0:y1, x0:x1][mb]; p0 = P[0][y0:y1, x0:x1][mb]; s += ' || %5.1f | %5.2f' % ((pb > 0.5).mean() * 100, np.corrcoef(pb, p0)[0, 1] if pb.std() > 1e-4 else 0)
    print(s)
h, w = m.shape; sc = 4; hh, ww = h // sc, w // sc; cols = 2; rows = (len(ks) + 1) // cols; cv = Image.new('L', (cols * (ww + 10), rows * (hh + 16)), 255); dr = ImageDraw.Draw(cv)
for i, k in enumerate(ks):
    im = Image.fromarray(np.where(m, 255 - P[k] * 255, 255).astype(np.uint8)).resize((ww, hh), Image.LANCZOS); x, y = (i // rows) * (ww + 10), (i % rows) * (hh + 16)
    cv.paste(im, (x, y + 14)); dr.text((x + 2, y + 1), 'depth %+d px' % k, fill=0)
cv.save(out); print('saved', out, cv.size)
