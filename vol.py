"""vol.py : tiny chunk-cached reader/sampler for the public Vesuvius zarr volumes (zarr v2, '/' separator)."""
import os, time, threading, concurrent.futures as cf
import numpy as np, requests, numcodecs
B = 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com'
SESS = requests.Session(); SESS.mount('https://', requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64))

class Vol:
    def __init__(self, url, chunk, codec=None, cache_dir=None):
        self.url, self.ch, self.codec, self.c, self.lock, self.bytes, self.dir = url, chunk, codec, {}, threading.Lock(), 0, cache_dir
        if cache_dir: os.makedirs(cache_dir, exist_ok=True)
    def _get(self, key):
        if key in self.c: return
        p = f'{self.dir}/{key[0]}_{key[1]}_{key[2]}.npy' if self.dir else None
        if p and os.path.exists(p):
            a = np.load(p); a = None if a.size == 0 else a
        else:
            for t in range(6):
                try:
                    r = SESS.get(f'{self.url}/{key[0]}/{key[1]}/{key[2]}', timeout=90)
                    if r.status_code in (403, 404): a = None
                    else:
                        r.raise_for_status(); raw = r.content if self.codec is None else self.codec.decode(r.content)
                        a = np.frombuffer(raw, np.uint8).reshape(self.ch, self.ch, self.ch)
                    break
                except Exception as e:
                    err = e; time.sleep(1 + 2 * t)
            else: raise err
            if p: np.save(p, a if a is not None else np.zeros(0, np.uint8))
        with self.lock:
            self.c[key] = a; self.bytes += 0 if a is None else a.nbytes
    def fetch(self, keys, threads=32):
        keys = [k for k in keys if k not in self.c]
        if keys:
            with cf.ThreadPoolExecutor(threads) as ex: list(ex.map(self._get, keys))
    def dense(self, lo, hi):
        """dense uint8 array of voxel box [lo, hi) (zyx), fetching what is needed"""
        lo = np.maximum(np.asarray(lo), 0); c0 = lo // self.ch; c1 = (np.asarray(hi) - 1) // self.ch
        keys = [(z, y, x) for z in range(c0[0], c1[0] + 1) for y in range(c0[1], c1[1] + 1) for x in range(c0[2], c1[2] + 1)]
        self.fetch(keys)
        out = np.zeros(tuple((c1 - c0 + 1) * self.ch), np.uint8)
        for k in keys:
            a = self.c.get(k)
            if a is not None:
                o = (np.array(k) - c0) * self.ch; out[o[0]:o[0] + self.ch, o[1]:o[1] + self.ch, o[2]:o[2] + self.ch] = a
        return out, c0 * self.ch
    def sample(self, q, order=1):
        """q (...,3) zyx float -> values (float32), via one dense box around the points"""
        from scipy import ndimage as ndi
        f = q.reshape(-1, 3); lo = np.floor(f.min(0)).astype(int) - 2; hi = np.ceil(f.max(0)).astype(int) + 3
        d, off = self.dense(lo, hi)
        v = ndi.map_coordinates(d, (f - off).T, order=order, mode='constant', cval=0, output=np.float32)
        return v.reshape(q.shape[:-1])

def open_zarr(path, level=0, cache=None):
    """open level `level` of a public OME-zarr (v2) under the open-data bucket, reading chunk size / codec from .zarray"""
    za = SESS.get(f'{B}/{path}/{level}/.zarray', timeout=60).json()
    assert za['chunks'][0] == za['chunks'][1] == za['chunks'][2] and za['dtype'] == '|u1', za
    codec = None if za.get('compressor') is None else numcodecs.get_codec(za['compressor'])
    v = Vol(f'{B}/{path}/{level}', za['chunks'][0], codec, cache); v.shape = za['shape']
    return v
def ct(scroll_vol, cache=None): return open_zarr(scroll_vol, 0, cache)
def m7(scroll_surf, cache=None): return open_zarr(scroll_surf, 0, cache)
CT800 = 'PHerc0800/volumes/20250521135224-8.640um-1.2m-116keV-masked.zarr'
M7800 = 'PHerc0800/representations/predictions/surfaces/20250521135224-surface-20260413222639-surface-m7-L0-th0.2.zarr'
CT306 = 'PHerc0306B/volumes/20250521133212-8.640um-1.2m-116keV-masked.zarr'
M7306 = 'PHerc0306B/representations/predictions/surfaces/20250521133212-surface-20260413222639-surface-m7-L0-th0.2.zarr'
