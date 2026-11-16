"""st2.py <scroll> <band,band,...> <face,face,...> <out.npz> : stitch matched sheets into tall strips (needs >=3 consecutive z-bands)."""
import sys, os, glob, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from st import load_band, match
from PIL import Image
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
scroll=sys.argv[1]; names=sys.argv[2].split(','); faces=sys.argv[3].split(','); out=sys.argv[4]
B=[load_band(scroll,n) for n in names]
# graph
par={}
def fnd(x):
    while par.get(x,x)!=x: x=par[x]
    return x
edges={}
for i in range(len(B)-1):
    for (a,b,n,dist,dr,dc,sdc) in match(B[i],B[i+1]):
        if sdc>12: continue
        edges[((i,a),(i+1,b))]=(dr,dc,n)
        ra,rb=fnd((i,a)),fnd((i+1,b))
        if ra!=rb: par[ra]=rb
comps={}
for (u,v) in edges:
    comps.setdefault(fnd(u),set()).update([u,v])
adj={}
for (u,v),(dr,dc,n) in edges.items():
    adj.setdefault(u,[]).append((v,-dr,-dc)); adj.setdefault(v,[]).append((u,dr,dc))   # offset to add to v's coords to land in u frame: r_u = r_v - dr
def maps_of(i,c,face):
    return np.asarray(Image.open(f"{B[i]['d']}/maps/c{c:02d}_{face}.png")).astype(np.float32)
stitched=[]
for root,mem in comps.items():
    bands=sorted({m[0] for m in mem})
    if len(bands)<3: continue
    start=min(mem); off={start:(0,0)}; q=[start]
    while q:
        u=q.pop()
        for (v,dr,dc) in adj.get(u,[]):
            if v not in off: off[v]=(off[u][0]+dr,off[u][1]+dc); q.append(v)
    r0=min(o[0] for o in off.values()); c0=min(o[1] for o in off.values())
    Hh=int(max(o[0]-r0+B[k[0]]['occ'].shape[1] for k,o in off.items()))+1; Ww=int(max(o[1]-c0+B[k[0]]['occ'].shape[2] for k,o in off.items()))+1
    acc={f:np.zeros((Hh,Ww),np.float32) for f in faces}; cnt=np.zeros((Hh,Ww),np.float32)
    for (i,c),(orr,occ_) in off.items():
        rr=int(round(orr-r0)); cc=int(round(occ_-c0)); m=B[i]['occ'][c]; h,w=m.shape
        cnt[rr:rr+h,cc:cc+w]+=m
        for f in faces:
            a=maps_of(i,c,f); acc[f][rr:rr+h,cc:cc+w]+=np.where(m,(a-1)/254.,0)
    S={f:acc[f]/np.maximum(cnt,1) for f in faces}; M=cnt>0
    stitched.append(dict(id='%s_c%02d'%(names[start[0]],start[1]),members=sorted(mem),M=M,S=S,bands=len(bands)))
    print('members',stitched[-1]['id'],[(names[i][:5],c) for i,c in sorted(mem)])
    print('sheet',stitched[-1]['id'],'bands',len(bands),'pieces',len(mem),'size',M.shape,'covered cells',int(M.sum()),'= %.1f cm2'%(M.sum()*(4*9.362e-4)**2))
np.savez_compressed(out,ids=[s['id'] for s in stitched],**{f'M{k}':s['M'] for k,s in enumerate(stitched)},**{f'S{k}_{f}':s['S'][f] for k,s in enumerate(stitched) for f in faces})
