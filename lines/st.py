"""st.py : load one swept band (tracked sheets + low-res maps), crop band edges, match sheets across neighbouring z-bands by 3D distance."""
import sys, os, glob, json, numpy as np, tifffile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); import geom
from PIL import Image
from scipy.spatial import cKDTree
MR=os.environ.get('MESHES','pmesh/meshes')          # base meshes (tifxyz), e.g. pscamillo/vesuvius-eligible-meshes/meshes
def find(scroll, name):
    root=os.environ.get('ROOT','out/sweep')          # sweep output root: ROOT/<scroll>/<mesh> (or ROOT/*/sweep/<scroll>/<mesh>)
    hits=glob.glob(f'{root}/{scroll}/{name}')+glob.glob(f'{root}/*/sweep/{scroll}/{name}')
    if not hits: raise SystemExit(f'no sweep output for {scroll}/{name} under {root}')
    return hits[0]
def load_band(scroll, name, facepat='studio_forward'):
    d=find(scroll,name); cz=np.load(d+'/canv.npz'); H=cz['H'].astype(np.float32); D=float(cz['D']); ls=int(cz['ls'])
    gz,gy,gx,gm=geom.read_grids(f'{MR}/{scroll}/{name}'); P,N,valid=geom.from_grids(gz,gy,gx,gm)
    P4=P[::ls,::ls][:H.shape[1],:H.shape[2]]; N4=N[::ls,::ls][:H.shape[1],:H.shape[2]]
    occ=[]
    for c in range(H.shape[0]):
        a=np.asarray(Image.open(f'{d}/maps/c{c:02d}_{facepat}.png')); occ.append(a>0)
    occ=np.stack(occ); occ[:,:48]=False; occ[:,-28:]=False
    X=P4[None]-(H-D)[...,None]*N4[None]          # (nc,h,w,3) zyx
    return dict(name=name,d=d,X=X,occ=occ,H=H,D=D)
def match(A,B,thr=4.0,minc=250):
    pts=[];lab=[];rc=[]
    for b in range(B['X'].shape[0]):
        r,c=np.where(B['occ'][b]); pts.append(B['X'][b][r,c]); lab.append(np.full(len(r),b)); rc.append(np.stack([r,c],1))
    pts=np.concatenate(pts); lab=np.concatenate(lab); rc=np.concatenate(rc); T=cKDTree(pts); out=[]
    for a in range(A['X'].shape[0]):
        r,c=np.where(A['occ'][a])
        if len(r)==0: continue
        dd,ii=T.query(A['X'][a][r,c],distance_upper_bound=thr); ok=np.isfinite(dd)
        if ok.sum()<minc: continue
        l=lab[ii[ok]]; b=np.bincount(l).argmax(); sel=l==b
        if sel.sum()<minc: continue
        dr=np.median(rc[ii[ok]][sel][:,0]-r[ok][sel]); dc=np.median(rc[ii[ok]][sel][:,1]-c[ok][sel]); sdc=np.std(rc[ii[ok]][sel][:,1]-c[ok][sel])
        out.append((a,int(b),int(sel.sum()),float(np.median(dd[ok][sel])),float(dr),float(dc),float(sdc)))
    return out
if __name__=='__main__':
    scroll=sys.argv[1]; names=sys.argv[2].split(',')
    B=[load_band(scroll,n) for n in names]
    for b in B:
        z=b['X'][...,0][b['occ']]; print(b['name'],'canvases',b['X'].shape[0],'grid',b['X'].shape[1:3],'z range %.0f..%.0f'%(z.min(),z.max()),'occ cells',int(b['occ'].sum()))
    for i in range(len(B)-1):
        m=match(B[i],B[i+1]); print(B[i]['name'],'->',B[i+1]['name'],len(m),'matches')
        for t in m: print('   c%02d -> c%02d  cells %5d  dist %.1f  drow %+.0f dcol %+.0f (sd %.1f)'%t)
