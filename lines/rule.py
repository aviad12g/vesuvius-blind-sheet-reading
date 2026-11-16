"""rule.py <face,face,...> <strips.npz> [...] : text-line spacing rule. Env WD (window width, cells, default 250), COV (min row coverage, default 0.35).
HIT = two neighbouring windows whose strongest autocorrelation peaks (lags 73-214 cells) agree within 6 cells, both >= 0.55."""
import sys, numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
MM=4*0.009362
def acf(p,KM=300):
    q=gaussian_filter1d(p,3); q=q-uniform_filter1d(q,241,mode='nearest'); q=(q-q.mean())/(q.std()+1e-9); L=len(q)
    ac=np.full(KM+1,np.nan)
    for k in range(0,min(KM,L-150)+1): ac[k]=(q[:L-k]*q[k:]).mean()
    return ac
import os
WD=int(os.environ.get('WD',250)); COV=float(os.environ.get('COV',0.35))
def sheetwins(S,M,Wd=None):
    Wd=WD
    out=[]
    for x0 in range(0,M.shape[1]-Wd+1,Wd//2):
        m=M[:,x0:x0+Wd]; n=m.sum(1); idx=np.where(n>=COV*Wd)[0]
        if len(idx)<420: continue
        runs=np.split(idx,np.where(np.diff(idx)>1)[0]+1); r=max(runs,key=len)
        if len(r)<420: continue
        ac=acf((S[:,x0:x0+Wd]*m).sum(1)[r]/n[r]); k=int(np.nanargmax(ac[70:221]))+70; out.append((x0,k,float(ac[k]),int(r[0]),int(r[-1])))
    return out
for path in sys.argv[2:]:
    z=np.load(path,allow_pickle=True); ids=list(z['ids'])
    for f in sys.argv[1].split(','):
        if f'S0_{f}' not in z.files: continue
        nw=0; hits=[]; strong=0
        for k,i in enumerate(ids):
            w=sheetwins(z[f'S{k}_{f}'],z[f'M{k}']); nw+=len(w); strong+=sum(1 for t in w if t[2]>=0.55)
            for a,b in zip(w[:-1],w[1:]):
                if 73<=a[1]<=214 and 73<=b[1]<=214 and b[0]-a[0]==WD//2 and abs(a[1]-b[1])<=6 and min(a[2],b[2])>=0.55: hits.append((i,a[0],a[1],a[2],b[1],b[2],a[3],b[4]))
        print('%-22s %-15s sheets %2d windows %3d | windows with peak>=0.55: %2d | adjacent pairs agreeing: %d'%(path.split('/')[-1],f,len(ids),nw,strong,len(hits)))
        for h in hits: print('      HIT sheet %s cols %d-%d  spacing %d/%d cells = %.2f mm  peak %.2f/%.2f  rows %d-%d'%(h[0],h[1],h[1]+WD+WD//2,h[2],h[4],0.5*(h[2]+h[4])*MM,h[3],h[5],h[6],h[7]))
