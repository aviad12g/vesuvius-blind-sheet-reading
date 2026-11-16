"""an.py <strips.npz> <face,face,...> [window] : pooled autocorrelation of row profiles per face (diagnostic)."""
import sys, numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
z=np.load(sys.argv[1],allow_pickle=True); faces=sys.argv[2].split(','); ids=list(z['ids'])
Wd=int(sys.argv[3]) if len(sys.argv)>3 else 250; MINL=420; KM=300; MM=4*0.009362
def wins(S,M):
    H,W=S.shape; out=[]
    for x0 in range(0,W-Wd+1,Wd//2):
        m=M[:,x0:x0+Wd]; n=m.sum(1); ok=n>=0.35*Wd
        idx=np.where(ok)[0]
        if len(idx)<MINL: continue
        runs=np.split(idx,np.where(np.diff(idx)>1)[0]+1); r=max(runs,key=len)
        if len(r)<MINL: continue
        p=(S[:,x0:x0+Wd]*m).sum(1)[r]/n[r]
        q=gaussian_filter1d(p,3); q=q-uniform_filter1d(q,241,mode='nearest'); q=(q-q.mean())/(q.std()+1e-9); L=len(q)
        ac=np.full(KM+1,np.nan)
        for k in range(0,min(KM,L-150)+1): ac[k]=(q[:L-k]*q[k:]).mean()
        out.append((x0,L,ac))
    return out
res={}
for f in faces:
    A=[];who=[]
    for k,i in enumerate(ids):
        for (x0,L,ac) in wins(z[f'S{k}_{f}'],z[f'M{k}']): A.append(ac); who.append(k)
    A=np.array(A); who=np.array(who); res[f]=(A,who)
    if len(A)==0: print(f,'no windows'); continue
    pooled=np.nanmean(A,0); seg=pooled[70:221]; kp=int(np.nanargmax(seg))+70
    pk=np.array([int(np.nanargmax(a[70:221]))+70 for a in A]); pv=np.array([np.nanmax(a[70:221]) for a in A])
    rng=np.random.default_rng(0); u=np.unique(who)
    bs=[]
    for _ in range(2000):
        s=rng.choice(u,len(u)); bs.append(np.nanmean(np.concatenate([A[who==t] for t in s]),0)[kp])
    hist=np.histogram(pk,bins=[70,90,110,130,150,170,190,221])[0]
    print('%-16s windows %3d sheets %2d | pooled side peak at %3d cells = %.2f mm: %.3f [%.3f,%.3f] | min before peak %.3f | 2x lag %.3f'%(f,len(A),len(u),kp,kp*MM,pooled[kp],np.percentile(bs,2.5),np.percentile(bs,97.5),np.nanmin(pooled[20:kp]),pooled[min(2*kp,KM)]))
    print('    pooled ACF every 10 cells:',' '.join('%+.2f'%pooled[k] for k in range(0,KM+1,10)))
    print('    window peak-lag histogram (cells 70-90,-110,-130,-150,-170,-190,-220):',hist.tolist(),' median peak value %.2f'%np.median(pv))
