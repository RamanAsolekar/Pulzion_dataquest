import sys; SP=r'C:\Users\Raman\AppData\Local\Temp\claude\d--pulzion-dataquest\afa1477c-92e6-453e-bcb3-5e6f28607ff7\scratchpad'; sys.path.insert(0,SP)
from common import *
from sklearn.model_selection import KFold
D=load(); Y=D['Y']; n=len(Y); nt=len(D['te_S1'])
folds=list(KFold(5,shuffle=True,random_state=0).split(np.arange(n)))
L1=np.log(interp(D['tr_S1'])); L2=interp(D['tr_S2']); T1=np.log(interp(D['te_S1'])); T2=interp(D['te_S2'])
def emb(A,B,k,decay,b2):
    w=decay**np.arange(k)[::-1]
    return np.concatenate([A[:,-k:]*np.sqrt(w),b2*B[:,-k:]*np.sqrt(w)],1)
def pred(Etr,Ytr,Eq,K,tau,exclude_self=False):
    d=((Eq[:,None,:]-Etr[None])**2).sum(-1)
    if exclude_self: np.fill_diagonal(d,np.inf)
    idx=np.argsort(d,1)[:,:K]; dd=np.take_along_axis(d,idx,1)
    w=np.exp(-(dd-dd[:,:1])/tau); w/=w.sum(1,keepdims=True)
    return (Ytr[idx]*w[...,None]).sum(1)
best=None
for k in (6,10,15,20):
  for decay in (0.8,0.9,1.0):
    for b2 in (0,0.3):
      E=emb(L1,L2,k,decay,b2)
      for K in (20,40,80):
        for tau in (1,3,10):
          p=pred(E,Y,E,K,tau,exclude_self=True); s=((p-Y)**2).mean()
          if best is None or s<best[0]: best=(s,k,decay,b2,K,tau)
print('best loo',best)
s,k,decay,b2,K,tau=best
E=emb(L1,L2,k,decay,b2); Et=emb(T1,T2,k,decay,b2)
oof=np.zeros_like(Y)
for tri,vai in folds: oof[vai]=pred(E[tri],Y[tri],E[vai],K,tau)
te=pred(E,Y,Et,K,tau)
np.save(f'{SP}/oof_knn.npy',oof); np.save(f'{SP}/te_knn.npy',te)
print('knn CV',((oof-Y)**2).mean().round(3))
