import sys; SP=r'C:\Users\Raman\AppData\Local\Temp\claude\d--pulzion-dataquest\afa1477c-92e6-453e-bcb3-5e6f28607ff7\scratchpad'; sys.path.insert(0,SP)
from common import *
from scipy.optimize import nnls
from sklearn.model_selection import KFold
import lightgbm as lgb
D=load(); Y=D['Y']; n=len(Y)
def nnls_cv(names,seed=42):
    O=np.stack([np.load(f'{SP}/oof_{m}.npy') for m in names],-1)
    def fitw(O,Y):
        A=np.concatenate([O.reshape(-1,O.shape[-1]),np.ones((O.size//O.shape[-1],1)),-np.ones((O.size//O.shape[-1],1))],1); return nnls(A,Y.ravel())[0]
    ap=lambda w,O:O@w[:-2]+w[-2]-w[-1]
    oo=np.zeros_like(Y)
    for tri,vai in KFold(5,shuffle=True,random_state=seed).split(Y): oo[vai]=ap(fitw(O[tri],Y[tri]),O[vai])
    return ((oo-Y)**2).mean()
v1=['auglgb_v1','direct','ridge','nn']
import shutil
all7=['lgbbag','auglgb','augcat','auglgb_b','direct','ridge','nn']
for nm,L in [('v3 set',all7),('v3+knn',all7+['knn']),('core+knn',['lgbbag','auglgb_b','direct','nn','knn'])]:
    print(nm,[round(nnls_cv(L,s),3) for s in (42,7,123)])
# stacking: per-row LGB on model preds + context features, long format
names=all7+['knn']
O=np.stack([np.load(f'{SP}/oof_{m}.npy') for m in names],-1)
S1=interp(D['tr_S1'])
ctx=np.stack([S1[:,-1],S1[:,-3:].mean(1),S1.std(1),S1.mean(1)],1)
def longX(O,ctx):
    N=O.shape[0]; rows=[]
    for h in range(10): rows.append(np.concatenate([O[:,h,:],np.full((N,1),h),ctx,O[:,h,:].std(1,keepdims=True)],1))
    return np.concatenate(rows)
X=longX(O,ctx); yl=Y.T.ravel(); bi=np.tile(np.arange(n),10)
res=[]
for seed in (42,7,123):
    oo=np.zeros(len(yl))
    for tri,vai in KFold(5,shuffle=True,random_state=seed).split(np.arange(n)):
        mtr=np.isin(bi,tri); mva=np.isin(bi,vai)
        m=lgb.LGBMRegressor(n_estimators=300,learning_rate=0.02,num_leaves=7,min_child_samples=100,subsample=0.8,subsample_freq=1,reg_lambda=10,verbose=-1).fit(X[mtr],yl[mtr])
        oo[mva]=m.predict(X[mva])
    res.append(((oo-yl)**2).mean())
print('lgb stack',np.round(res,3))
