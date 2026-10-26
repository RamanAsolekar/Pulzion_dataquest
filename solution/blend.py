import sys; SP=r'C:\Users\Raman\AppData\Local\Temp\claude\d--pulzion-dataquest\afa1477c-92e6-453e-bcb3-5e6f28607ff7\scratchpad'; sys.path.insert(0,SP)
from common import *
from scipy.optimize import nnls
from sklearn.model_selection import KFold
D=load(); Y=D['Y']
names=[m for m in ['lgbbag','auglgb_b','direct','nn','knn'] if __import__('os').path.exists(f'{SP}/te_{m}.npy')]
O=np.stack([np.load(f'{SP}/oof_{m}.npy') for m in names],-1); T=np.stack([np.load(f'{SP}/te_{m}.npy') for m in names],-1)
for i,m in enumerate(names): print(m, round(((O[...,i]-Y)**2).mean(),3))
def fitw(O,Y):
    A=np.concatenate([O.reshape(-1,O.shape[-1]),np.ones((O.size//O.shape[-1],1))],1)
    # intercept via centering trick: allow +/- intercept with two columns
    A=np.concatenate([A,-A[:,-1:]],1); w,_=nnls(A,Y.ravel()); return w
def apply(w,O): return O@w[:-2]+w[-2]-w[-1]
w=fitw(O,Y); print('weights',dict(zip(names+['b+','b-'],w.round(3))),'in-sample',round(((apply(w,O)-Y)**2).mean(),3))
# nested check
oo=np.zeros_like(Y)
for tri,vai in KFold(5,shuffle=True,random_state=42).split(Y): oo[vai]=apply(fitw(O[tri],Y[tri]),O[vai])
print('blend nested CV',round(((oo-Y)**2).mean(),3), ((oo-Y)**2).mean(0).round(1))
pred=np.clip(apply(w,T),0,None)
te=pd.read_csv(r'D:\pulzion_dataquest\test.csv',dtype={'id':str}); te['block']=te.id.str[:3]
f=te[te.step>=20].copy()
bmap={b:i for i,b in enumerate(D['te_blocks'])}
f['colony_power_demand_kw']=[pred[bmap[b],s-20] for b,s in zip(f.block,f.step)]
ss=pd.read_csv(r'D:\pulzion_dataquest\sample_submission.csv',dtype={'id':str})
sub=ss[['id']].merge(f[['id','colony_power_demand_kw']],on='id',how='left')
assert len(sub)==len(ss) and sub.colony_power_demand_kw.notna().all()
sub.to_csv(r'D:\pulzion_dataquest\submission.csv',index=False); print(sub.head(), sub.describe())
