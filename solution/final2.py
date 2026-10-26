import sys; SP=r'C:\Users\Raman\AppData\Local\Temp\claude\d--pulzion-dataquest\afa1477c-92e6-453e-bcb3-5e6f28607ff7\scratchpad'; sys.path.insert(0,SP)
from aug import *
import lightgbm as lgb, time
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
which=sys.argv[1]
D=load(); Y=D['Y']; n=len(Y); nt=len(D['te_S1'])
folds=list(KFold(5,shuffle=True,random_state=0).split(np.arange(n)))
tr_all=pd.read_pickle(f'{SP}/tr_base_causal.pkl'); te_aug=pd.read_pickle(f'{SP}/te_base_causal.pkl')
te_fin=build(D['te_S1'],D['te_S2'],np.zeros((nt,10)),only_final=True)
drop=['y','bi','tstep']
def mk(seed):
    if which=='auglgb':
        return lgb.LGBMRegressor(n_estimators=1000,learning_rate=0.02,num_leaves=63,min_child_samples=100,subsample=0.7,subsample_freq=1,colsample_bytree=0.5,reg_lambda=5,verbose=-1,n_jobs=12,random_state=seed)
    if which=='auglgb_b':  # different flavour: deeper regularised, more feature subsampling, wY via isY
        return lgb.LGBMRegressor(n_estimators=1600,learning_rate=0.015,num_leaves=127,min_child_samples=200,subsample=0.6,subsample_freq=1,colsample_bytree=0.4,reg_lambda=20,min_split_gain=0.0,verbose=-1,n_jobs=12,random_state=seed+100)
    if which=='augcat':
        return CatBoostRegressor(iterations=2000,learning_rate=0.05,depth=7,l2_leaf_reg=5,verbose=0,thread_count=12,random_seed=seed)
def fit(trd,seed):
    w=np.where(trd.isY==1,2.0,1.0); return mk(seed).fit(trd.drop(columns=drop),trd.y,sample_weight=w)
t=time.time(); oof=np.zeros((n,10))
for tri,vai in folds:
    m=fit(pd.concat([tr_all[tr_all.bi.isin(tri)],te_aug]),0)
    va=tr_all[tr_all.bi.isin(vai)&(tr_all.o==20)]; oof[va.bi.values,va.h.values-1]=m.predict(va.drop(columns=drop))
    print(which,'fold',((oof[vai]-Y[vai])**2).mean().round(2),round(time.time()-t),flush=True)
te=np.zeros((nt,10)); full=pd.concat([tr_all,te_aug])
for s in range(3):
    m=fit(full,s); te[te_fin.bi.values,te_fin.h.values-1]+=m.predict(te_fin.drop(columns=drop))/3
np.save(f'{SP}/oof_{which}.npy',oof); np.save(f'{SP}/te_{which}.npy',te)
print(which,'CV',((oof-Y)**2).mean().round(3),((oof-Y)**2).mean(0).round(1),flush=True)
