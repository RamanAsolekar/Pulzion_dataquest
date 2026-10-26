import sys; SP=r'C:\Users\Raman\AppData\Local\Temp\claude\d--pulzion-dataquest\afa1477c-92e6-453e-bcb3-5e6f28607ff7\scratchpad'; sys.path.insert(0,SP)
from aug import *
import nn as NNm, lightgbm as lgb, time
from sklearn.model_selection import KFold
which=sys.argv[1]; K=int(sys.argv[2])
D=load(); Y=D['Y']; n=len(Y); nt=len(D['te_S1'])
folds=list(KFold(5,shuffle=True,random_state=0).split(np.arange(n)))
t=time.time()
if which=='lgbbag':
    tr_all=pd.read_pickle(f'{SP}/tr_base_causal.pkl'); te_aug=pd.read_pickle(f'{SP}/te_base_causal.pkl')
    te_fin=build(D['te_S1'],D['te_S2'],np.zeros((nt,10)),only_final=True); drop=['y','bi','tstep']
    def fit(trd,s):
        P=[dict(num_leaves=127,min_child_samples=200,colsample_bytree=0.4,reg_lambda=20,n_estimators=1600,learning_rate=0.015),
           dict(num_leaves=63,min_child_samples=100,colsample_bytree=0.5,reg_lambda=5,n_estimators=1000,learning_rate=0.02)][s%2]
        w=np.where(trd.isY==1,2.0,1.0)
        return lgb.LGBMRegressor(**P,subsample=0.6+0.05*(s%3),subsample_freq=1,verbose=-1,n_jobs=12,random_state=1000+s).fit(trd.drop(columns=drop),trd.y,sample_weight=w)
    oof=np.zeros((n,10)); te=np.zeros((nt,10))
    for tri,vai in folds:
        trd=pd.concat([tr_all[tr_all.bi.isin(tri)],te_aug]); va=tr_all[tr_all.bi.isin(vai)&(tr_all.o==20)]
        for s in range(K): oof[va.bi.values,va.h.values-1]+=fit(trd,s).predict(va.drop(columns=drop))/K
        print('fold',((oof[vai]-Y[vai])**2).mean().round(2),round(time.time()-t),flush=True)
    full=pd.concat([tr_all,te_aug])
    for s in range(K): te[te_fin.bi.values,te_fin.h.values-1]+=fit(full,s).predict(te_fin.drop(columns=drop))/K
else:
    Ztr=np.concatenate([D['tr_S1'],Y],1)
    Xa,Ta,Ma,Fa,Ba=NNm.windows(D['tr_S1'],D['tr_S2'],Ztr,range(4,21))
    Xe,Te,Me,Fe,Be=NNm.windows(D['te_S1'],D['te_S2'],D['te_S1'],range(4,20))
    Xf,_,_,_,Bf=NNm.windows(D['tr_S1'],D['tr_S2'],Ztr,[20])
    Xtf,_,_,_,_=NNm.windows(D['te_S1'],D['te_S2'],np.concatenate([D['te_S1'],np.zeros((nt,10))],1),[20],need_target=False)
    oof=np.zeros((n,10)); te=np.zeros((nt,10))
    for tri,vai in folds:
        s_=np.isin(Ba,tri); args=(np.concatenate([Xa[s_],Xe]),np.concatenate([Ta[s_],Te]),np.concatenate([Ma[s_],Me]),np.concatenate([Fa[s_],Fe]))
        for s in range(K): oof[vai]+=NNm.train_pred(*args,Xf[vai],seed=10+s)/K
        print('fold',((oof[vai]-Y[vai])**2).mean().round(2),round(time.time()-t),flush=True)
    for s in range(K):
        te+=NNm.train_pred(np.concatenate([Xa,Xe]),np.concatenate([Ta,Te]),np.concatenate([Ma,Me]),np.concatenate([Fa,Fe]),Xtf,seed=20+s)/K
np.save(f'{SP}/oof_{which}.npy',oof); np.save(f'{SP}/te_{which}.npy',te)
print(which,'CV',((oof-Y)**2).mean().round(3),((oof-Y)**2).mean(0).round(1),flush=True)
