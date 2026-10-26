import sys; SP=r'C:\Users\Raman\AppData\Local\Temp\claude\d--pulzion-dataquest\afa1477c-92e6-453e-bcb3-5e6f28607ff7\scratchpad'; sys.path.insert(0,SP)
from aug import *
import nn as NNm
import lightgbm as lgb, time, os
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
which=sys.argv[1]
D=load(); Y=D['Y']; n=len(Y); nt=len(D['te_S1'])
folds=list(KFold(5,shuffle=True,random_state=0).split(np.arange(n)))
def save(name,oof,te): np.save(f'{SP}/oof_{name}.npy',oof); np.save(f'{SP}/te_{name}.npy',te); print(name,'CV',round(((oof-Y)**2).mean(),3),((oof-Y)**2).mean(0).round(1),flush=True)
if which=='auglgb':
    tr_all=build(D['tr_S1'],D['tr_S2'],Y)
    te_aug=build(D['te_S1'],D['te_S2'],None,origins=range(4,20)); te_aug=te_aug[~te_aug.y.isna()]
    te_fin=build(D['te_S1'],D['te_S2'],np.zeros((nt,10)),only_final=True)
    drop=['y','bi','tstep']
    P=dict(n_estimators=1000,learning_rate=0.02,num_leaves=63,min_child_samples=100,subsample=0.7,subsample_freq=1,colsample_bytree=0.5,reg_lambda=5,verbose=-1,n_jobs=12)
    def fit(trd,seed):
        w=np.where(trd.isY==1,2.0,1.0); return lgb.LGBMRegressor(**P,random_state=seed).fit(trd.drop(columns=drop),trd.y,sample_weight=w)
    oof=np.zeros((n,10))
    for tri,vai in folds:
        m=fit(pd.concat([tr_all[tr_all.bi.isin(tri)],te_aug]),0)
        va=tr_all[tr_all.bi.isin(vai)&(tr_all.o==20)]; oof[va.bi.values,va.h.values-1]=m.predict(va.drop(columns=drop))
    te=np.zeros((nt,10)); full=pd.concat([tr_all,te_aug])
    for s in range(3):
        m=fit(full,s); te[te_fin.bi.values,te_fin.h.values-1]+=m.predict(te_fin.drop(columns=drop))/3
    save('auglgb',oof,te)
elif which in ('direct','ridge'):
    from common import feats
    X=feats(D['tr_S1'],D['tr_S2']); Xt=feats(D['te_S1'],D['te_S2'])
    if which=='direct':
        P=dict(n_estimators=300,learning_rate=0.02,num_leaves=8,min_child_samples=20,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=5,verbose=-1)
        f=lambda a,b,c: np.stack([np.mean([lgb.LGBMRegressor(**P,random_state=s).fit(a,b[:,k]).predict(c) for s in range(3)],0) for k in range(10)],1)
    else:
        keep=[c for c in X.columns if not ((c.startswith('s1_') or c.startswith('s2_')) and c[3:].isdigit())]
        def f(a,b,c):
            sc=StandardScaler().fit(a[keep]); return RidgeCV(alphas=np.logspace(-1,4,20)).fit(sc.transform(a[keep]),b).predict(sc.transform(c[keep]))
    oof=np.zeros((n,10))
    for tri,vai in folds: oof[vai]=f(X.iloc[tri],Y[tri],X.iloc[vai])
    save(which,oof,f(X,Y,Xt))
elif which=='nn':
    Ztr=np.concatenate([D['tr_S1'],Y],1)
    Xa,Ta,Ma,Fa,Ba=NNm.windows(D['tr_S1'],D['tr_S2'],Ztr,range(4,21))
    Xe,Te,Me,Fe,Be=NNm.windows(D['te_S1'],D['te_S2'],D['te_S1'],range(4,20))
    Xf,_,_,_,Bf=NNm.windows(D['tr_S1'],D['tr_S2'],Ztr,[20])
    Xtf,_,_,_,_=NNm.windows(D['te_S1'],D['te_S2'],np.concatenate([D['te_S1'],np.zeros((nt,10))],1),[20],need_target=False)
    oof=np.load(f'{SP}/oof_nn_r0.npy')  # same folds (KFold seed 0), already computed
    te=np.zeros((nt,10))
    for s in range(3):
        te+=NNm.train_pred(np.concatenate([Xa,Xe]),np.concatenate([Ta,Te]),np.concatenate([Ma,Me]),np.concatenate([Fa,Fe]),Xtf,seed=s)/3
        print('nn seed',s,flush=True)
    save('nn',oof,te)
