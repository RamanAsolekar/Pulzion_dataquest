import pandas as pd, numpy as np
def load():
    tr=pd.read_csv('train.csv',dtype={'id':str}); te=pd.read_csv('test.csv',dtype={'id':str})
    out={}
    for name,d in (('tr',tr),('te',te)):
        d['block']=d.id.str[:3]
        H=d[d.step<20]
        S1=H.pivot(index='block',columns='step',values='sensor_power')
        S2=H.pivot(index='block',columns='step',values='sensor_power_2')
        out[name+'_blocks']=S1.index.values
        out[name+'_S1']=S1.values; out[name+'_S2']=S2.values
    F=tr[tr.step>=20]; Y=F.pivot(index='block',columns='step',values='colony_power_demand_kw')
    assert (Y.index.values==out['tr_blocks']).all()
    out['Y']=Y.values
    return out
def interp(S):
    S=pd.DataFrame(S).T.interpolate(limit_direction='both').T.values
    return S
def feats(S1,S2):
    S1i=interp(S1); S2i=interp(S2)
    f={}
    for j in range(20):
        f[f's1_{j}']=S1[:,j]; f[f's2_{j}']=S2[:,j]; f[f's1i_{j}']=S1i[:,j]
    L1=np.log(S1i)
    for name,A in (('s1',S1i),('s2',S2i),('l1',L1)):
        f[name+'_mean']=A.mean(1); f[name+'_std']=A.std(1); f[name+'_max']=A.max(1); f[name+'_min']=A.min(1)
        f[name+'_med']=np.median(A,1)
        for w in (3,5,10): f[f'{name}_last{w}']=A[:,-w:].mean(1)
    f['s1_q10']=np.percentile(S1i,10,1); f['s1_q25']=np.percentile(S1i,25,1); f['s1_q75']=np.percentile(S1i,75,1)
    f['s1_argmax']=S1i.argmax(1); f['s1_d1']=S1i[:,19]-S1i[:,18]; f['s1_d2']=S1i[:,18]-S1i[:,17]
    f['nan1']=np.isnan(S1).sum(1); f['nan2']=np.isnan(S2).sum(1)
    # run length of current state (high vs low)
    thr=np.minimum(S1i.min(1)*2,6)[:,None]; hi=S1i>np.maximum(thr,5)
    f['frac_hi']=hi.mean(1)
    rl=np.zeros(len(S1i))
    for i in range(len(S1i)):
        k=0
        while k<20 and hi[i,19-k]==hi[i,19]: k+=1
        rl[i]=k
    f['runlen']=rl; f['cur_hi']=hi[:,19].astype(float)
    # time since last hi
    tsl=np.array([ (19-np.where(h)[0].max()) if h.any() else 25 for h in hi]); f['tsl_hi']=tsl
    tfh=np.array([ (np.where(h)[0].min()) if h.any() else 25 for h in hi]); f['first_hi']=tfh
    # number of burst onsets and mean interval
    on=(hi[:,1:]&~hi[:,:-1]).sum(1); f['n_onsets']=on
    for k in range(1,8): f[f's2s1_{k}']=S2i[:,-k]*S1i[:,-k]
    return pd.DataFrame(f)
