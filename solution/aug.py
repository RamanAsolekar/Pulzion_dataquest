import sys; sys.path.insert(0,r'C:\Users\Raman\AppData\Local\Temp\claude\d--pulzion-dataquest\afa1477c-92e6-453e-bcb3-5e6f28607ff7\scratchpad')
from common import *
def build(S1,S2,Y=None,origins=range(4,21),only_final=False):
    """rows: (block idx, origin o, horizon h). features relative to origin."""
    n=len(S1); S1i=interp(S1)
    Z=S1 if Y is None else np.concatenate([S1,Y],1)
    rows=[];tg=[];meta=[]
    for o in ([20] if only_final else origins):
        lag1=np.full((n,20),np.nan); lag2=np.full((n,20),np.nan); lagi=np.full((n,20),np.nan)
        hist=interp(S1[:,:o])  # causal: only values before origin
        lag1[:,:o]=S1[:,o-1::-1][:,:20]; lag2[:,:o]=S2[:,o-1::-1][:,:20]; lagi[:,:o]=hist[:,::-1][:,:20]
        base=dict()
        for j in range(20): base[f'l1_{j}']=lag1[:,j]; base[f'l2_{j}']=lag2[:,j]; base[f'li_{j}']=lagi[:,j]
        base['o']=np.full(n,o); base['mean']=hist.mean(1); base['std']=hist.std(1); base['max']=hist.max(1); base['min']=hist.min(1)
        base['med']=np.median(hist,1); base['q25']=np.percentile(hist,25,1)
        base['m2mean']=np.nanmean(S2[:,:o],1)
        for w in (3,5,10): base[f'last{w}']=hist[:,-w:].mean(1)
        hi=hist>np.maximum(np.minimum(hist.min(1)*2,6),5)[:,None]
        base['frac_hi']=hi.mean(1)
        rl=np.zeros(n)
        for i in range(n):
            k=0
            while k<o and hi[i,o-1-k]==hi[i,o-1]: k+=1
            rl[i]=k
        base['runlen']=rl; base['cur_hi']=hi[:,-1].astype(float)
        base['tsl_hi']=np.array([(o-1-np.where(h)[0].max()) if h.any() else 30 for h in hi])
        base['n_onsets']=(hi[:,1:]&~hi[:,:-1]).sum(1)
        bdf=pd.DataFrame(base)
        for h in range(1,11):
            t=o+h-1
            if t>=Z.shape[1]: break
            df=bdf.copy(); df['h']=h; df['isY']=float(t>=20); df['tstep']=t
            df['bi']=np.arange(n)
            df['y']=Z[:,t]
            rows.append(df)
    out=pd.concat(rows,ignore_index=True)
    return out[~out.y.isna()].reset_index(drop=True) if Y is not None or not only_final else out

def regime_feats(hist, thr=6.0):
    """semi-Markov on/off burst descriptors of an interpolated history (n,o)."""
    n,o=hist.shape; out=np.full((n,20),np.nan)
    for i in range(n):
        x=hist[i]; s=x>thr
        # segments
        seg=[]; st=0
        for t in range(1,o+1):
            if t==o or s[t]!=s[st]: seg.append((s[st],st,t)); st=t
        cs,cst,cen=seg[-1]; cur=x[cst:cen]
        on=[e-b for v,b,e in seg[1:-1] if v]; off=[e-b for v,b,e in seg[1:-1] if not v]
        onsets=[b for v,b,e in seg if v and b>0]
        prev_on=[(b,e) for v,b,e in seg[:-1] if v]
        prev_off=[(b,e) for v,b,e in seg[:-1] if not v]
        lvl=[x[b:e].mean() for v,b,e in seg if v]
        out[i,0]=cs; out[i,1]=cen-cst; out[i,2]=float(cst==0)
        out[i,3]=cur.mean(); out[i,4]=cur.max(); out[i,5]=cur[-1]-cur.mean()
        out[i,6]=np.mean(on) if on else np.nan; out[i,7]=np.mean(off) if off else np.nan
        out[i,8]=(prev_on[-1][1]-prev_on[-1][0]) if prev_on else np.nan
        out[i,9]=(prev_off[-1][1]-prev_off[-1][0]) if prev_off else np.nan
        out[i,10]=np.mean(np.diff(onsets)) if len(onsets)>1 else np.nan
        out[i,11]=(o-onsets[-1]) if onsets else np.nan
        out[i,12]=(out[i,10]-out[i,11]) if len(onsets)>1 else np.nan
        out[i,13]=np.mean(lvl) if lvl else np.nan
        out[i,14]=(prev_on[-1][1]-prev_on[-1][0]) if prev_on else np.nan
        out[i,15]=np.mean([x[b:e].mean() for b,e in prev_on]) if prev_on else np.nan
        out[i,16]=len(on)+len(off)
        out[i,17]=out[i,1]-(out[i,6] if cs else out[i,7]) if (on if cs else off) else np.nan  # elapsed minus typical
        lowv=x[~s]; out[i,18]=lowv.mean() if len(lowv) else np.nan; out[i,19]=lowv.std() if len(lowv)>1 else np.nan
    names=['rg_state','rg_run','rg_cens','rg_curmean','rg_curmax','rg_curdev','rg_meanon','rg_meanoff','rg_prevon','rg_prevoff',
           'rg_onint','rg_since_onset','rg_next_onset','rg_hilvl','rg_lastonlen','rg_prevonlvl','rg_nseg','rg_elapsed_rel','rg_lowlvl','rg_lowstd']
    return pd.DataFrame(out,columns=names)

def build2(S1,S2,Y=None,origins=range(4,21),only_final=False):
    df=build(S1,S2,Y,origins,only_final)
    parts={}
    for o in sorted(df.o.unique()):
        parts[o]=regime_feats(interp(S1[:,:o]))
    R=np.zeros((len(df),20))
    for o,g in df.groupby('o').groups.items():
        R[np.asarray(g)]=parts[o].values[df.loc[g,'bi'].values]
    R=pd.DataFrame(R,columns=parts[list(parts)[0]].columns)
    df=pd.concat([df.reset_index(drop=True),R],axis=1)
    # horizon-interaction features
    df['h_minus_next_onset']=df.h-df.rg_next_onset
    df['run_plus_h']=df.rg_run+df.h
    return df
