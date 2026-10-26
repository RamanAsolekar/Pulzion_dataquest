import sys; sys.path.insert(0,r'C:\Users\Raman\AppData\Local\Temp\claude\d--pulzion-dataquest\afa1477c-92e6-453e-bcb3-5e6f28607ff7\scratchpad')
from common import *
import torch, torch.nn as nn
from sklearn.model_selection import KFold
torch.set_num_threads(12)
def windows(S1,S2,Z,origins,need_target=True):
    """returns X (N,20,5) left-padded, T (N,10), M mask, isY flags (N,10), bi"""
    n=len(S1); Xs=[];Ts=[];Ms=[];Fs=[];B=[]
    S1n=np.nan_to_num(S1,nan=0.0); S2n=np.nan_to_num(S2,nan=0.0)
    for o in origins:
        x=np.zeros((n,20,5),np.float32)
        L=min(o,20)
        x[:,20-L:,0]=np.log(np.where(np.isnan(S1[:,o-L:o]),1,S1[:,o-L:o])); x[:,20-L:,1]=S2n[:,o-L:o]/2
        x[:,20-L:,2]=np.isnan(S1[:,o-L:o]); x[:,20-L:,3]=np.isnan(S2[:,o-L:o]); x[:,20-L:,4]=1
        T=np.full((n,10),np.nan,np.float32)
        hi=min(o+10,Z.shape[1]); T[:,:hi-o]=Z[:,o:hi]
        F=(np.arange(o,o+10)>=20).astype(np.float32)[None].repeat(n,0)
        Xs.append(x);Ts.append(T);Fs.append(F);B.append(np.arange(n))
    X=np.concatenate(Xs);T=np.concatenate(Ts);F=np.concatenate(Fs);B=np.concatenate(B)
    M=~np.isnan(T)
    keep=M.any(1) if need_target else np.ones(len(X),bool)
    return X[keep],np.nan_to_num(T[keep]),M[keep].astype(np.float32),F[keep],B[keep]
class Net(nn.Module):
    def __init__(s,h=96):
        super().__init__()
        s.gru=nn.GRU(5,h,num_layers=2,batch_first=True,dropout=0.2)
        s.head=nn.Sequential(nn.Linear(h+20*2,128),nn.GELU(),nn.Dropout(0.2),nn.Linear(128,20))
    def forward(s,x):
        o,_=s.gru(x); z=torch.cat([o[:,-1],x[:,:,0],x[:,:,1]],1)
        out=s.head(z); return out[:,:10],out[:,10:]  # pred for S1-type target, delta for Y-type
def train_pred(Xt,Tt,Mt,Ft,Xv,epochs=25,seed=0,wY=2.0):
    torch.manual_seed(seed); np.random.seed(seed)
    net=Net(); opt=torch.optim.AdamW(net.parameters(),2e-3,weight_decay=1e-4)
    N=len(Xt); bs=256; steps=epochs*(N//bs); sch=torch.optim.lr_scheduler.OneCycleLR(opt,3e-3,total_steps=steps)
    Xt=torch.tensor(Xt);Tt=torch.tensor(Tt);Mt=torch.tensor(Mt);Ft=torch.tensor(Ft)
    W=Mt*(1+(wY-1)*Ft)
    net.train(); k=0
    for ep in range(epochs):
        perm=torch.randperm(N)
        for i in range(0,N-bs+1,bs):
            idx=perm[i:i+bs]; a,b=net(Xt[idx]); p=a+Ft[idx]*b
            loss=((p-Tt[idx])**2*W[idx]).sum()/W[idx].sum()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step(); k+=1
    net.eval()
    with torch.no_grad():
        a,b=net(torch.tensor(Xv)); return (a+b).numpy()
if __name__=='__main__':
    D=load(); Y=D['Y']; n=len(Y)
    Ztr=np.concatenate([D['tr_S1'],Y],1)
    Xa,Ta,Ma,Fa,Ba=windows(D['tr_S1'],D['tr_S2'],Ztr,range(4,21))
    Xe,Te,Me,Fe,Be=windows(D['te_S1'],D['te_S2'],D['te_S1'],range(4,20))
    Xf,_,_,_,Bf=windows(D['tr_S1'],D['tr_S2'],Ztr,[20])
    oof=np.zeros((n,10))
    for tri,vai in KFold(5,shuffle=True,random_state=0).split(np.arange(n)):
        s=np.isin(Ba,tri)
        p=train_pred(np.concatenate([Xa[s],Xe]),np.concatenate([Ta[s],Te]),np.concatenate([Ma[s],Me]),np.concatenate([Fa[s],Fe]),Xf[vai])
        oof[vai]=p; print('fold',((p-Y[vai])**2).mean(),flush=True)
    print('NN',((oof-Y)**2).mean(), ((oof-Y)**2).mean(0).round(1))
    np.save(r'C:\Users\Raman\AppData\Local\Temp\claude\d--pulzion-dataquest\afa1477c-92e6-453e-bcb3-5e6f28607ff7\scratchpad\oof_nn_r0.npy',oof)
