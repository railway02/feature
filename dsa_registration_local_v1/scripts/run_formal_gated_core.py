import sys,torch,numpy as np
from pathlib import Path
from torch import nn
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score,average_precision_score
sys.path.insert(0,'/root/autodl-tmp/aneurysm/code/api_png2d_segresnet_cave_main_fusion_v6_strict')
from model import MainFusionModel
OUT=Path(sys.argv[1]); D=np.load(OUT/'feature_master/core781_train.npz,.npz',allow_pickle=True);V=np.load(OUT/'feature_master/core207_valid.npz',allow_pickle=True);dev=torch.device('cuda:0')
def fit(x,t,y,epochs,seed):
 torch.manual_seed(seed);m=MainFusionModel(spatial_dim=1024,temporal_dim=t.shape[1]).to(dev);o=AdamW(m.parameters(),lr=1e-4,weight_decay=1e-3);w=torch.tensor([(len(y)-y.sum())/max(1,y.sum())],device=dev)
 for _ in range(epochs):
  q=torch.randperm(len(y));
  for z in q.split(32):
   o.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(m(torch.tensor(x[z],device=dev),torch.tensor(t[z],device=dev))['main_logit'].ravel(),torch.tensor(y[z],device=dev,dtype=torch.float32),pos_weight=w);loss.backward();o.step()
 return m
for name,t,tv in [('3168_formal_gated',np.c_[D['reg_linear']],np.c_[V['reg_linear']]),('3169_formal_gated',np.c_[D['reg_linear'],D['reg_nonlinear']],np.c_[V['reg_linear'],V['reg_nonlinear']])]:
 y=D['target'].astype(int);f=D['fold'].astype(int);x=D['z2d'].astype('float32');xv=V['z2d'].astype('float32');oof=np.zeros(len(y));epochs=[]
 for k in range(1,6):
  ii=np.where(f!=k)[0];hh=np.where(f==k)[0];m=fit(x[ii],t[ii].astype('float32'),y[ii],35,20260813+k);oof[hh]=torch.sigmoid(m(torch.tensor(x[hh],device=dev),torch.tensor(t[hh],device=dev))['main_logit']).detach().cpu().numpy().ravel();epochs.append(35)
 m=fit(x,t.astype('float32'),y,35,20260813);p=torch.sigmoid(m(torch.tensor(xv,device=dev),torch.tensor(tv.astype('float32'),device=dev))['main_logit']).detach().cpu().numpy().ravel();d=OUT/name;d.mkdir(exist_ok=True);np.savez(d/'FINAL_VALID_ONCE.npz',series_uid=V['series_uid'],probability=p);(d/'metrics.txt').write_text(f'oof_auc={roc_auc_score(y,oof)}\noof_ap={average_precision_score(y,oof)}\nvalid_auc={roc_auc_score(V["target"],p)}\nvalid_ap={average_precision_score(V["target"],p)}\nfinal_epoch=35\n')
