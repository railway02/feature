#!/usr/bin/env python3
"""Strict Train781 CV + one-full-train/one-Valid207 outcome fits."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix

def met(y,p,t=.5):
 q=(p>=t).astype(int);tn,fp,fn,tp=confusion_matrix(y,q,labels=[0,1]).ravel()
 return {'ROC_AUC':float(roc_auc_score(y,p)),'AUPRC':float(average_precision_score(y,p)),'sensitivity':tp/(tp+fn) if tp+fn else 0.,'specificity':tn/(tn+fp) if tn+fp else 0.,'accuracy':float((q==y).mean())}
def main():
 a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);args=a.parse_args();out=args.out
 tr=np.load(out/'feature_master/core781_train.npz,.npz',allow_pickle=True);va=np.load(out/'feature_master/core207_valid.npz',allow_pickle=True)
 y=tr['target'].astype(int);fold=tr['fold'].astype(int)
 schemas={'3167':(tr['z2d'],va['z2d']),'reg_only_linear':(tr['reg_linear'],va['reg_linear']),'reg_only_full':(np.c_[tr['reg_linear'],tr['reg_nonlinear']],np.c_[va['reg_linear'],va['reg_nonlinear']]),'3168':(np.c_[tr['z2d'],tr['reg_linear']],np.c_[va['z2d'],va['reg_linear']]),'3169':(np.c_[tr['z2d'],tr['reg_linear'],tr['reg_nonlinear']],np.c_[va['z2d'],va['reg_linear'],va['reg_nonlinear']])}
 summary=[]
 for name,(x,xv) in schemas.items():
  d=out/name;d.mkdir(exist_ok=True);o=np.full(len(y),np.nan);rows=[]
  for f in range(1,6):
   dev=np.where(fold!=f)[0];hold=np.where(fold==f)[0];m=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=5000,random_state=20260820))
   m.fit(x[dev],y[dev]);o[hold]=m.predict_proba(x[hold])[:,1];rows.append({'fold':f,**met(y[hold],o[hold]),'n':len(hold)})
  pd.DataFrame({'series_uid':tr['series_uid'],'patient_id':tr['patient_id'],'target':y,'fold':fold,'probability':o}).to_csv(d/'TRAIN_OOF_PREDICTIONS.csv',index=False);pd.DataFrame(rows).to_csv(d/'TRAIN_FOLD_METRICS.csv',index=False)
  # Final model: one full-Train fit and precisely one Valid probability vector.
  m=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=5000,random_state=20260820));m.fit(x,y);p=m.predict_proba(xv)[:,1]
  pd.DataFrame({'series_uid':va['series_uid'],'patient_id':va['patient_id'],'target':va['target'],'probability':p}).to_csv(d/'FINAL_VALID_PREDICTIONS.csv',index=False)
  fm=met(va['target'].astype(int),p);json.dump({'model':name,'train_oof':met(y,o),'folds':rows,'final_valid207':fm,'valid_inference_count':1,'outcome_fold_models_applied_to_valid':False},open(d/'FINAL_VALID_METRICS.json','w'),indent=2);summary.append({'model':name,'train_oof_auc':met(y,o)['ROC_AUC'],'train_oof_auprc':met(y,o)['AUPRC'],'final_valid_auc':fm['ROC_AUC'],'final_valid_auprc':fm['AUPRC']})
 pd.DataFrame(summary).to_csv(out/'reports/CORE781_COMPARISON.csv',index=False)
if __name__=='__main__':main()
