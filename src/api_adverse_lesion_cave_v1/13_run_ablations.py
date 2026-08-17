#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from common import atomic_csv, atomic_json, configure_runtime, load_config, run_checked, stage_logger, write_marker


def metric(y,p,threshold:float=0.5):
    prediction=(p>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,prediction,labels=[0,1]).ravel()
    return {"auroc":float(roc_auc_score(y,p)),"auprc":float(average_precision_score(y,p)),"brier":float(brier_score_loss(y,p)),"balanced_accuracy":float(balanced_accuracy_score(y,prediction)),"sensitivity":float(tp/max(tp+fn,1)),"specificity":float(tn/max(tn+fp,1))}


def youden_threshold(y,p):
    fpr,tpr,thresholds=roc_curve(y,p); finite=np.isfinite(thresholds)
    return float(thresholds[finite][int(np.argmax(tpr[finite]-fpr[finite]))]) if finite.any() else 0.5


def fit_meta(x,y):
    scaler=StandardScaler().fit(x)
    model=LogisticRegression(C=1,class_weight="balanced",solver="newton-cg",max_iter=1000,tol=1e-4)
    with warnings.catch_warnings():
        warnings.simplefilter("error",ConvergenceWarning)
        model.fit(scaler.transform(x).astype(np.float64),y)
    return scaler,model


def selected_predictions(model_root:Path):
    run=json.loads((model_root/"adverse_patient"/"run_config.json").read_text(encoding="utf-8")); name=run["best_model_selected_by_train_oof_auprc"]; column=name.casefold()+"_probability"
    train=pd.read_csv(model_root/"adverse_patient"/"train_oof_predictions.csv",dtype={"patient_id":str}); valid=pd.read_csv(model_root/"adverse_patient"/"valid_predictions.csv",dtype={"patient_id":str})
    return name,column,train,valid


def late_fusion(whole_root:Path,roi_root:Path,output:Path):
    whole_name,whole_col,whole_train,whole_valid=selected_predictions(whole_root); roi_name,roi_col,roi_train,roi_valid=selected_predictions(roi_root)
    train=whole_train[["patient_id","target",whole_col]].merge(roi_train[["patient_id","target",roi_col]],on=["patient_id","target"],validate="one_to_one",suffixes=("_whole","_roi")); valid=whole_valid[["patient_id","target",whole_col]].merge(roi_valid[["patient_id","target",roi_col]],on=["patient_id","target"],validate="one_to_one",suffixes=("_whole","_roi"))
    x=train[[f"{whole_col}_whole",f"{roi_col}_roi"]].to_numpy(float); y=train.target.to_numpy(int); xv=valid[[f"{whole_col}_whole",f"{roi_col}_roi"]].to_numpy(float); oof=np.full(len(y),np.nan); folds=StratifiedKFold(5,shuffle=True,random_state=42); audits=[]
    for fold,(development,holdout) in enumerate(folds.split(x,y),1):
        scaler,model=fit_meta(x[development],y[development]); oof[holdout]=model.predict_proba(scaler.transform(x[holdout]))[:,1]; audits.append({"fold":fold,"n_iter":int(model.n_iter_[0]),"development":len(development),"holdout":len(holdout)})
    scaler,model=fit_meta(x,y); pv=model.predict_proba(scaler.transform(xv))[:,1]
    threshold=youden_threshold(y,oof)
    train["late_fusion_probability"]=oof; valid["late_fusion_probability"]=pv; output.mkdir(parents=True,exist_ok=True); atomic_csv(train,output/"train_meta_oof_predictions.csv"); atomic_csv(valid,output/"valid_predictions.csv"); metrics=pd.DataFrame([{"task":"adverse_patient","model":"LateFusion_crossfit","split":"Train_OOF","threshold":threshold,**metric(y,oof,threshold)},{"task":"adverse_patient","model":"LateFusion_crossfit","split":"Valid","threshold":threshold,**metric(valid.target.to_numpy(int),pv,threshold)}]); atomic_csv(metrics,output/"metrics.csv")
    atomic_json({"whole_model":whole_name,"roi_model":roi_name,"meta_cross_fitted":True,"fold_audits":audits,"full_train_n_iter":int(model.n_iter_[0]),"threshold_from_train_oof":threshold,"convergence_warning_is_hard_failure":True},output/"run_config.json"); return metrics


def paired_bootstrap(whole_root:Path,roi_root:Path,iterations:int=2000):
    whole_name,whole_col,_,whole=selected_predictions(whole_root); roi_name,roi_col,_,roi=selected_predictions(roi_root); data=whole[["patient_id","target",whole_col]].merge(roi[["patient_id","target",roi_col]],on=["patient_id","target"],validate="one_to_one",suffixes=("_whole","_roi")); y=data.target.to_numpy(int); pw=data[f"{whole_col}_whole"].to_numpy(float); pr=data[f"{roi_col}_roi"].to_numpy(float); rng=np.random.default_rng(42); rows=[]
    for index in range(iterations):
        sample=rng.integers(0,len(y),len(y)); ys=y[sample]
        if len(np.unique(ys))<2: continue
        rows.append({"bootstrap":index,"auroc_diff_roi_minus_whole":roc_auc_score(ys,pr[sample])-roc_auc_score(ys,pw[sample]),"auprc_diff_roi_minus_whole":average_precision_score(ys,pr[sample])-average_precision_score(ys,pw[sample])})
    frame=pd.DataFrame(rows); summary=[]
    for column in [c for c in frame.columns if c!="bootstrap"]:
        values=frame[column].to_numpy(); summary.append({"metric":column,"mean":float(values.mean()),"ci_low":float(np.quantile(values,0.025)),"ci_high":float(np.quantile(values,0.975)),"iterations":len(values),"whole_model":whole_name,"roi_model":roi_name})
    return frame,pd.DataFrame(summary)
def selected_bootstrap(model_root:Path,variant:str,iterations:int=2000)->pd.DataFrame:
    name,column,train,valid=selected_predictions(model_root)
    run=json.loads((model_root/"adverse_patient"/"run_config.json").read_text(encoding="utf-8"))
    threshold=float(run["thresholds_from_train_oof"][name])
    rng=np.random.default_rng(42); rows=[]
    for split,data in (("Train_OOF",train),("Valid",valid)):
        y=data.target.to_numpy(int); probability=data[column].to_numpy(float)
        point=metric(y,probability,threshold); draws={key:[] for key in point}
        for _ in range(iterations):
            sample=rng.integers(0,len(y),len(y)); ys=y[sample]
            if len(np.unique(ys))<2: continue
            values=metric(ys,probability[sample],threshold)
            for key,value in values.items(): draws[key].append(value)
        for key,value in point.items():
            samples=np.asarray(draws[key],dtype=float)
            rows.append({"ablation":variant,"model":name,"split":split,"metric":key,"estimate":value,"ci_low":float(np.quantile(samples,0.025)),"ci_high":float(np.quantile(samples,0.975)),"bootstrap_iterations":len(samples),"threshold_from_train_oof":threshold})
    return pd.DataFrame(rows)




def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--config"); args=parser.parse_args(); config=load_config(args.config); configure_runtime(config); finish=stage_logger("13_run_ablations")
    code=Path(config["paths"]["code"]); outputs=Path(config["paths"]["outputs"]); reports=Path(config["paths"]["reports"])
    variants=["pred_roi","whole_same_cohort","gt_roi","all_nonzero_roi","pred_roi_pre","pred_roi_post","pred_roi_max","pred_roi_top2_mean","whole_same_cohort_max","whole_same_cohort_top2_mean","pred_morphology","gt_morphology"]
    for variant in variants:
        marker=outputs/"adverse_models"/variant/".MODELS_SUCCESS"
        if marker.is_file(): continue
        run_checked([config["prediction_python"],str(code/"12_train_adverse_models_fixed.py"),"--config",config["_config_path"],"--variant",variant],cwd=Path(config["project_root"]))
    model_root=outputs/"adverse_models"; all_metrics=[]
    for variant in variants:
        path=model_root/variant/"all_task_metrics.csv"; frame=pd.read_csv(path); frame.insert(0,"ablation",variant); all_metrics.append(frame)
    late=late_fusion(model_root/"whole_same_cohort",model_root/"pred_roi",model_root/"late_fusion"); late.insert(0,"ablation","whole_plus_pred_roi_late_fusion"); all_metrics.append(late)
    metrics=pd.concat(all_metrics,ignore_index=True); atomic_csv(metrics,reports/"all_ablation_metrics.csv")
    main_run=json.loads((model_root/"pred_roi"/"adverse_patient"/"run_config.json").read_text(encoding="utf-8")); selected=main_run["best_model_selected_by_train_oof_auprc"]; selected_metrics=metrics[(metrics.ablation=="pred_roi")&(metrics.model==selected)].copy(); atomic_csv(selected_metrics,reports/"selected_model_metrics.csv")
    boot,boot_summary=paired_bootstrap(model_root/"whole_same_cohort",model_root/"pred_roi"); atomic_csv(boot,reports/"whole_vs_roi_paired_bootstrap_samples.csv"); atomic_csv(boot_summary,reports/"whole_vs_roi_paired_bootstrap.csv")
    ci=pd.concat([selected_bootstrap(model_root/variant,variant) for variant in variants],ignore_index=True); atomic_csv(ci,reports/"all_ablation_selected_bootstrap_ci.csv")
    summary={"variants":variants,"selected_pred_roi_model":selected,"late_fusion_meta_cross_fitted":True,"paired_bootstrap_iterations":len(boot),"per_ablation_bootstrap_iterations":2000}; atomic_json(summary,reports/"ablation_summary.json"); write_marker(reports/".ABLATIONS_SUCCESS","13_run_ablations",config,{},summary)
    finish(summary); print(json.dumps(summary,ensure_ascii=False,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
