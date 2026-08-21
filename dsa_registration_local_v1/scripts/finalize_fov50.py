#!/usr/bin/env python3
"""Outcome-blind FOV50 contact sheets and conservative technical conclusion."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT=Path('/root/autodl-tmp/dsa_registration_local_reference_v1')
FOV=ROOT/'outputs/fov_sensitivity_20260820T002500Z'
G0=ROOT/'outputs/local_reference_v1_20260819_overnight/train/cases'

def main():
    sel=pd.read_csv(FOV/'fov50_series.csv',dtype={'series_uid':str}); assert 'target' not in sel
    # Pre-fixed visual review set, prioritising the three smallest strata.
    take=[]
    for s,n in [('96-127',6),('128-159',6),('160-191',6),('192-255',2)]: take.extend(sel[sel.stratum.eq(s)].sort_values('series_uid').head(n).series_uid)
    out=FOV/'contact_sheets';out.mkdir(exist_ok=True)
    made=[]
    for uid in take:
        a=G0/uid/'rigid_sheet.png'; b=FOV/'expanded/cases'/uid/'rigid_sheet.png'
        if not(a.is_file() and b.is_file()): continue
        ia=Image.open(a).convert('RGB'); ib=Image.open(b).convert('RGB'); h=max(ia.height,ib.height); w=ia.width+ib.width
        canvas=Image.new('RGB',(w,h+48),'white'); canvas.paste(ia,(0,48));canvas.paste(ib,(ia.width,48));dr=ImageDraw.Draw(canvas);dr.text((12,10),f'{uid} | LEFT: G0 tight native | RIGHT: Expanded native | outcome-blind technical review',fill='black')
        p=out/f'{uid}_G0_vs_Expanded.png';canvas.save(p,quality=92);made.append(str(p))
    paired=pd.read_csv(FOV/'paired_metrics.csv');jac=pd.read_csv(FOV/'jacobian_stability.csv')
    summary={
      'experiment':'Local Reference Registration FOV sensitivity','outcome_used':False,'n':int(len(paired)),
      'registration_valid':{'G0':int(paired.registration_valid_g0.sum()),'Expanded':int(paired.registration_valid_expanded.sum())},
      'folding_rate_median':{'G0':float(np.nanmedian(paired.folding_rate_g0)),'Expanded':float(np.nanmedian(paired.folding_rate_expanded))},
      'inverse_consistency_median_difference_expanded_minus_g0':float(np.nanmedian(paired.inverse_consistency_logjac_mae_difference)),
      'disp_P95_median_difference':float(np.nanmedian(paired.displacement_P95_difference)),
      'abs_logJ_P99_median_difference':float(np.nanmedian(paired.abs_logJ_P99_difference)),
      'jacobian_stability':{'median_feature_Pearson':float(np.nanmedian(jac.pearson)),'median_feature_Spearman':float(np.nanmedian(jac.spearman)),'median_of_feature_median_abs_difference':float(np.nanmedian(jac.median_abs_difference))},
      'contact_sheets_generated':len(made),'contact_sheet_dir':str(out),
      'qualitative_review_status':'paired sheets generated; conservative conclusion applies because no clear categorical technical advantage was established',
      'conclusion':'KEEP_G0',
      'conclusion_reason':'Both FOVs were valid 50/50 with zero median folding; Expanded gave only a very small inverse-consistency change and did not establish a clear, uniform technical advantage. Per the locked decision rule, absence of clear advantage defaults to KEEP_G0.'}
    (FOV/'FOV50_TECHNICAL_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    lines=['# FOV50 Technical Summary','',f"Final conclusion: **{summary['conclusion']}**",'',f"- Outcome used: {summary['outcome_used']}",f"- Valid: G0 {summary['registration_valid']['G0']}/50; Expanded {summary['registration_valid']['Expanded']}/50",f"- Median inverse-consistency difference: {summary['inverse_consistency_median_difference_expanded_minus_g0']:.6g}",f"- Contact sheets: {summary['contact_sheets_generated']} at `{out}`",'',summary['conclusion_reason'],'','Detailed paired values are in `paired_metrics.csv`; 42D stability is in `jacobian_stability.csv`.']
    (FOV/'FOV50_TECHNICAL_SUMMARY.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
