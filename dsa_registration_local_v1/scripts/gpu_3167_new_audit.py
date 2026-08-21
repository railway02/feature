#!/usr/bin/env python3
"""Independent fail-closed audit for the requested formal 3167-new branch."""
import json, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
run=ROOT/'outputs'/__import__('sys').argv[1]
legacy=Path('/root/autodl-tmp/aneurysm/configs/api_png2d_segresnet_cave_fusion_v5_series_mapped_reference_ready_strict.json')
d=json.loads(legacy.read_text()); exp=d.get('expected',{})
issues=[]
if exp.get('train_rows') != 800 or exp.get('valid_rows') != 211: issues.append('legacy_3167_input_contract_is_781_207_not_formal_800_211')
payload={'branch':'3167-new','checked_utc':time.strftime('%FT%TZ',time.gmtime()),'status':'BLOCKED' if issues else 'READY','integrity_class':'fundamental_cohort_contract' if issues else None,'issues':issues,'legacy_config':str(legacy),'legacy_expected':exp,'required_contract':{'train_rows':800,'valid_rows':211,'fold_source':'local_reference_train800_grouped_folds.csv'},'registration_branch_independent':True}
(run/'gpu_3167_new').mkdir(parents=True,exist_ok=True)
(run/'gpu_3167_new'/'STATUS.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps(payload),flush=True)
