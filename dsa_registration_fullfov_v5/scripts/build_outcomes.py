#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import pandas as pd
from dsa_reg.outcomes import derive_abs_rel_outcomes

p=argparse.ArgumentParser()
p.add_argument('--input',required=True); p.add_argument('--out',required=True)
p.add_argument('--post-col',required=True); p.add_argument('--follow-col',required=True)
p.add_argument('--enlarged-col',default=None)
a=p.parse_args()
df=pd.read_csv(a.input)
out=derive_abs_rel_outcomes(df,a.post_col,a.follow_col,a.enlarged_col)
Path(a.out).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.out,index=False)
print('saved',a.out,'rows',len(out),'y_abs_valid',int(out.y_abs.notna().sum()),'y_rel_valid',int(out.y_rel.notna().sum()))

