#!/usr/bin/env python3
"""Assemble a final report only after both strict backbone pipelines finish."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from common import atomic_json,atomic_text,load_config
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=load_config(a.config);report=Path(cfg['report_root']);protocol=json.loads((report/'FROZEN_TWO_BACKBONE_PROTOCOL.json').read_text());result={'status':'success','protocol':protocol,'backbones':{}}
 for family in ['segresnet','deeplabv3plus_resnet50_imagenet']:
  audit=json.loads((report/'expanded_strict_audit'/family/'SUCCESS.json').read_text());fusion=json.loads((report/'expanded_strict_fusion'/family/'SUCCESS.json').read_text());verify=json.loads((Path(cfg['output_root'])/'expanded_strict'/'featurebanks'/family/'verification.json').read_text());result['backbones'][family]={'segmentation_audit':audit,'featurebank_verification':verify,'fusion':fusion}
 atomic_json(result,report/'FINAL_EXPANDED_STRICT_REPORT.json');atomic_text('# Expanded strict final report\n\nBoth confirmatory backbones completed under the frozen protocol.\n',report/'FINAL_EXPANDED_STRICT_REPORT.md')
if __name__=='__main__':main()
