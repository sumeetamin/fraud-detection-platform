"""Explicit local model pointer switch; model promotion is never automatic."""
import argparse
import re
from .core import ROOT

def activate(version):
    if not re.fullmatch('[0-9a-f]{12}',version): raise ValueError('Invalid model version.')
    if not (ROOT/f'artifacts/{version}.joblib').exists(): raise ValueError('Model artifact does not exist.')
    temp=ROOT/'artifacts/current.tmp'; temp.write_text(version); temp.replace(ROOT/'artifacts/current.txt')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('version'); args=p.parse_args(); activate(args.version)
    print('Activated',args.version)
