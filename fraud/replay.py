import argparse
import json
import time
import numpy as np
import pandas as pd
import requests
from .core import ROOT,FEATURES

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--url',default='http://127.0.0.1:8003'); parser.add_argument('--limit',type=int,default=200)
    parser.add_argument('--incident',action='store_true'); args=parser.parse_args()
    df=pd.read_csv(ROOT/'data/test_transactions.csv').head(args.limit)
    if args.incident: df['amount']*=4; df['distance_km']+=250
    timings=[]
    for _,row in df.iterrows():
        data={k:row[k] for k in FEATURES}; data['transaction_id']=('SHIFT-' if args.incident else 'REPLAY-')+row.transaction_id
        start=time.perf_counter(); response=requests.post(args.url+'/predict',json=data,timeout=30); response.raise_for_status()
        timings.append((time.perf_counter()-start)*1000)
    print(json.dumps(dict(requests=len(timings),p50_ms=float(np.percentile(timings,50)),p95_ms=float(np.percentile(timings,95)),
                         note='Sequential local HTTP replay including network overhead, not a concurrency benchmark.'),indent=2))

if __name__=='__main__': main()
