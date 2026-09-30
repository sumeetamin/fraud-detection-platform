import json
import os
import sqlite3
import time
from pathlib import Path
from datetime import datetime, timezone
from functools import lru_cache
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI,HTTPException,Query
from pydantic import BaseModel,Field,ConfigDict
from .core import ROOT,FEATURES,drift

app=FastAPI(title='Fraud Detection Platform',version='1.0.0')

class Transaction(BaseModel):
    model_config=ConfigDict(extra='forbid')
    transaction_id: str=Field(min_length=1,max_length=80,pattern=r'^[A-Za-z0-9_-]+$')
    features: dict[str,float]

class Label(BaseModel):
    transaction_id: str
    is_fraud: int=Field(ge=0,le=1)

def connect():
    path=Path(os.environ.get('FRAUD_DB',str(ROOT/'runtime/events.sqlite')))
    path.parent.mkdir(parents=True,exist_ok=True)
    conn=sqlite3.connect(path,timeout=15)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('CREATE TABLE IF NOT EXISTS events (transaction_id TEXT PRIMARY KEY, received_at TEXT, payload TEXT, score REAL, model_version TEXT, latency_ms REAL, label INTEGER, label_received_at TEXT)')
    return conn

@lru_cache(maxsize=4)
def load_version(version):
    return joblib.load(ROOT/f'artifacts/{version}.joblib')

def active():
    pointer=ROOT/'artifacts/current.txt'
    if not pointer.exists(): raise HTTPException(503,'Run python -m fraud.train first.')
    version=pointer.read_text().strip()
    if not (ROOT/f'artifacts/{version}.joblib').exists(): raise HTTPException(503,'Active model artifact missing; retrain.')
    return load_version(version)

@app.get('/health')
def health():
    bundle=active()
    return {'status':'ready','model_version':bundle['model_version']}

@app.post('/predict')
def predict(transaction:Transaction):
    start=time.perf_counter()
    payload=transaction.model_dump()
    bundle=active()
    if set(payload['features'])!=set(bundle['features']):
        raise HTTPException(422,'features must contain exactly: '+', '.join(bundle['features']))
    if not np.isfinite(list(payload['features'].values())).all() or payload['features']['amount']<0:
        raise HTTPException(422,'features must be finite and amount must be non-negative.')
    encoded=json.dumps(payload,sort_keys=True)
    with connect() as conn:
        old=conn.execute('SELECT payload,score,model_version FROM events WHERE transaction_id=?',(transaction.transaction_id,)).fetchone()
    if old:
        if old[0]!=encoded: raise HTTPException(409,'Transaction ID already exists with different features.')
        return dict(transaction_id=transaction.transaction_id,score=old[1],model_version=old[2],idempotent=True)
    score=float(bundle['model'].predict_proba(pd.DataFrame([payload['features']])[FEATURES])[0,1])
    latency=(time.perf_counter()-start)*1000
    try:
        with connect() as conn:
            conn.execute('INSERT INTO events VALUES (?,?,?,?,?,?,NULL,NULL)',
                (transaction.transaction_id,datetime.now(timezone.utc).isoformat(),encoded,score,bundle['model_version'],latency))
    except sqlite3.IntegrityError:
        # Concurrent duplicate: re-enter once now that committed row is visible.
        return predict(transaction)
    return dict(transaction_id=transaction.transaction_id,score=score,model_version=bundle['model_version'],idempotent=False)

@app.post('/labels')
def label(value:Label):
    with connect() as conn:
        row=conn.execute('SELECT label FROM events WHERE transaction_id=?',(value.transaction_id,)).fetchone()
        if row is None: raise HTTPException(404,'Transaction not found.')
        if row[0] is not None and row[0]!=value.is_fraud: raise HTTPException(409,'Conflicting label already recorded.')
        if row[0] is None:
            conn.execute('UPDATE events SET label=?,label_received_at=? WHERE transaction_id=?',
                (value.is_fraud,datetime.now(timezone.utc).isoformat(),value.transaction_id))
    return {'status':'recorded'}

@app.get('/queue')
def queue(budget:int=Query(default=10,ge=1,le=1000)):
    # Capacity applies to the current UTC received day. Historical simulation lives in core.review_queue.
    today=datetime.now(timezone.utc).date().isoformat()
    with connect() as conn:
        rows=conn.execute('SELECT transaction_id,score,model_version FROM events WHERE label IS NULL AND substr(received_at,1,10)=? ORDER BY score DESC,transaction_id LIMIT ?', (today,budget)).fetchall()
    return [dict(transaction_id=r[0],score=r[1],model_version=r[2]) for r in rows]

@app.get('/monitor')
def monitor():
    with connect() as conn:
        rows=conn.execute('SELECT payload,score,latency_ms,label,model_version FROM events ORDER BY received_at DESC LIMIT 1000').fetchall()
    if not rows: return {'events':0,'drift':[],'labelled_events':0}
    current=pd.DataFrame([json.loads(r[0])['features'] for r in rows])
    reference=pd.read_csv(ROOT/'data/reference.csv')
    labelled=[r for r in rows if r[3] is not None]
    report=dict(events=len(rows),labelled_events=len(labelled),latency_p95_ms=float(np.percentile([r[2] for r in rows],95)),
                drift=drift(reference,current) if len(rows)>=100 else [],
                note='Drift requires 100 observations. Labelled metrics may be biased by analyst selection. No automatic retraining.')
    if labelled:
        report['labelled_brier']=float(np.mean([(r[1]-r[3])**2 for r in labelled]))
    return report
