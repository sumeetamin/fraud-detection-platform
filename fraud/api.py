import json
import os
import secrets
import sqlite3
import time
import threading
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timezone
from functools import lru_cache
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI,HTTPException,Query,Header,Request,Depends
from fastapi.responses import HTMLResponse
from pydantic import BaseModel,Field,ConfigDict
from .core import ROOT,FEATURES,drift

try:
    import psycopg
except ImportError:  # SQLite-only local installs can omit the Postgres extra.
    psycopg=None

app=FastAPI(title='Fraud Detection Platform',version='1.0.0')
_db_init_lock=threading.Lock()
_initialized_databases=set()
INTEGRITY_ERRORS=tuple(error for error in (sqlite3.IntegrityError, psycopg.IntegrityError if psycopg else None) if error)

@app.get('/',response_class=HTMLResponse,include_in_schema=False)
def home():
    return HTMLResponse('''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fraud Detection Platform API</title>
<style>
body{margin:0;background:#f4f7fb;color:#182230;font:16px/1.6 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:760px;margin:8vh auto;padding:36px;background:#fff;border:1px solid #dce4ee;border-radius:18px;box-shadow:0 18px 48px #18223012}
h1{font-size:2rem;margin:0 0 8px}p{color:#526174}.tag{display:inline-block;background:#e8f3ff;color:#135ca8;border-radius:999px;padding:4px 12px;font-size:.85rem;font-weight:650}
.links{display:flex;gap:12px;flex-wrap:wrap;margin:24px 0}.links a{padding:10px 16px;background:#125fc4;color:white;text-decoration:none;border-radius:9px;font-weight:650}.links a.secondary{background:#edf2f7;color:#223247}
section{border-top:1px solid #e6ebf1;padding-top:18px;margin-top:20px}code{background:#eef2f6;padding:2px 6px;border-radius:5px}
.note{background:#fff7e6;border-left:4px solid #e2a62b;padding:12px 16px;color:#5e4a20;border-radius:6px}
</style></head><body><main>
<span class="tag">API service</span><h1>Fraud Detection Platform</h1>
<p>A reproducible scoring and analyst-review demo with model versioning, delayed-label tracking, and drift monitoring.</p>
<div class="links"><a href="/docs">Interactive API docs</a><a class="secondary" href="/health">Readiness status</a></div>
<section><h2>Authentication</h2><p>Scoring and analyst endpoints require a private <code>X-API-Key</code> header. Keep the key out of browser links, source code, and shared screenshots. Open the API docs to inspect request fields.</p></section>
<p class="note">This is a portfolio prototype. Scores support human review and are not proof of fraud or a stand-alone basis for customer action.</p>
</main></body></html>''')

class Transaction(BaseModel):
    model_config=ConfigDict(extra='forbid')
    transaction_id: str=Field(min_length=1,max_length=80,pattern=r'^[A-Za-z0-9_-]+$')
    features: dict[str,float]

class Label(BaseModel):
    transaction_id: str
    is_fraud: int=Field(ge=0,le=1)

def require_api_access(request:Request,x_api_key:str|None=Header(default=None)):
    expected=os.environ.get('FRAUD_API_TOKEN')
    production=os.environ.get('FRAUD_ENV','local').lower()=='production'
    if expected:
        if len(expected)<32:
            raise HTTPException(503,'FRAUD_API_TOKEN must contain at least 32 characters.')
        if not secrets.compare_digest(x_api_key or '',expected):
            raise HTTPException(401,'A valid X-API-Key is required.')
        return
    if production:
        raise HTTPException(503,'FRAUD_API_TOKEN must be configured in production.')
    client=request.client
    if client is None or client.host not in {'127.0.0.1','::1','testclient'}:
        raise HTTPException(503,'Configure FRAUD_API_TOKEN before serving non-loopback requests.')

class PostgresConnection:
    """Small SQL adapter so the API can keep one placeholder style for both stores."""
    def __init__(self, connection):
        self.connection=connection

    def execute(self, statement, parameters=()):
        return self.connection.execute(statement.replace('?', '%s'), parameters)

    def commit(self): self.connection.commit()
    def rollback(self): self.connection.rollback()
    def close(self): self.connection.close()


def connect():
    database_url=os.environ.get('DATABASE_URL')
    production=os.environ.get('FRAUD_ENV','local').lower()=='production'
    if database_url:
        if psycopg is None:
            raise HTTPException(503,'Install psycopg to use DATABASE_URL.')
        key='postgres:'+database_url
        if key not in _initialized_databases:
            with _db_init_lock:
                if key not in _initialized_databases:
                    setup=psycopg.connect(database_url)
                    try:
                        setup.execute('CREATE TABLE IF NOT EXISTS events (transaction_id TEXT PRIMARY KEY, received_at TEXT, payload TEXT, score DOUBLE PRECISION, model_version TEXT, latency_ms DOUBLE PRECISION, label INTEGER, label_received_at TEXT)')
                        setup.commit()
                    finally:
                        setup.close()
                    _initialized_databases.add(key)

        @contextmanager
        def managed_postgres_connection():
            conn=PostgresConnection(psycopg.connect(database_url))
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        return managed_postgres_connection()

    if production:
        raise HTTPException(503,'DATABASE_URL must point to persistent Postgres in production.')
    path=Path(os.environ.get('FRAUD_DB',str(ROOT/'runtime/events.sqlite')))
    path.parent.mkdir(parents=True,exist_ok=True)
    key=str(path.resolve())
    if key not in _initialized_databases:
        with _db_init_lock:
            if key not in _initialized_databases:
                with sqlite3.connect(path,timeout=15) as setup:
                    setup.execute('PRAGMA journal_mode=WAL')
                    setup.execute('CREATE TABLE IF NOT EXISTS events (transaction_id TEXT PRIMARY KEY, received_at TEXT, payload TEXT, score REAL, model_version TEXT, latency_ms REAL, label INTEGER, label_received_at TEXT)')
                _initialized_databases.add(key)

    @contextmanager
    def managed_connection():
        conn=sqlite3.connect(path,timeout=15)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    return managed_connection()

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

@app.post('/predict',dependencies=[Depends(require_api_access)])
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
    except INTEGRITY_ERRORS:
        # Concurrent duplicate: re-enter once now that committed row is visible.
        return predict(transaction)
    return dict(transaction_id=transaction.transaction_id,score=score,model_version=bundle['model_version'],idempotent=False)

@app.post('/labels',dependencies=[Depends(require_api_access)])
def label(value:Label):
    with connect() as conn:
        row=conn.execute('SELECT label FROM events WHERE transaction_id=?',(value.transaction_id,)).fetchone()
        if row is None: raise HTTPException(404,'Transaction not found.')
        if row[0] is not None and row[0]!=value.is_fraud: raise HTTPException(409,'Conflicting label already recorded.')
        if row[0] is None:
            conn.execute('UPDATE events SET label=?,label_received_at=? WHERE transaction_id=?',
                (value.is_fraud,datetime.now(timezone.utc).isoformat(),value.transaction_id))
    return {'status':'recorded'}

@app.get('/queue',dependencies=[Depends(require_api_access)])
def queue(budget:int=Query(default=10,ge=1,le=1000)):
    # Capacity applies to the current UTC received day. Historical simulation lives in core.review_queue.
    today=datetime.now(timezone.utc).date().isoformat()
    with connect() as conn:
        rows=conn.execute('SELECT transaction_id,score,model_version FROM events WHERE label IS NULL AND substr(received_at,1,10)=? ORDER BY score DESC,transaction_id LIMIT ?', (today,budget)).fetchall()
    return [dict(transaction_id=r[0],score=r[1],model_version=r[2]) for r in rows]

@app.get('/monitor',dependencies=[Depends(require_api_access)])
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

