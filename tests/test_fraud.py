import numpy as np
import pytest
from fastapi.testclient import TestClient
from fraud.core import demo_data,validate,temporal_split,review_queue,drift,FEATURES
from fraud.api import app
from fraud.registry import activate

def test_label_delay_and_no_overlap():
    train,val,test,t1,t2=temporal_split(validate(demo_data(n=4000)))
    assert train.label_available_at.max()<t1
    assert val.label_available_at.max()<t2
    assert set(train.transaction_id).isdisjoint(val.transaction_id)
    assert set(val.transaction_id).isdisjoint(test.transaction_id)

def test_review_budget_and_drift():
    df=demo_data(n=2000)
    queue=review_queue(df,np.linspace(0,1,len(df)),3)
    assert queue.groupby('day').size().max()<=3
    shifted=df.copy(); shifted['amount']*=10
    assert next(x for x in drift(df,shifted) if x['feature']=='amount')['alert']
    assert max(x['psi'] for x in drift(df,df))<1e-8

def test_api_idempotency_labels_and_monitor(tmp_path,monkeypatch):
    monkeypatch.setenv('FRAUD_DB',str(tmp_path/'events.sqlite'))
    c=TestClient(app)
    data=dict(transaction_id='TEST-1',amount=100,hour=12,account_age_days=100,distance_km=20,transactions_1h=2,international=0,device_new=0)
    first=c.post('/predict',json=data); assert first.status_code==200
    assert 0<=first.json()['score']<=1
    assert c.post('/predict',json=data).json()['idempotent']
    assert c.post('/predict',json=dict(data,amount=101)).status_code==409
    assert len(c.get('/queue').json())==1
    assert c.post('/labels',json={'transaction_id':'TEST-1','is_fraud':1}).status_code==200
    assert c.get('/queue').json()==[]
    assert c.get('/monitor').json()['labelled_events']==1
    assert c.post('/labels',json={'transaction_id':'TEST-1','is_fraud':0}).status_code==409
    assert c.post('/predict',json=dict(data,hour=25)).status_code==422

def test_bad_registry_and_data():
    with pytest.raises(ValueError): activate('../bad')
    df=demo_data(n=2000); df.loc[0,'amount']=-1
    with pytest.raises(ValueError): validate(df)
