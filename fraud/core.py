from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

ROOT=Path(__file__).resolve().parents[1]
FEATURES=['amount','hour','account_age_days','distance_km','transactions_1h','international','device_new']

def demo_data(seed=42, n=16000):
    rng=np.random.default_rng(seed)
    timestamps=pd.date_range('2025-01-01',periods=n,freq='10min')
    df=pd.DataFrame(dict(transaction_id=[f'TX-{i:06d}' for i in range(n)], timestamp=timestamps,
        amount=np.round(rng.lognormal(4,1.1,n),2), hour=timestamps.hour,
        account_age_days=rng.integers(0,1800,n), distance_km=np.round(rng.exponential(40,n),2),
        transactions_1h=rng.poisson(2,n),international=rng.binomial(1,.12,n),device_new=rng.binomial(1,.15,n)))
    logits=(-6.2+1.2*df.international+1.7*df.device_new+.35*df.transactions_1h
            +.003*df.distance_km+.8*(df.amount>180)+1.1*(df.account_age_days<40)+.6*(df.hour<5))
    df['is_fraud']=rng.binomial(1,1/(1+np.exp(-logits)))
    df['label_available_at']=df.timestamp+pd.Timedelta(days=3)
    return df

def validate(df):
    required=set(FEATURES+['transaction_id','timestamp','is_fraud','label_available_at'])
    if not required<=set(df): raise ValueError('Missing columns: '+', '.join(sorted(required-set(df))))
    df=df.copy()
    for col in ['timestamp','label_available_at']: df[col]=pd.to_datetime(df[col],errors='raise')
    if df[list(required)].isna().any().any(): raise ValueError('Missing values are not allowed.')
    if df.transaction_id.duplicated().any(): raise ValueError('Duplicate transaction IDs.')
    for col in FEATURES+['is_fraud']: df[col]=pd.to_numeric(df[col],errors='raise')
    if not np.isfinite(df[FEATURES]).all().all(): raise ValueError('Features must be finite.')
    if (df[FEATURES]<0).any().any(): raise ValueError('Negative feature values are not allowed.')
    for col in ['international','device_new','is_fraud']:
        if not df[col].isin([0,1]).all(): raise ValueError(f'{col} must be 0 or 1.')
    if not df.hour.between(0,23).all(): raise ValueError('Hour must be 0–23.')
    for col in ['hour','account_age_days','transactions_1h']:
        if (df[col]%1!=0).any(): raise ValueError(f'{col} must be an integer.')
    if (df.label_available_at<df.timestamp).any(): raise ValueError('Labels cannot arrive before transactions.')
    if len(df)<1000: raise ValueError('At least 1,000 rows required.')
    return df.sort_values('timestamp').reset_index(drop=True)

def temporal_split(df):
    # Model trained as of t1; validation labels known by t2; test is untouched.
    t1=df.timestamp.iloc[int(len(df)*.60)]
    t2=df.timestamp.iloc[int(len(df)*.80)]
    train=df[(df.timestamp<t1)&(df.label_available_at<t1)]
    val=df[(df.timestamp>=t1)&(df.timestamp<t2)&(df.label_available_at<t2)]
    test=df[df.timestamp>=t2]
    for name,part in [('train',train),('validation',val),('test',test)]:
        if part.is_fraud.nunique()!=2: raise ValueError(f'{name} needs positive and negative examples.')
    return train,val,test,t1,t2

def review_queue(frame,scores,daily_budget=10):
    if not 1<=daily_budget<=1000: raise ValueError('Daily review budget must be 1–1000.')
    ranked=frame.copy(); ranked['score']=scores
    ranked['day']=pd.to_datetime(ranked.timestamp).dt.strftime('%Y-%m-%d')
    return ranked.sort_values(['day','score','transaction_id'],ascending=[True,False,True]).groupby('day').head(daily_budget)

def metrics(df,scores,budget=10):
    q=review_queue(df,scores,budget)
    return dict(average_precision=float(average_precision_score(df.is_fraud,scores)),
                roc_auc=float(roc_auc_score(df.is_fraud,scores)),brier=float(brier_score_loss(df.is_fraud,scores)),
                precision_at_daily_budget=float(q.is_fraud.mean()),
                recall_at_daily_budget=float(q.is_fraud.sum()/max(1,df.is_fraud.sum())),
                reviewed=len(q), fraud_prevalence=float(df.is_fraud.mean()))

def drift(reference,current):
    report=[]
    for col in FEATURES:
        a=np.asarray(reference[col],dtype=float); b=np.asarray(current[col],dtype=float)
        boundaries=np.unique(np.quantile(a,np.linspace(0,1,11)))
        boundaries=np.r_[-np.inf,boundaries[1:-1],np.inf]
        if len(boundaries)<3: boundaries=np.array([-np.inf,.5,np.inf])
        p=np.histogram(a,bins=boundaries)[0]/len(a)
        q=np.histogram(b,bins=boundaries)[0]/len(b)
        p=np.maximum(p,1e-6); q=np.maximum(q,1e-6)
        psi=float(np.sum((q-p)*np.log(q/p)))
        report.append(dict(feature=col,psi=psi,alert=bool(psi>.2),reference_mean=float(a.mean()),current_mean=float(b.mean())))
    return report
