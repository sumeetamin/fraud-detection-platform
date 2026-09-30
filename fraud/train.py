import argparse
import hashlib
import json
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .core import ROOT,FEATURES,demo_data,validate,temporal_split,metrics,review_queue,drift
from .sources import openml_creditcard

def main():
    parser=argparse.ArgumentParser()
    source=parser.add_mutually_exclusive_group()
    source.add_argument('--csv')
    source.add_argument('--openml-creditcard',help='Path to creditcard.arff from OpenML dataset 1597.')
    args=parser.parse_args()
    if args.openml_creditcard:
        df=validate(openml_creditcard(args.openml_creditcard))
        source_name='ULB/Worldline credit-card fraud benchmark via OpenML dataset 1597; V1–V28 and Amount; simulated three-hour label delay'
    elif args.csv:
        df=validate(pd.read_csv(args.csv)); source_name='user CSV'
    else:
        df=validate(demo_data()); source_name='synthetic transactions; seed 42; no real financial records'
    for folder in ['data','reports','artifacts']: (ROOT/folder).mkdir(exist_ok=True)
    train,val,test,t1,t2=temporal_split(df)
    models={'logistic':make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,random_state=42)),
            'boosted':HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=10,l2_regularization=5,random_state=42)}
    validation={}
    for name,model in models.items():
        model.fit(train[FEATURES],train.is_fraud)
        validation[name]=metrics(val,model.predict_proba(val[FEATURES])[:,1])
    selected=max(validation,key=lambda name: validation[name]['average_precision'])
    version=hashlib.sha256(pd.util.hash_pandas_object(train,index=False).values.tobytes()+selected.encode()).hexdigest()[:12]
    report={'source':source_name,
            'model_version':version,'selected_model':selected,'selection_metric':'validation average precision',
            'train_rows':len(train),'validation_rows':len(val),'test_rows':len(test),
            'train_as_of':str(t1),'evaluation_as_of':str(t2),
            'train_label_max':str(train.label_available_at.max()),'validation_label_max':str(val.label_available_at.max()),
            'validation':validation,'test':{}}
    for name,model in models.items():
        scores=model.predict_proba(test[FEATURES])[:,1]
        report['test'][name]=metrics(test,scores)
    bundle=dict(model=models[selected],model_version=version,features=FEATURES,report=report)
    joblib.dump(bundle,ROOT/f'artifacts/{version}.joblib')
    # Atomic pointer swap; prior artifacts remain available for explicit rollback.
    pointer=ROOT/'artifacts/current.tmp'; pointer.write_text(version); pointer.replace(ROOT/'artifacts/current.txt')
    scores=models[selected].predict_proba(test[FEATURES])[:,1]
    test=test.copy(); test['score']=scores
    test.to_csv(ROOT/'data/test_transactions.csv',index=False)
    train[FEATURES].to_csv(ROOT/'data/reference.csv',index=False)
    df.head(2000).to_csv(ROOT/'data/schema_example.csv',index=False)
    review_queue(test,scores).to_csv(ROOT/'reports/review_queue.csv',index=False)
    shifted=test.copy(); shifted['amount']*=4; shifted['v1']+=3
    report['drift_normal']=drift(train,test)
    report['drift_incident']=drift(train,shifted)
    (ROOT/'reports/evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if not k.startswith('drift')},indent=2))

if __name__=='__main__': main()
