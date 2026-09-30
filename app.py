import json
import joblib
import pandas as pd
import streamlit as st
from fraud.core import ROOT,FEATURES,review_queue,drift

st.set_page_config(page_title='Fraud Operations',page_icon='🛡️',layout='wide')
st.caption('RISK LAB / MACHINE LEARNING ENGINEERING')
st.title('Fraud Operations')
st.write('Score transactions. Prioritize reviews. Detect changing conditions.')
if not (ROOT/'reports/evaluation.json').exists(): st.info('Run `python -m fraud.train` first.'); st.stop()
report=json.loads((ROOT/'reports/evaluation.json').read_text())
st.info('Demonstration only · '+report['source']+' · scores are not production fraud decisions')
version=(ROOT/'artifacts/current.txt').read_text().strip()
bundle=joblib.load(ROOT/f'artifacts/{version}.joblib')
df=pd.read_csv(ROOT/'data/test_transactions.csv',parse_dates=['timestamp'])
reference=pd.read_csv(ROOT/'data/reference.csv')
budget=st.sidebar.slider('Reviews per day',1,30,10)
st.sidebar.caption('Active model: '+version)
tabs=st.tabs(['Analyst queue','Score a transaction','Model comparison','Incident simulation'])
with tabs[0]:
    queue=review_queue(df,df.score,budget)
    a,b,c=st.columns(3)
    a.metric('Precision at capacity',f'{queue.is_fraud.mean():.1%}')
    b.metric('Fraud captured',f'{queue.is_fraud.sum()/df.is_fraud.sum():.1%}')
    c.metric('Selected reviews',len(queue))
    st.caption('Historical test labels are shown for evaluation. Live API queues exclude labels and already reviewed events.')
    st.dataframe(queue[['transaction_id','timestamp','amount','score','is_fraud']],hide_index=True,width='stretch')
    st.download_button('Export queue',queue.to_csv(index=False),'review-queue.csv','text/csv')
with tabs[1]:
    with st.form('score'):
        a,b=st.columns(2)
        with a:
            amount=st.number_input('Amount',min_value=0.,value=240.)
            hour=st.slider('Hour',0,23,2)
            age=st.number_input('Account age in days',min_value=0,value=15)
            distance=st.number_input('Distance from usual location (km)',min_value=0.,value=250.)
        with b:
            count=st.number_input('Transactions in preceding hour',min_value=0,value=7)
            international=st.checkbox('International')
            new=st.checkbox('New device',value=True)
        submitted=st.form_submit_button('Calculate score')
    if submitted:
        row=pd.DataFrame([dict(amount=amount,hour=hour,account_age_days=age,distance_km=distance,transactions_1h=count,international=int(international),device_new=int(new))])
        score=bundle['model'].predict_proba(row[FEATURES])[0,1]
        st.metric('Model risk score',f'{score:.1%}')
        st.caption('Local interactive inference; does not add an event to the API review queue.')
with tabs[2]:
    st.write('Model selected using validation average precision: **'+report['selected_model']+'**')
    st.dataframe(pd.DataFrame(report['test']).T,width='stretch')
    st.write('Labels arrive after three days in the demo. Training and validation exclude labels that were unavailable at their respective cutoff dates.')
with tabs[3]:
    shift=st.toggle('Simulate changed transaction distribution')
    current=df.copy()
    if shift: current['amount']*=4; current['distance_km']+=250
    result=pd.DataFrame(drift(reference,current))
    st.bar_chart(result.set_index('feature').psi)
    st.dataframe(result,hide_index=True)
    st.caption('PSI > 0.2 is a demonstration alert threshold, not a statistical guarantee. Investigate drift before deciding whether retraining is appropriate.')
