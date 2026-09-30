import json
import joblib
import pandas as pd
import streamlit as st
from fraud.core import ROOT,FEATURES,review_queue,drift

st.set_page_config(page_title='Fraud Operations',page_icon='🛡️',layout='wide')
st.caption('RISK LAB / MACHINE LEARNING ENGINEERING')
st.title('Fraud Operations')
st.write('Score transactions. Prioritize reviews. Detect changing conditions.')
required=[ROOT/'reports/evaluation.json',ROOT/'artifacts/current.txt',ROOT/'data/test_transactions.csv',ROOT/'data/reference.csv']
if not all(path.exists() for path in required): st.info('Run `python -m fraud.train` first. Raw benchmark data and model artifacts are intentionally not committed.'); st.stop()
report=json.loads((ROOT/'reports/evaluation.json').read_text())
st.info('Research demonstration only · '+report['source']+' · scores are not production fraud decisions')
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
        amount=st.number_input('Amount',min_value=0.,value=240.)
        row=reference[FEATURES].median().to_frame().T
        row['amount']=amount
        st.caption('The benchmark exposes anonymized PCA components V1–V28. Adjust them only if you have compatible upstream features.')
        edited=st.data_editor(row,hide_index=True,num_rows='fixed',width='stretch')
        submitted=st.form_submit_button('Calculate score')
    if submitted:
        score=bundle['model'].predict_proba(edited[FEATURES])[0,1]
        st.metric('Model risk score',f'{score:.1%}')
        st.caption('Local interactive inference; does not add an event to the API review queue.')
with tabs[2]:
    st.write('Model selected using validation average precision: **'+report['selected_model']+'**')
    st.dataframe(pd.DataFrame(report['test']).T,width='stretch')
    st.write('A three-hour label delay is simulated for the two-day benchmark. Training and validation exclude labels that were unavailable at their respective cutoff dates.')
with tabs[3]:
    shift=st.toggle('Simulate changed transaction distribution')
    current=df.copy()
    if shift: current['amount']*=4; current['v1']+=3
    result=pd.DataFrame(drift(reference,current))
    st.bar_chart(result.set_index('feature').psi)
    st.dataframe(result,hide_index=True)
    st.caption('PSI > 0.2 is a demonstration alert threshold, not a statistical guarantee. Investigate drift before deciding whether retraining is appropriate.')
