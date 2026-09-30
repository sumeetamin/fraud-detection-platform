"""Adapters for public datasets. Raw files stay in ignored data/private/."""
from pathlib import Path
import pandas as pd
from scipy.io import arff

ULB_FEATURES=['amount']+[f'v{i}' for i in range(1,29)]

def openml_creditcard(path: str | Path, label_delay_hours: int=3) -> pd.DataFrame:
    """Load the ULB/Worldline credit-card benchmark downloaded from OpenML 1597.

    The benchmark's Time feature is elapsed seconds rather than a calendar timestamp.
    A fixed UTC origin is used only to construct chronological train/validation/test
    windows; it does not claim to be the original transaction date.
    """
    records,_=arff.loadarff(path)
    raw=pd.DataFrame(records)
    required={'Time','Amount','Class'}|{f'V{i}' for i in range(1,29)}
    missing=required-set(raw.columns)
    if missing: raise ValueError('Missing ULB benchmark columns: '+', '.join(sorted(missing)))
    timestamp=pd.Timestamp('2013-01-01',tz='UTC')+pd.to_timedelta(pd.to_numeric(raw['Time']),unit='s')
    out=pd.DataFrame({'transaction_id':[f'ULB-{i:06d}' for i in range(len(raw))],
        'timestamp':timestamp,'label_available_at':timestamp+pd.Timedelta(hours=label_delay_hours),
        'is_fraud':raw['Class'].map(lambda value: int(value.decode() if isinstance(value,bytes) else value)),
        'amount':pd.to_numeric(raw['Amount'])})
    for i in range(1,29): out[f'v{i}']=pd.to_numeric(raw[f'V{i}'])
    return out
