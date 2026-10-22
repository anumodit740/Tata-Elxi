"""Fresh frozen-model inference and unchanged rule-agent integration. No fitting."""
from pathlib import Path
import sys,io,json,hashlib,zipfile,urllib.request
from functools import lru_cache
import numpy as np
import pandas as pd
import joblib
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'model'))
from core import CANONICAL_COLUMNS,DataValidationError,validate_trajectory
from inference_v3 import predict_bundle
import agent_v3

SUBSETS=('FD001','FD002','FD003','FD004')
def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def metadata():return json.loads((ROOT/'model/manifest.json').read_text(encoding='utf-8'))
def load_bundle(subset):
    if subset not in SUBSETS:raise ValueError('Choose a supported subset.')
    record=metadata()['models'][subset];path=ROOT/'model'/record['file']
    if sha256(path)!=record['sha256']:raise ValueError('Model integrity check failed. Inference stopped.')
    return joblib.load(path)
def validate_history(raw):
    validate_trajectory(raw)
    for _,g in raw.groupby('engine_id',sort=False):
        if int(g.cycle.iloc[0])!=1 or not np.array_equal(g.cycle.to_numpy(),np.arange(1,len(g)+1)):
            raise DataValidationError('Supply complete available history from cycle 1, with no missing cycles.')
def load_sample(subset):
    if subset not in SUBSETS:raise ValueError('Unsupported subset.')
    p=ROOT/'.runtime/examples'/f'{subset}_TRAIN_engine1.csv'
    info=json.loads((ROOT/'examples/provenance.json').read_text())['samples'][subset]
    if not p.exists():
        data=download_train_member(subset)
        frame=pd.read_csv(io.BytesIO(data),sep=r'\s+',header=None);frame.columns=CANONICAL_COLUMNS
        sample=frame[frame.engine_id==info['engine_id']].reset_index(drop=True)
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(sample.to_csv(index=False,float_format='%.17g',lineterminator='\n'),encoding='utf-8')
    if sha256(p)!=info['sample_sha256']:raise ValueError('Demo sample integrity check failed.')
    out=pd.read_csv(p,float_precision='round_trip');validate_history(out);return out
@lru_cache(maxsize=1)
def _download_archive():
    info=json.loads((ROOT/'examples/provenance.json').read_text())
    request=urllib.request.Request(info['source_url'],headers={'User-Agent':'AI-FBI-training-demo/1.0'})
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request,timeout=60) as response:payload=response.read()
    if hashlib.sha256(payload).hexdigest()!=info['archive_sha256']:
        raise ValueError('NASA archive hash mismatch. Use your own canonical trajectory instead.')
    return payload
def download_train_member(subset):
    info=json.loads((ROOT/'examples/provenance.json').read_text())['samples'][subset]
    name=f'train_{subset}.txt'
    def find(payload,depth=0):
        if depth>2:raise ValueError('Unexpected NASA archive structure.')
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names=archive.namelist()
            target=next((n for n in names if Path(n).name==name),None)
            if target is not None:return archive.read(target)
            nested=next((n for n in names if Path(n).name=='CMAPSSData.zip'),None)
            if nested is not None:return find(archive.read(nested),depth+1)
        raise ValueError('Requested TRAIN file missing from NASA archive.')
    data=find(_download_archive())
    if hashlib.sha256(data).hexdigest()!=info['train_sha256']:raise ValueError('TRAIN source hash mismatch.')
    return data
def parse_input(payload,format='csv'):
    if len(payload)>5*1024*1024:raise DataValidationError('Input exceeds the 5 MB demo limit.')
    try:
        if format=='csv':out=pd.read_csv(io.BytesIO(payload),float_precision='round_trip')
        elif format=='nasa':
            out=pd.read_csv(io.BytesIO(payload),sep=r'\s+',header=None)
            if out.shape[1]!=26:raise DataValidationError('NASA input must have exactly 26 numeric columns.')
            out.columns=CANONICAL_COLUMNS
        else:raise DataValidationError('Choose CSV or NASA text.')
    except DataValidationError:raise
    except Exception as e:raise DataValidationError('Input could not be parsed as the chosen numeric format.') from e
    validate_history(out)
    if len(out)>5000:raise DataValidationError('Limit this demo to 5,000 observations per input.')
    return out
def infer(raw,subset):
    validate_history(raw)
    out=predict_bundle(load_bundle(subset),raw)
    if not np.isfinite(out.to_numpy(dtype=float)).all():raise ValueError('Model produced non-finite outputs.')
    return out
def process_prefix(raw,subset,state_dir):
    validate_history(raw)
    if raw.engine_id.nunique()!=1:raise DataValidationError('Process one engine at a time.')
    prior=Path(state_dir)/'agent_state.json'
    if prior.exists():
        saved=json.loads(prior.read_text(encoding='utf-8'))
        key=f'{subset}:{int(raw.engine_id.iloc[0])}'
        last=int(saved.get('engines',{}).get(key,{}).get('last_cycle',0))
        if int(raw.cycle.iloc[-1])<last:
            raise DataValidationError('This session already processed a later cycle. Start a new session before rewinding.')
    out=infer(raw,subset);record=metadata()['models'][subset]
    prefix=hashlib.sha256(','.join(CANONICAL_COLUMNS).encode())
    decisions=[];model_version=str(record['model_version']);info=[]
    cfg={'namespace':'ai-fbi-frozen-policy-demo'}
    for pos,r in enumerate(out.itertuples(index=False)):
        # Incremental prefix hash stays identical when later cycles are appended.
        prefix.update(raw.iloc[[pos]].to_csv(index=False,header=False,float_format='%.17g',lineterminator='\n').encode())
        input_hash=prefix.hexdigest()
        event={'subset':subset,'engine_id':int(r.engine_id),'cycle':int(r.cycle),
               'event_id':f'{subset}:{int(r.engine_id)}:{int(r.cycle)}','valid':True,
               'predicted_rul':float(r.predicted_rul),'warning_probability':0.,'warning_threshold':1.,
               'warning_horizon':30,'mad_score':float(r.mad_score),'mad_threshold':float(r.mad_threshold),
               'model_version':model_version,'input_hash':input_hash}
        result=agent_v3.process_observation(event,Path(state_dir),cfg)
        if result['status'] not in ('processed','replayed'):
            raise DataValidationError('This session contains a changed or out-of-order event. Start a new session.')
        decisions.append(result)
        info.append({'cycle':int(r.cycle),'warning_probability':float(r.warning_probability),
                     'warning_threshold':float(r.warning_threshold),'role':'INFORMATIONAL_ONLY',
                     'agent_probability_control':0.,'agent_threshold_control':1.,'input_hash':input_hash})
    state=json.loads((Path(state_dir)/'agent_state.json').read_text(encoding='utf-8'))
    orders=json.loads((Path(state_dir)/'work_orders.json').read_text(encoding='utf-8'))
    open_keys=[(o['asset']['subset'],o['asset']['engine_id']) for o in orders if o['status']=='OPEN']
    summary={'observed_cycles':len(out),'processed_events':sum(d['status']=='processed' for d in decisions),
             'exact_replays':sum(d['status']=='replayed' for d in decisions),'open_orders':len(open_keys),
             'duplicate_open_orders':len(open_keys)-len(set(open_keys)),
             'severity':decisions[-1]['decision']['severity'],'input_prefix_sha256':prefix.hexdigest(),
             'model_version':model_version,'model_sha256':record['sha256'],
             'state_sha256':sha256(Path(state_dir)/'agent_state.json')}
    return {'predictions':out,'decisions':decisions,'orders':orders,'summary':summary,
            'informational_warning':info,'model_sha256':record['sha256'],
            'scope':'Fresh frozen-model inference. Bundled example is fitting-TRAIN data; no accuracy estimate.'}
