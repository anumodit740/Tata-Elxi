"""AI-FBI engine workspace: Streamlit controls, local React views, frozen inference."""
from pathlib import Path
from html import escape
import hashlib
import json
import tempfile

import streamlit as st
from service import SUBSETS, metadata, load_sample, parse_input, process_prefix
from ui import apply_styles, render_dashboard

st.set_page_config(page_title='AI-FBI | Engine Health',page_icon='✈',layout='wide',initial_sidebar_state='expanded')
apply_styles()

def footer():
    st.markdown('<div class="footer-note"><span>AI-FBI · Anumodit Shukla · Yogesh Singh · Anoop Kumar</span><span>Tata Elxsi Teliport / Case 1 · No LLM</span></div>',unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-icon">≋</div><div><div class="brand-name">AI-FBI</div><div class="brand-sub">Engine intelligence</div></div></div>',unsafe_allow_html=True)
    st.markdown('<div class="side-label">01 / Data source</div>',unsafe_allow_html=True)
    subset=st.selectbox('C-MAPSS subset',SUBSETS)
    source=st.radio('Input source',['NASA TRAIN example','Upload trajectory'])
    uploaded=None
    if source=='Upload trajectory':
        fmt=st.selectbox('File format',['CSV with canonical headers','NASA 26-column text'])
        uploaded=st.file_uploader('Trajectory file',type=['csv','txt'],help='Complete available history from cycle 1. Maximum 5 MB and 5,000 observations.')
    st.markdown('<div class="side-policy"><strong>Decisions with a trace</strong><p>Observed history → RUL → confirmed rules → work order.</p><p>Classifier and deviation signals are informational. They do not create maintenance orders.</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="side-label">Session controls</div>',unsafe_allow_html=True)
    reset=st.button('Start new session',use_container_width=True,help='Clears this session’s decisions so you can replay an earlier cycle or a different history.')
    st.caption('Orders are temporary. Export before restarting. A NORMAL state does not certify engine health.')

if reset or 'agent_tmp' not in st.session_state:
    previous=st.session_state.get('agent_tmp')
    if previous is not None:previous.cleanup()
    st.session_state.agent_tmp=tempfile.TemporaryDirectory(prefix='ai_fbi_session_')
    st.session_state.engine_fingerprints={}
    st.session_state.pop('result',None)

st.markdown('<div class="ops-eyebrow">Operations workspace <span>/</span> Engine health</div><div class="ops-title-row"><h1>Predictive maintenance</h1><span class="ops-tag"><i></i>Frozen-model inference</span></div><p class="ops-subtitle">Inspect an observed trajectory. Estimate remaining cycles. Follow every maintenance decision back to its evidence.</p>',unsafe_allow_html=True)

try:
    if source=='NASA TRAIN example':
        with st.spinner('Loading the hash-verified NASA TRAIN example…'):raw=load_sample(subset)
        source_label='Fitting-TRAIN example'
        st.caption('TRAIN example · In-sample runtime demonstration, not a held-out accuracy test. The model receives only the selected observed history.')
    elif uploaded is None:
        render_dashboard({'mode':'empty','subset':subset,'source_label':'Upload trajectory','message':'Choose your engine history','detail':'Upload a canonical CSV or NASA 26-column text file. Supply consecutive observations starting at cycle 1.'})
        footer()
        st.stop()
    else:
        raw=parse_input(uploaded.getvalue(),'csv' if fmt.startswith('CSV') else 'nasa')
        source_label='Uploaded trajectory'
        st.caption('Uploaded trajectory · Select the matching subset. Complete available history from cycle 1 is required.')

    with st.container(border=True):
        left,right=st.columns([1,3.3],vertical_alignment='bottom')
        with left:engine=st.selectbox('Engine',raw.engine_id.unique().tolist())
        raw=raw[raw.engine_id==engine].reset_index(drop=True)
        identity=hashlib.sha256(raw.to_csv(index=False,float_format='%.17g').encode()).hexdigest()
        last_available=int(raw.cycle.max())
        with right:
            if last_available==1:
                cutoff=1
                st.caption('Observed cycle: 1 · Single observation')
            else:
                cutoff=st.slider('Observed cycle',min_value=1,max_value=last_available,value=min(100,last_available),key=f'cycle_{subset}_{identity}',help='Only observations at or before this cycle enter the model. Start a new session before rewinding after a run.')
        st.markdown(f'<div class="context-strip"><div><small>Model subset</small><strong>{escape(subset)}</strong></div><div><small>Engine</small><strong>#{int(engine):03d}</strong></div><div><small>Observed prefix</small><strong>{cutoff} / {last_available} cycles</strong></div><div><small>Input</small><strong>{escape(source_label)}</strong></div></div>',unsafe_allow_html=True)
        display_key=[subset,int(engine),identity,cutoff]
        current=st.session_state.get('result')
        is_current=bool(current and current.get('display_key')==display_key)
        a,b,c=st.columns([2.2,1.8,2.6],vertical_alignment='center')
        run=a.button('Run model and agent',type='primary',use_container_width=True)
        retry=b.button('Retry identical prefix',use_container_width=True,disabled=not is_current)
        c.caption('Causal features · Hash-checked model · Explicit rules')

    if run or retry:
        fingerprint_key=f'{subset}:{int(engine)}'
        known=st.session_state.engine_fingerprints.get(fingerprint_key)
        if known is not None and known!=identity:
            raise ValueError('This engine history differs from the one already processed. Start a new session before changing its observations.')
        prefix=raw[raw.cycle<=cutoff].copy()
        directory=Path(st.session_state.agent_tmp.name)/subset/str(int(engine))
        with st.spinner('Running the model and applying maintenance rules…'):
            result=process_prefix(prefix,subset,directory)
        result['display_key']=display_key
        st.session_state.engine_fingerprints[fingerprint_key]=identity
        st.session_state.result=result
        if run:st.rerun()

    result=st.session_state.get('result')
    if not result or result.get('display_key')!=display_key:
        render_dashboard({'mode':'empty','subset':subset,'engine':int(engine),'observed_cycle':cutoff,'last_available':last_available,'source_label':source_label,'message':'Your next decision starts here','detail':'Run the selected observed prefix to see remaining-life estimates, decision rules and the resulting maintenance record.'})
        footer()
        st.stop()

    out=result['predictions']
    last=out.iloc[-1]
    s=result['summary']
    m1,m2,m3,m4=st.columns(4)
    m1.metric('Predicted RUL',f'{last.predicted_rul:.1f} cycles',help='Estimated remaining operating cycles. This is a prediction, not a guaranteed service interval.')
    m2.metric('Active agent severity',s['severity'],help='Current rule state. An earlier maintenance order can remain open at a higher severity until human closure.')
    m3.metric('Open work orders',s['open_orders'],help='Persisted open maintenance episodes for this engine in this session.')
    m4.metric('Duplicate open orders',s['duplicate_open_orders'],help='More than one open order for the same engine. Expected to remain zero under identical retries.')
    st.markdown(f'<div class="run-receipt"><span>MODEL <b>{escape(subset)} / v{escape(str(s["model_version"]))}</b></span><code>{escape(s["model_sha256"][:12])}…</code><span>This run: <b>{s["processed_events"]} fresh</b> · <b>{s["exact_replays"]} identical replays</b></span></div>',unsafe_allow_html=True)
    if retry and s['exact_replays']==len(out):
        st.success('Identical retry accepted. No duplicate open order was created.')

    render_dashboard({'mode':'ready','subset':subset,'engine':int(engine),'observed_cycle':cutoff,'last_available':last_available,'source_label':source_label,'rows':out.to_dict(orient='records'),'summary':s,'orders':result['orders'],'decisions':result['decisions'][-1:]})

    st.markdown('<div class="export-heading">Export this run</div>',unsafe_allow_html=True)
    e1,e2,e3=st.columns(3)
    e1.download_button('Download work orders',json.dumps(result['orders'],indent=2),file_name=f'{subset}_work_orders.json',mime='application/json',use_container_width=True)
    e2.download_button('Download predictions',out.to_csv(index=False,float_format='%.17g'),file_name=f'{subset}_predictions.csv',mime='text/csv',use_container_width=True)
    receipt={k:v for k,v in result.items() if k not in ('predictions','decisions','display_key')}
    e3.download_button('Download inference receipt',json.dumps(receipt,indent=2),file_name=f'{subset}_inference_receipt.json',mime='application/json',use_container_width=True)

    with st.expander('Model, input and validation details'):
        st.json({'subset':subset,'engine_id':int(engine),'observed_cycle':cutoff,'model_sha256':s['model_sha256'],'input_prefix_sha256':s['input_prefix_sha256'],'state_sha256':s['state_sha256'],'target_cap_cycles':metadata()['target_cap'],'classifier_role':'INFORMATIONAL_ONLY'})
        st.caption('Missing values, incorrect schemas, incomplete histories and conflicting events stop processing. Broad automated ingestion review and rollback remain planned. Cloud/session storage is temporary; download outputs before restarting.')
        st.caption('All official TEST sets were previously inspected. Historical TEST results remain “benchmark, previously inspected; no selection use”. This app does not include or open TEST data.')
    footer()
except Exception as exc:
    st.session_state.pop('result',None)
    st.error(str(exc) if isinstance(exc,ValueError) else 'A model or runtime file is unavailable. Verify the package files and start a new session.')
    st.caption('No new order should be accepted from invalid input. Correct the trajectory or start a new session; existing session orders are not automatically closed.')
    footer()
