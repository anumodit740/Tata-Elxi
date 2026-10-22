"""Local React renderer and presentation helpers for the Streamlit app."""
from pathlib import Path
import streamlit as st

FRONTEND=Path(__file__).resolve().parent/'frontend'

def dashboard_component():
    return st.components.v2.component(
        'ai_fbi_engine_dashboard',
        css=(FRONTEND/'dashboard.css').read_text(encoding='utf-8'),
        js=(FRONTEND/'dashboard.js').read_text(encoding='utf-8'),
        isolate_styles=True,
    )

def render_dashboard(data):
    dashboard_component()(data=data,key='engine_dashboard',height='content',width='stretch')

def apply_styles():
    st.html('<style>'+(FRONTEND/'app.css').read_text(encoding='utf-8')+'</style>')
