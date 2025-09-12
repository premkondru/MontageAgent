import streamlit as st
import yaml, json
import sys, os
# Ensure repo root is importable when running `streamlit run ui/streamlit_app.py`
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from agent.supervisor import Supervisor

st.set_page_config(page_title='Montage Photo Agent', layout='wide')

st.title('Montage Photo Agent')
st.write('Automate sorting → selection → captioning → (optional) publishing.')

cfg_path = 'configs/agent.yaml'
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)

if st.button('Run Pipeline'):
    sup = Supervisor(cfg)
    results = sup.run()
    for r in results:
        st.write(f'**{r.name}**:', r.output)

st.info('Place images under `data/events/sample_event_day*/`. This is a demo UI.')
