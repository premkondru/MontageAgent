import sys, os
import yaml, json
import time
import streamlit as st

# Ensure repo root is importable
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

st.subheader('Upload images (optional)')
uploads = st.file_uploader('Drop JPG/PNG files', type=['jpg','jpeg','png'], accept_multiple_files=True)

session_dir = None
if uploads:
    ts = int(time.time())
    session_dir = os.path.join('data', 'events', f'upload_session_{ts}')
    os.makedirs(session_dir, exist_ok=True)
    for uf in uploads:
        fname = os.path.basename(uf.name)
        safe = ''.join(c for c in fname if (c.isalnum() or c in ('-','_','.'))).strip('.')
        target = os.path.join(session_dir, safe or f'upload_{ts}.jpg')
        with open(target, 'wb') as out:
            out.write(uf.getbuffer())
    st.success(f'Uploaded {len(uploads)} image(s) to {session_dir}')

if st.button('Run Pipeline'):
    sup = Supervisor(cfg)
    results = sup.run()
    for r in results:
        st.write(f'**{r.name}**:', r.output)

    posts = None
    for r in results:
        if r.name == 'captioner' and isinstance(r.output, dict) and 'posts' in r.output:
            posts = r.output['posts']
            break

    if posts:
        st.subheader('Preview Posts')
        for p in posts:
            st.image(p['image_path'])
            st.markdown(f"""**Caption**  
{p['caption']}""")
            st.write('**Hashtags:**', ' '.join(p.get('hashtags', [])))
            if p.get('labels'):
                st.write('**Labels:**', ', '.join(p['labels']))
