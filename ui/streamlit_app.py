import sys, os, time, yaml, json, importlib.util
from pathlib import Path
import streamlit as st
from PIL import Image

# Make repo importable and robust fallback
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
try:
    from agent.supervisor import Supervisor
except ModuleNotFoundError:
    sup_path = repo_root / "agent" / "supervisor.py"
    spec = importlib.util.spec_from_file_location("agent.supervisor", sup_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    Supervisor = mod.Supervisor

def resize_for_instagram(img_path, target_ratio=(4,5), target_size=(1080,1350)):
    im = Image.open(img_path).convert("RGB")
    w,h = im.size
    ta = target_ratio[0]/target_ratio[1]
    ca = w/h
    if ca>ta:
        nw=int(h*ta); left=(w-nw)//2; im=im.crop((left,0,left+nw,h))
    elif ca<ta:
        nh=int(w/ta); top=(h-nh)//2; im=im.crop((0,top,w,top+nh))
    return im.resize(target_size, Image.LANCZOS)

def apply_zoom(im, z):
    z = max(0.5, min(2.0, float(z)))
    w,h=im.size
    return im.resize((int(w*z), int(h*z)), Image.LANCZOS)

st.set_page_config(page_title="Montage Photo Agent", layout="wide")
st.title("Montage Photo Agent")

cfg = {}
cfg_path = repo_root / "configs" / "agent.yaml"
if cfg_path.exists():
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

for k,v in [("upload_session_dir",None),("results",None),("posts",None),("preview_zoom",1.0),("include_map",{})]:
    if k not in st.session_state: st.session_state[k]=v

st.subheader("Preview controls")
c1,c2,c3,c4 = st.columns([1,1,2,8])
with c1:
    if st.button("Zoom -"): st.session_state.preview_zoom=max(0.5, round(st.session_state.preview_zoom-0.1,2))
with c2:
    if st.button("Zoom +"): st.session_state.preview_zoom=min(2.0, round(st.session_state.preview_zoom+0.1,2))
with c3:
    if st.button("Reset"): st.session_state.preview_zoom=1.0
with c4:
    st.write(f"Current Zoom: **{int(st.session_state.preview_zoom*100)}%**")

st.subheader("Upload images (optional)")
uploads = st.file_uploader("Drop JPG/PNG files", type=["jpg","jpeg","png"], accept_multiple_files=True)
if uploads:
    if not st.session_state.upload_session_dir:
        ts=int(time.time())
        st.session_state.upload_session_dir = str(repo_root / "data" / "events" / f"upload_session_{ts}")
        os.makedirs(st.session_state.upload_session_dir, exist_ok=True)
    saved=0
    for i,uf in enumerate(uploads,1):
        fname=os.path.basename(uf.name)
        safe="".join(c for c in fname if (c.isalnum() or c in("-","_","."))).strip(".") or f"upload_{i}.jpg"
        target=os.path.join(st.session_state.upload_session_dir, safe)
        if not os.path.exists(target):
            with open(target,"wb") as out: out.write(uf.getbuffer()); saved+=1
    if saved: st.success(f"Saved {saved} new file(s) to `{st.session_state.upload_session_dir}`")
    else: st.info(f"Files already saved in `{st.session_state.upload_session_dir}`")

a1,a2,a3,_ = st.columns([1,1,2,6])
with a1: run_clicked = st.button("Run Pipeline", type="primary")
with a2: use_upload_only = st.checkbox("Use only current upload session", value=False)
with a3:
    if st.button("Clear Preview"):
        st.session_state.results=None; st.session_state.posts=None; st.session_state.include_map={}

if run_clicked:
    rcfg=dict(cfg)
    rcfg.setdefault("ingest", {})
    if use_upload_only and st.session_state.upload_session_dir:
        rcfg["ingest"]["dirs"]=[st.session_state.upload_session_dir]
    sup=Supervisor(rcfg); results=sup.run(); st.session_state.results=results
    posts=None
    for r in results:
        if r.name=="captioner" and isinstance(r.output, dict) and "posts" in r.output:
            posts=r.output["posts"]; break
    st.session_state.posts=posts

if st.session_state.results:
    with st.expander("Pipeline step outputs"):
        for r in st.session_state.results:
            st.write(f"**{r.name}**"); st.json(r.output)

posts = st.session_state.posts
if posts:
    st.subheader("Preview Posts (per cluster)")
    for idx,p in enumerate(posts):
        images=[ip for ip in (p.get("images") or []) if isinstance(ip,str)]
        n=len(images)
        inc = st.session_state.include_map.get(idx) or {path: True for path in images}
        for path in images: inc.setdefault(path, True)
        st.session_state.include_map[idx]=inc
        included=[path for path in images if inc.get(path, True)]
        st.markdown(f"### Post {idx+1} — {len(included)} selected / {n} total photo(s)")
        st.markdown("**Caption**"); st.write(p["caption"])
        st.write("**Hashtags:**"," ".join(p.get("hashtags",[])))
        if n==0:
            st.warning("This cluster contains no previewable images."); st.divider(); continue
        cols=st.columns([3,1])
        with cols[0]:
            if len(included)==0:
                st.info("No images selected. Toggle below to include.")
            else:
                shown = included[0]
                if os.path.exists(shown):
                    im=resize_for_instagram(shown)
                    st.image(apply_zoom(im, st.session_state.preview_zoom))
                else:
                    st.info(f"(Missing file) {shown}")
        with cols[1]:
            st.caption("Use the thumbnail strip to select another image.")
        st.write("**Thumbnails**")
        for start in range(0,n,6):
            row=images[start:start+6]; c=st.columns(len(row))
            for j,ip in enumerate(row):
                with c[j]:
                    try:
                        th=resize_for_instagram(ip, target_size=(216,270)); st.image(th, caption=os.path.basename(ip))
                    except Exception: st.info("(thumb unavailable)")
                    ck=st.checkbox("Include", value=inc.get(ip,True), key=f"inc_{idx}_{start+j}")
                    inc[ip]=ck
                    if ck and st.button("Use", key=f"use_{idx}_{start+j}"):
                        included=[path for path in images if inc.get(path, True)]
                        if ip in included:
                            # No selector; just show first (this minimal UI keeps it simple)
                            pass
        st.divider()

    export_rows=[]
    for i,p in enumerate(posts):
        imgs=[ip for ip in (p.get('images') or []) if isinstance(ip,str)]
        inc=st.session_state.include_map.get(i,{}) if isinstance(st.session_state.include_map,dict) else {}
        sel=[path for path in imgs if inc.get(path, True)]
        export_rows.append({"caption":p.get("caption",""),"hashtags":p.get("hashtags",[]),"images":sel})
    export={"generated_at":int(time.time()),"posts":export_rows}
    st.subheader("Export")
    st.download_button("Download Instagram Carousel JSON", data=json.dumps(export, indent=2), file_name=f"ig_carousels_{int(time.time())}.json", mime="application/json", use_container_width=True)
