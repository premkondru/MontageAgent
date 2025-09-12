import sys, os, time, yaml, json
import streamlit as st
from PIL import Image

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from agent.supervisor import Supervisor

def resize_for_instagram(img_path: str, target_ratio=(4,5), target_size=(1080,1350)) -> Image.Image:
    im = Image.open(img_path).convert("RGB")
    w,h = im.size
    target_aspect = target_ratio[0]/target_ratio[1]
    cur = w/h
    if cur > target_aspect:
        new_w = int(h*target_aspect); left=(w-new_w)//2
        im = im.crop((left,0,left+new_w,h))
    elif cur < target_aspect:
        new_h = int(w/target_aspect); top=(h-new_h)//2
        im = im.crop((0,top,w,top+new_h))
    return im.resize(target_size, Image.LANCZOS)

def apply_zoom(im: Image.Image, z: float) -> Image.Image:
    z = max(0.5, min(2.0, float(z)))
    w,h = im.size
    return im.resize((int(w*z), int(h*z)), Image.LANCZOS)

st.set_page_config(page_title="Montage Photo Agent", layout="wide")
st.title("Montage Photo Agent")
st.write("Automate sorting → **dedupe (CLIP)** → **clustering (CLIP)** → captioning → (optional) publishing.")

cfg = {}
if os.path.exists("configs/agent.yaml"):
    with open("configs/agent.yaml","r") as f:
        cfg = yaml.safe_load(f) or {}

for k, v in [("upload_session_dir", None), ("results", None), ("posts", None), ("preview_zoom", 1.0)]:
    if k not in st.session_state: st.session_state[k]=v

st.subheader("Preview controls")
c1,c2,c3,c4 = st.columns([1,1,2,8])
with c1:
    if st.button("Zoom -"): st.session_state.preview_zoom = max(0.5, round(st.session_state.preview_zoom-0.1,2))
with c2:
    if st.button("Zoom +"): st.session_state.preview_zoom = min(2.0, round(st.session_state.preview_zoom+0.1,2))
with c3:
    if st.button("Reset"): st.session_state.preview_zoom = 1.0
with c4:
    st.write(f"Current Zoom: **{int(st.session_state.preview_zoom*100)}%**")

st.subheader("Upload images (optional)")
uploads = st.file_uploader("Drop JPG/PNG files", type=["jpg","jpeg","png"], accept_multiple_files=True)
if uploads:
    if not st.session_state.upload_session_dir:
        ts=int(time.time()); st.session_state.upload_session_dir=os.path.join("data","events",f"upload_session_{ts}")
        os.makedirs(st.session_state.upload_session_dir, exist_ok=True)
    saved=0
    for i, uf in enumerate(uploads, 1):
        name=os.path.basename(uf.name)
        safe="".join(c for c in name if (c.isalnum() or c in ("-","_","."))).strip(".") or f"upload_{i}.jpg"
        tgt=os.path.join(st.session_state.upload_session_dir, safe)
        if not os.path.exists(tgt):
            with open(tgt,"wb") as out: out.write(uf.getbuffer()); saved+=1
    if saved:
        st.success(f"Saved {saved} new file(s) to `{st.session_state.upload_session_dir}`")
    else:
        st.info(f"Files already saved in `{st.session_state.upload_session_dir}`")


d1,d2,d3,d4 = st.columns([1,1,2,6])
with d1:
    run_clicked = st.button("Run Pipeline", type="primary")
with d2:
    use_upload_only = st.checkbox("Use only current upload session", value=False)
with d3:
    if st.button("Clear Preview"): st.session_state.results=None; st.session_state.posts=None

if run_clicked:
    runtime_cfg=dict(cfg)
    if "ingest" not in runtime_cfg: runtime_cfg["ingest"]={}
    if use_upload_only and st.session_state.upload_session_dir:
        runtime_cfg["ingest"]["dirs"]=[st.session_state.upload_session_dir]
    sup=Supervisor(runtime_cfg); results=sup.run(); st.session_state.results=results
    posts=None
    for r in results:
        if r.name=="captioner" and isinstance(r.output,dict) and "posts" in r.output: posts=r.output["posts"]; break
    st.session_state.posts=posts

if st.session_state.results:
    with st.expander("Pipeline step outputs", expanded=False):
        for r in st.session_state.results:
            st.write(f"**{r.name}**")
            try: st.json(r.output)
            except Exception: st.write(r.output)
            if getattr(r,"success",True) is False:
                st.error(getattr(r,"error","unknown error"))

posts = st.session_state.posts
if posts:
    st.subheader("Preview Posts (per cluster)")
    for idx, p in enumerate(posts):
        images=[ip for ip in (p.get("images") or []) if isinstance(ip,str)]
        n=len(images)
        st.markdown(f"### Post {idx+1} — {n} photo(s)")
        if p.get("labels"): st.write("**Labels:**", ", ".join(p["labels"]))
        st.markdown("**Caption**"); st.write(p["caption"])
        st.write("**Hashtags:**", " ".join(p.get("hashtags", [])))

        if n==0:
            st.warning("This cluster contains no previewable images."); st.divider(); continue

        cols=st.columns([3,1])
        with cols[0]:
            if n==1:
                img_path=images[0]
                if os.path.exists(img_path):
                    im=resize_for_instagram(img_path); st.image(apply_zoom(im, st.session_state.preview_zoom))
                else: st.info(f"(Missing file) {img_path}")
            else:
                options=list(range(1,n+1))
                choice=st.select_slider(f"Image in cluster {idx+1}", options=options, value=options[0], key=f"car_{idx}")
                img_path=images[choice-1]
                if os.path.exists(img_path):
                    im=resize_for_instagram(img_path); st.image(apply_zoom(im, st.session_state.preview_zoom))
                else: st.info(f"(Missing file) {img_path}")
        with cols[1]:
            if n>1: st.caption("Use the selector to browse this cluster.")
        st.divider()

    # Export
    export_rows=[{"caption":p.get("caption",""),"hashtags":p.get("hashtags",[]),"images":[ip for ip in (p.get("images") or []) if isinstance(ip,str)]} for p in posts]
    export_obj={"generated_at": int(time.time()), "posts": export_rows}
    export_json=json.dumps(export_obj, indent=2)
    st.subheader("Export")
    st.download_button("Download Instagram Carousel JSON", data=export_json, file_name=f"ig_carousels_{int(time.time())}.json", mime="application/json", use_container_width=True)
