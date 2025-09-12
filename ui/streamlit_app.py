# Montage Photo Agent — Streamlit UI (improved Preview)
# - Upload images (persisted across reruns)
# - Run pipeline (Supervisor) once; persist results/posts in session_state
# - Preview per cluster with: card border, IG-style frame, ◀ ▶ nav, thumbnails, include toggles
# - IG-sized (4:5, 1080x1350) with zoom (starts at 25%)
# - CLIP dedupe+clustering; Export JSON for IG carousels

import sys, os, time, yaml, json, importlib.util, base64
from io import BytesIO
from pathlib import Path
import streamlit as st
from PIL import Image

# Ensure repo root is importable even if Streamlit launched from elsewhere
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Try normal import; fall back to direct file import if needed
try:
    from agent.supervisor import Supervisor
except ModuleNotFoundError:
    sup_path = repo_root / "agent" / "supervisor.py"
    spec = importlib.util.spec_from_file_location("agent.supervisor", sup_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader, "Failed to load agent.supervisor"
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    Supervisor = mod.Supervisor

# ---------- Helpers ----------
def resize_for_instagram(img_path: str, target_ratio=(4, 5), target_size=(1080, 1350)) -> Image.Image:
    # Center-crops to 4:5 and resizes to 1080x1350 for IG portrait previews.
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    target_aspect = target_ratio[0] / target_ratio[1]
    current_aspect = w / h
    if current_aspect > target_aspect:
        new_w = int(h * target_aspect)
        left = (w - new_w) // 2
        im = im.crop((left, 0, left + new_w, h))
    elif current_aspect < target_aspect:
        new_h = int(w / target_aspect)
        top = (h - new_h) // 2
        im = im.crop((0, top, w, top + new_h))
    im = im.resize(target_size, Image.LANCZOS)
    return im

def apply_zoom(im: Image.Image, zoom: float) -> Image.Image:
    # Allow 25%–200% zoom
    zoom = max(0.35, min(2.0, float(zoom)))
    w, h = im.size
    return im.resize((int(w * zoom), int(h * zoom)), Image.LANCZOS)

def pil_to_base64(im: Image.Image) -> str:
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=92, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def render_ig_post_html(preview_im: Image.Image, caption: str, hashtags):
    b64 = pil_to_base64(preview_im)
    tags = " ".join(hashtags or [])
    w, h = preview_im.size
    html = f"""
    <div class="ig-frame">
      <img src="data:image/jpeg;base64,{b64}" width="{w}" height="{h}" style="display:block;height:auto;" />
      <div class="ig-caption">{caption}</div>
      <div class="ig-hashtags">{tags}</div>
    </div>
    """
    return html

# ---------- Page ----------
st.set_page_config(page_title="Montage Photo Agent", layout="wide")
st.title("Montage Photo Agent")
st.write("Automate sorting → **dedupe (CLIP)** → **clustering (CLIP)** → captioning → (optional) publishing.")

# Global CSS (post card + IG frame)
st.markdown("""
<style>
.post-card{border:1px solid #d0d0d0; border-radius:8px; padding:14px; margin:18px 0; background:#fafafa;}
.ig-frame{border:2px solid #000; padding:10px; background:#fff; display:inline-block;}
.ig-caption{font-weight:600; margin-top:10px; font-size:1.05rem;}
.ig-hashtags{color:#444; margin-top:6px; font-size:0.95rem; word-wrap:break-word;}
.navbtn{width:100%; height:42px; font-size:1.1rem;}
.thumb-caption{font-size:0.8rem;}
</style>
""", unsafe_allow_html=True)

# Load config (if present)
cfg = {}
cfg_path = repo_root / "configs" / "agent.yaml"
if cfg_path.exists():
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

# ---------- Persistent state ----------
for key, default in [
    ("upload_session_dir", None),
    ("results", None),
    ("posts", None),
    ("preview_zoom", 0.35),     # Start at 25% zoom (was 1.0 before)
    ("include_map", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------- Preview controls (Zoom) ----------
st.subheader("Preview controls")
zc1, zc2, zc3, zc4 = st.columns([1, 1, 2, 8])
with zc1:
    if st.button("Zoom -"):
        st.session_state.preview_zoom = max(0.35, round(st.session_state.preview_zoom - 0.1, 2))
with zc2:
    if st.button("Zoom +"):
        st.session_state.preview_zoom = min(2.0, round(st.session_state.preview_zoom + 0.1, 2))
with zc3:
    if st.button("Reset"):
        st.session_state.preview_zoom = 1.0
with zc4:
    st.write(f"Current Zoom: **{int(st.session_state.preview_zoom * 100)}%**")

# ---------- Upload images (persist across reruns) ----------
st.subheader("Upload images (optional)")
uploads = st.file_uploader(
    "Drop JPG/PNG files",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)
if uploads:
    if not st.session_state.upload_session_dir:
        ts = int(time.time())
        st.session_state.upload_session_dir = str(repo_root / "data" / "events" / f"upload_session_{ts}")
        os.makedirs(st.session_state.upload_session_dir, exist_ok=True)

    saved = 0
    for i, uf in enumerate(uploads, start=1):
        fname = os.path.basename(uf.name)
        safe = "".join(c for c in fname if (c.isalnum() or c in ("-", "_", "."))).strip(".")
        if not safe:
            safe = f"upload_{i}.jpg"
        target = os.path.join(st.session_state.upload_session_dir, safe)
        if not os.path.exists(target):
            with open(target, "wb") as out:
                out.write(uf.getbuffer())
            saved += 1

    if saved:
        st.success(f"Saved {saved} new file(s) to `{st.session_state.upload_session_dir}`")
    else:
        st.info(f"Files already saved in `{st.session_state.upload_session_dir}`")

# ---------- Actions ----------
c1, c2, c3, c4 = st.columns([1, 1, 2, 6])
with c1:
    run_clicked = st.button("Run Pipeline", type="primary")
with c2:
    use_upload_only = st.checkbox("Use only current upload session", value=False)
with c3:
    if st.button("Clear Preview"):
        st.session_state.results = None
        st.session_state.posts = None
        st.session_state.include_map = {}

# Run the pipeline only when requested; persist results/posts
if run_clicked:
    runtime_cfg = dict(cfg)
    if "ingest" not in runtime_cfg:
        runtime_cfg["ingest"] = {}
    if use_upload_only and st.session_state.upload_session_dir:
        runtime_cfg["ingest"]["dirs"] = [st.session_state.upload_session_dir]

    sup = Supervisor(runtime_cfg)
    results = sup.run()
    st.session_state.results = results

    posts = None
    for r in results:
        if r.name == "captioner" and isinstance(r.output, dict) and "posts" in r.output:
            posts = r.output["posts"]
            break
    st.session_state.posts = posts

# Optional: step outputs for debugging (also show errors if any)
if st.session_state.results:
    with st.expander("Pipeline step outputs", expanded=False):
        for r in st.session_state.results:
            st.write(f"**{r.name}**")
            try:
                st.json(r.output)
            except Exception:
                st.write(r.output)

# ---------- Preview posts (always from session state) ----------
posts = st.session_state.posts
if posts:
    st.subheader("Preview Posts (per cluster)")
    for idx, p in enumerate(posts):
        images = [ip for ip in (p.get("images") or []) if isinstance(ip, str)]
        n = len(images)

        # Init include/exclude map for this cluster
        inc = st.session_state.include_map.get(idx)
        if inc is None:
            inc = {path: True for path in images}
            st.session_state.include_map[idx] = inc
        else:
            for path in images:
                inc.setdefault(path, True)

        # Included images list and per-cluster carousel index
        included = [path for path in images if inc.get(path, True)]
        n_included = len(included)

        # Post card border
        st.markdown('<div class="post-card">', unsafe_allow_html=True)

        st.markdown(f"**Post {idx+1}** — {n_included} selected / {n} total photo(s)")

        if n == 0:
            st.warning("This cluster contains no previewable images.")
            st.markdown('</div>', unsafe_allow_html=True)  # close .post-card
            st.divider()
            continue

        # Maintain a per-cluster current index (1-based) and clamp to included list
        cur_key = f"car_{idx}"
        if cur_key not in st.session_state:
            st.session_state[cur_key] = 1
        if n_included == 0:
            # Nothing selected: show info instead of preview
            st.info("No images selected. Use the checkboxes below to include images in this post.")
        else:
            st.session_state[cur_key] = max(1, min(st.session_state[cur_key], n_included))

            navL, mid, navR = st.columns([1, 8, 1])
            with navL:
                st.button("◀", key=f"prev_{idx}", use_container_width=True, disabled=(n_included < 2))
                if st.session_state.get(f"prev_{idx}"):
                    st.session_state[cur_key] = 1 if (st.session_state[cur_key] - 1) < 1 else (st.session_state[cur_key] - 1)

            with mid:
                cur_img_path = included[st.session_state[cur_key] - 1]
                if os.path.exists(cur_img_path):
                    base = resize_for_instagram(cur_img_path)
                    zoomed = apply_zoom(base, st.session_state.preview_zoom)
                    # IG frame (black border) with caption+hashtags underneath, like Instagram
                    ig_html = render_ig_post_html(zoomed, p.get("caption", ""), p.get("hashtags", []))
                    st.markdown(ig_html, unsafe_allow_html=True)
                else:
                    st.info(f"(Missing file) {cur_img_path}")

            with navR:
                st.button("▶", key=f"next_{idx}", use_container_width=True, disabled=(n_included < 2))
                if st.session_state.get(f"next_{idx}"):
                    st.session_state[cur_key] = n_included if (st.session_state[cur_key] + 1) > n_included else (st.session_state[cur_key] + 1)

        # Thumbnails with include/exclude toggles (no "Use" button)
        st.write("**Thumbnails**")
        thumbs_per_row = 6
        for start in range(0, n, thumbs_per_row):
            row_paths = images[start:start+thumbs_per_row]
            cols = st.columns(len(row_paths))
            for j, img_path in enumerate(row_paths):
                with cols[j]:
                    try:
                        thumb = resize_for_instagram(img_path, target_size=(216, 270))
                        st.image(thumb, caption=os.path.basename(img_path))
                    except Exception:
                        st.info("(thumb unavailable)")

                    # Include/Exclude toggle (affects export and carousel list)
                    ck = st.checkbox("Include", value=inc.get(img_path, True), key=f"inc_{idx}_{start+j}")
                    inc[img_path] = ck

        # Close post card border
        st.markdown('</div>', unsafe_allow_html=True)
        st.divider()

    # ---------- Export IG carousel payloads (respecting include/exclude) ----------
    export_rows = []
    for p_idx, p in enumerate(posts):
        imgs = [ip for ip in (p.get("images") or []) if isinstance(ip, str)]
        inc = st.session_state.include_map.get(p_idx, {}) if isinstance(st.session_state.include_map, dict) else {}
        selected = [path for path in imgs if inc.get(path, True)]
        export_rows.append({
            "caption": p.get("caption", ""),
            "hashtags": p.get("hashtags", []),
            "images": selected
        })
    export_obj = {"generated_at": int(time.time()), "posts": export_rows}
    export_json = json.dumps(export_obj, indent=2)
    st.subheader("Export")
    st.download_button(
        "Download Instagram Carousel JSON",
        data=export_json,
        file_name=f"ig_carousels_{int(time.time())}.json",
        mime="application/json",
        use_container_width=True
    )
