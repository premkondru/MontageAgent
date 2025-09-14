# Montage Photo Agent — Streamlit UI (No-HTML, PIL Card)
# - Upload images (persisted across reruns)
# - Run pipeline (Supervisor) once; persist results/posts in session_state
# - Preview per cluster with: card border, IG-like composed image (black box), top ◀ ▶ nav
# - IG-sized (4:5, 1080x1350) with zoom (starts at 25%)
# - CLIP dedupe+clustering; Export JSON for IG carousels

import sys, os, time, yaml, json, importlib.util, re
from pathlib import Path
from io import BytesIO

import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageColor

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
    """Center-crop to 4:5 and resize to 1080x1350."""
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    ta = target_ratio[0] / target_ratio[1]
    ca = w / h
    if ca > ta:
        new_w = int(h * ta)
        left = (w - new_w) // 2
        im = im.crop((left, 0, left + new_w, h))
    elif ca < ta:
        new_h = int(w / ta)
        top = (h - new_h) // 2
        im = im.crop((0, top, w, top + new_h))
    im = im.resize(target_size, Image.LANCZOS)
    return im

def apply_zoom(im: Image.Image, zoom: float) -> Image.Image:
    """Scale composed preview. 25%–200%."""
    z = max(0.25, min(2.0, float(zoom)))
    w, h = im.size
    return im.resize((max(1, int(w * z)), max(1, int(h * z))), Image.LANCZOS)

def strip_hashtags(text: str) -> str:
    if not text:
        return ""
    txt = re.sub(r'(^|\s)#[\w_]+', r'\1', text)
    return re.sub(r'\s{2,}', ' ', txt).strip()

def _load_font(size: int) -> ImageFont.ImageFont:
    """
    Try to load a nice TTF font; fall back to PIL default.
    """
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
        "/System/Library/Fonts/Supplemental/Arial.ttf",     # macOS
        "/Library/Fonts/Arial.ttf",                         # macOS (alt)
        "C:\\Windows\\Fonts\\arial.ttf",                    # Windows
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return ImageFont.load_default()

def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    """Greedy wrap text so that each line fits within max_width."""
    words = (text or "").split()
    lines = []
    cur = []
    for w in words:
        test = (" ".join(cur + [w])).strip()
        if draw.textlength(test, font=font) <= max_width:
            cur.append(w)
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines

def compose_ig_card(base_img: Image.Image, caption: str, hashtags: list[str]) -> Image.Image:
    """
    Build a single image containing:
      - black rectangular border (outer)
      - the IG-cropped image
      - caption (bold-ish)
      - hashtags (lighter)
    No HTML used.
    """
    # Frame metrics
    frame_border = 8        # outer black border thickness
    inner_pad = 20          # white padding inside the black frame
    gap_img_to_text = 16
    gap_lines = 6

    # Typography
    cap_font = _load_font(36)
    tag_font = _load_font(30)
    cap_color = (0, 0, 0)
    tag_color = (40, 40, 40)

    # Box width equals image width + paddings
    img_w, img_h = base_img.size
    box_inner_w = img_w
    text_max_w = box_inner_w

    # Prepare text
    caption = caption or ""
    tags_line = " ".join(hashtags or [])

    # Measure wrapped text
    tmp = Image.new("RGB", (10, 10), "white")
    draw = ImageDraw.Draw(tmp)
    cap_lines = _wrap_text(draw, caption, cap_font, text_max_w)
    tag_lines = _wrap_text(draw, tags_line, tag_font, text_max_w) if tags_line else []

    # Compute text block height
    def line_height(font):  # conservative height
        ascent, descent = font.getmetrics() if hasattr(font, "getmetrics") else (font.size, 0)
        return ascent + descent + 4

    cap_h = sum(line_height(cap_font) for _ in cap_lines) if cap_lines else 0
    tag_h = sum(line_height(tag_font) for _ in tag_lines) if tag_lines else 0
    text_block_h = (gap_img_to_text if (cap_h or tag_h) else 0) + cap_h + (gap_lines if (cap_h and tag_h) else 0) + tag_h

    # Final card size
    card_w = box_inner_w + 2 * (inner_pad + frame_border)
    card_h = img_h + text_block_h + 2 * (inner_pad + frame_border)

    # Create white canvas then draw black border
    card = Image.new("RGB", (card_w, card_h), "white")
    # Outer black rectangle
    ImageDraw.Draw(card).rectangle([(0, 0), (card_w - 1, card_h - 1)], outline="black", width=frame_border)

    # Paste image (top area)
    x0 = frame_border + inner_pad
    y0 = frame_border + inner_pad
    card.paste(base_img, (x0, y0))

    # Text area origin
    ty = y0 + img_h + (gap_img_to_text if (cap_h or tag_h) else 0)
    draw = ImageDraw.Draw(card)

    # Caption
    for line in cap_lines:
        draw.text((x0, ty), line, fill=cap_color, font=cap_font)
        ty += line_height(cap_font)

    # Gap between caption and tags
    if cap_lines and tag_lines:
        ty += gap_lines

    # Hashtags
    for line in tag_lines:
        draw.text((x0, ty), line, fill=tag_color, font=tag_font)
        ty += line_height(tag_font)

    return card


# ---------- Page ----------
st.set_page_config(page_title="Montage Photo Agent", layout="wide")
st.title("Montage Photo Agent")
st.write("Automate sorting → **dedupe (CLIP)** → **clustering (CLIP)** → captioning → (optional) publishing.")

# Minimal CSS for card container
st.markdown("""
<style>
.post-card{border:1px solid #d0d0d0; border-radius:8px; padding:14px; margin:18px 0; background:#fafafa;}
.thumb-caption{font-size:0.8rem;}
</style>
""", unsafe_allow_html=True)

# Load config (if present)
cfg = {}
cfg_path = repo_root / "configs" / "agent.yaml"
if cfg_path.exists():
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

# ---------- Run config (sidebar) ----------
with st.sidebar:
    st.header("Configurations")
    # 1) Categorize: labels
    default_labels = []
    # prefill from file config if available
    try:
        default_labels = (cfg.get("categorize", {}) or {}).get("labels", [])
    except Exception:
        pass
    labels_text = st.text_area(
        "Labels (, or /n separated)",
        value="\n".join(default_labels) if default_labels else "stage\naudience\nspeaker\ngroup photo\nportrait\nnight\naward\nsports\nfood\nindoors\noutdoors\ncandid",
        height=140,
        help="These are used by the categorizer/labeler (CLIP zero-shot).",
    )
    #ui_labels = [x.strip() for x in re.split(r"[,\\n]+", labels_text) if x.strip()]
    ui_labels = [x.strip() for x in re.split(r"[,\n\r]+", labels_text) if x.strip()]

    # 2) Cluster: max images per post
    max_imgs_default = int((cfg.get("clusterer", {}) or {}).get("max_images_per_post", 10))
    ui_max_images = st.slider(
        "Max images per post (cluster cap)",
        min_value=1, max_value=10, value=6, step=1,
        help="Upper bound of images that will be kept per clustered post."
    )

    # 3) Captioner: mode
    cap_mode_default = (cfg.get("captioner", {}) or {}).get("mode", "template")
    ui_cap_mode = st.selectbox(
        "Captioner mode",
        options=["template", "blip2"],
        index=0 if str(cap_mode_default).lower() == "template" else 1,
        help="Use 'blip2' to enable the (optional) LoRA/BLIP-2 captioner."
    )

    # 4) Preview zoom zc1, zc2, zc4 = st.columns([1, 1, 4])
    st.session_state.preview_zoom = st.slider(
        "Preview Zoom Level",
        min_value=0.25, max_value=1.00, value=0.35, step=0.05,
        help="Preview zoom level of the Instagram Posts"
    )
   
    if st.button("Clear Preview"):
        st.session_state.results = None
        st.session_state.posts = None
        st.session_state.include_map = {}

# ---------- Persistent state ----------
for key, default in [
    ("upload_session_dir", None),
    ("results", None),
    ("posts", None),
    ("label_index", {}),       # <- NEW
    ("preview_zoom", 0.35),     # Start small
    ("include_map", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------- Event Name ----------
default_event = st.session_state.get("event_name_override", "")
event_input = st.text_input(
    "Event Name",
    value=default_event,
    placeholder="e.g., IITG Orientation 2025",
    help="Default is the name of the folder where the images are uploaded"
)
st.session_state.event_name_override = event_input.strip()

# ---------- Upload images ----------
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
        safe = "".join(c for c in fname if (c.isalnum() or c in ("-", "_", "."))).strip(".") or f"upload_{i}.jpg"
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
use_upload_only = st.checkbox("Use only current upload session", value=False)
run_clicked = st.button("Run Pipeline", type="primary")
# Run pipeline on demand
if run_clicked:
    runtime_cfg = dict(cfg)
    runtime_cfg.setdefault("ingest", {})
    if use_upload_only and st.session_state.upload_session_dir:
        runtime_cfg["ingest"]["dirs"] = [st.session_state.upload_session_dir]


    # ---- inject UI config into runtime cfg ----
    runtime_cfg.setdefault("categorize", {})
    runtime_cfg["categorize"]["labels"] = ui_labels

    runtime_cfg.setdefault("cluster", {})
    runtime_cfg["cluster"]["max_images_per_post"] = int(ui_max_images)

    runtime_cfg.setdefault("captioner", {})
    runtime_cfg["captioner"]["mode"] = str(ui_cap_mode).lower()
    runtime_cfg["captioner"]["event_name_override"] = st.session_state["event_name_override"]

    sup = Supervisor(runtime_cfg)
    results = sup.run()
    st.session_state.results = results

    posts = None
    label_index = {}
    for r in results:
        if r.name == "captioner" and isinstance(r.output, dict) and "posts" in r.output:
            posts = r.output["posts"]
            label_index = r.output.get("label_index", {})   # <- NEW
            break
    st.session_state.posts = posts
    st.session_state.label_index = label_index 

# Optional debug
if st.session_state.results:
    with st.expander("Pipeline step outputs", expanded=False):
        for r in st.session_state.results:
            st.write(f"**{r.name}**")
            try:
                st.json(r.output)
            except Exception:
                st.write(r.output)

# ---------- Preview posts ----------
posts = st.session_state.posts
if posts:
    #st.subheader("Preview Posts (per cluster)")
    with st.expander("Preview Posts (per cluster)", expanded=True):
        for idx, p in enumerate(posts):
            images = [ip for ip in (p.get("images") or []) if isinstance(ip, str)]
            n = len(images)
            # include/exclude map
            inc = st.session_state.include_map.get(idx)
            if inc is None:
                inc = {path: True for path in images}
                st.session_state.include_map[idx] = inc
            else:
                for path in images:
                    inc.setdefault(path, True)

            included = [path for path in images if inc.get(path, True)]
            n_included = len(included)
            with st.expander(f"**Post {idx+1}** — {n_included} selected / {n} total photo(s)", expanded=False):

                if n == 0:
                    st.warning("This cluster contains no previewable images.")
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.divider()
                    continue

                # Per-cluster index
                cur_key = f"car_{idx}"
                if cur_key not in st.session_state:
                    st.session_state[cur_key] = 1

                # NAV ROW (buttons side-by-side on the left)
                left_controls, _spacer = st.columns([2, 8])
                with left_controls:
                    cprev, cnext = st.columns([1, 1])
                    prev_clicked = cprev.button("◀", key=f"prev_{idx}", use_container_width=True, disabled=(n_included < 2))
                    next_clicked = cnext.button("▶", key=f"next_{idx}", use_container_width=True, disabled=(n_included < 2))
                    #st.caption(f"{st.session_state[cur_key]} / {n_included}")

                if n_included > 0:
                    if prev_clicked:
                        st.session_state[cur_key] = 1 if (st.session_state[cur_key] - 1) < 1 else (st.session_state[cur_key] - 1)
                    if next_clicked:
                        st.session_state[cur_key] = n_included if (st.session_state[cur_key] + 1) > n_included else (st.session_state[cur_key] + 1)

                # Compose and show IG-like card
                if n_included == 0:
                    st.info("No images selected. Use the checkboxes below to include images in this post.")
                else:
                    st.session_state[cur_key] = max(1, min(st.session_state[cur_key], n_included))
                    cur_img_path = included[st.session_state[cur_key] - 1]
                    if os.path.exists(cur_img_path):
                        base = resize_for_instagram(cur_img_path)
                        # Clean caption (no inline hashtags)
                        clean_caption = strip_hashtags(p.get("caption", ""))
                        card = compose_ig_card(base, clean_caption, p.get("hashtags", []))
                        zoomed = apply_zoom(card, st.session_state.preview_zoom)
                        st.image(zoomed)
                    else:
                        st.info(f"(Missing file) {cur_img_path}")

                # Thumbnails — tighter packing, 1/4 size of previous (216x270 -> 54x68)
                # Thumbnails — tiny + packed; show labels beneath each thumbnail
                st.write("**Thumbnails**")
                thumbs_per_row = 3
                thumb_w, thumb_h = 108, 135  # quarter-size thumbs
                label_index = st.session_state.get("label_index", {}) or {}

                for start in range(0, n, thumbs_per_row):
                    row_paths = images[start:start+thumbs_per_row]
                    try:
                        cols = st.columns(len(row_paths), gap=None)
                    except TypeError:
                        cols = st.columns(len(row_paths))
                    for j, img_path in enumerate(row_paths):
                        with cols[j]:
                            try:
                                thumb = resize_for_instagram(img_path).resize((thumb_w, thumb_h), Image.LANCZOS)
                                st.image(thumb)
                            except Exception:
                                st.info("(thumb unavailable)")

                            # NEW: labels under the thumbnail
                            labs = label_index.get(img_path, [])
                            st.caption(", ".join(labs) if labs else "—")

                            # Include/Exclude toggle
                            ck = st.checkbox("Include", value=inc.get(img_path, True), key=f"inc_{idx}_{start+j}")
                            inc[img_path] = ck

    # ---------- Export ----------
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
    st.download_button(
        "EXPORT: Download Instagram Carousel JSON",
        data=export_json,
        file_name=f"ig_carousels_{int(time.time())}.json",
        mime="application/json",
        use_container_width=True
    )
