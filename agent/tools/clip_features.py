# agent/tools/clip_features.py
import numpy as np, torch
from PIL import Image
import open_clip

_MODEL=None; _PRE=None; _DEV='cpu'; _TOKENIZER=None; _MODEL_NAME='ViT-B-32'; _PRETRAINED='laion2b_s34b_b79k'

def _lazy(model='ViT-B-32',pretrained='laion2b_s34b_b79k',device='cpu'):
    global _MODEL,_PRE,_DEV,_TOKENIZER,_MODEL_NAME,_PRETRAINED
    if (_MODEL is None) or (_PRE is None) or (_DEV!=device) or (_MODEL_NAME!=model) or (_PRETRAINED!=pretrained):
        _DEV = device if (device=='cpu' or (device.startswith('cuda') and torch.cuda.is_available())) else 'cpu'
        m,_,pp=open_clip.create_model_and_transforms(model, pretrained=pretrained, device=_DEV)
        m.eval(); _MODEL, _PRE = m, pp
        _TOKENIZER = open_clip.get_tokenizer(model)
        _MODEL_NAME, _PRETRAINED = model, pretrained
    return _MODEL,_PRE,_DEV

def _lazy_text():
    # Ensure model/tokenizer exist, then return both
    _lazy(_MODEL_NAME, _PRETRAINED, _DEV)
    return _MODEL, _TOKENIZER, _DEV

def encode_paths(paths, model='ViT-B-32', pretrained='laion2b_s34b_b79k', device='cpu', batch_size=16):
    m, pp, dev = _lazy(model, pretrained, device)
    ims=[]; idx=[]
    for i,p in enumerate(paths):
        try:
            ims.append(pp(Image.open(p).convert('RGB'))); idx.append(i)
        except Exception:
            ims.append(None)
    ok=[x for x in ims if x is not None]
    if not ok:
        out_dim=getattr(m.visual,'output_dim',512)
        return np.zeros((len(paths), out_dim), dtype=np.float32)
    fs=[]
    with torch.no_grad():
        for i in range(0,len(ok),batch_size):
            b=torch.stack(ok[i:i+batch_size]).to(dev)
            f=m.encode_image(b); f=f/(f.norm(dim=-1,keepdim=True)+1e-8); fs.append(f.cpu())
    fs=torch.cat(fs,0).numpy().astype('float32')
    out=np.zeros((len(paths), fs.shape[1]), dtype='float32')
    for j,i in enumerate(idx): out[i]=fs[j]
    return out

# NEW: encode natural-language prompts to CLIP text embeddings
def encode_texts(texts, model='ViT-B-32', pretrained='laion2b_s34b_b79k', device='cpu', batch_size=64):
    _ = _lazy(model, pretrained, device)
    m, tok, dev = _lazy_text()
    feats=[]
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            t = tok(texts[i:i+batch_size]).to(dev)
            f = m.encode_text(t)
            f = f / (f.norm(dim=-1, keepdim=True) + 1e-8)
            feats.append(f.cpu())
    return torch.cat(feats,0).numpy().astype('float32')
