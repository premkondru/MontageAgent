import numpy as np, torch
from PIL import Image
import open_clip
_MODEL=None; _PRE=None; _DEV='cpu'

def _lazy(model='ViT-B-32',pretrained='laion2b_s34b_b79k',device='cpu'):
    global _MODEL,_PRE,_DEV
    if _MODEL is None or _PRE is None or _DEV!=device:
        _DEV = device if (device=='cpu' or (device.startswith('cuda') and torch.cuda.is_available())) else 'cpu'
        m,_,pp=open_clip.create_model_and_transforms(model, pretrained=pretrained, device=_DEV)
        m.eval(); _MODEL, _PRE = m, pp
    return _MODEL,_PRE,_DEV

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
