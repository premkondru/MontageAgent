from typing import List
import torch, numpy as np
from PIL import Image
import open_clip
_MODEL=_PREPROC=None; _DEVICE='cpu'
def _lazy_load(model_name='ViT-B-32', pretrained='laion2b_s34b_b79k', device='cpu'):
    global _MODEL,_PREPROC,_DEVICE
    if _MODEL is None or _PREPROC is None or _DEVICE!=device:
        _DEVICE = device if (device=='cpu' or (device.startswith('cuda') and torch.cuda.is_available())) else 'cpu'
        model,_,preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=_DEVICE)
        model.eval(); _MODEL, _PREPROC = model, preprocess
    return _MODEL,_PREPROC,_DEVICE
def encode_paths(paths: List[str], model_name='ViT-B-32', pretrained='laion2b_s34b_b79k', device='cpu', batch_size=16)->np.ndarray:
    model,preprocess,dev = _lazy_load(model_name, pretrained, device)
    imgs=[]; idx=[]
    for i,p in enumerate(paths):
        try:
            im=Image.open(p).convert('RGB'); imgs.append(preprocess(im)); idx.append(i)
        except Exception: imgs.append(None)
    valid=[im for im in imgs if im is not None]
    if not valid:
        out_dim=getattr(model.visual,'output_dim',512)
        import numpy as np
        return np.zeros((len(paths), out_dim), dtype=np.float32)
    feats=[]
    import torch
    with torch.no_grad():
        for i in range(0,len(valid),batch_size):
            b=torch.stack(valid[i:i+batch_size]).to(dev)
            f=model.encode_image(b); f=f/(f.norm(dim=-1,keepdim=True)+1e-8)
            feats.append(f.cpu())
    feats=torch.cat(feats,0).numpy().astype('float32')
    import numpy as np
    out=np.zeros((len(paths), feats.shape[1]), dtype='float32')
    j=0
    for i in range(len(imgs)):
        if imgs[i] is not None:
            out[i]=feats[j]; j+=1
    return out
