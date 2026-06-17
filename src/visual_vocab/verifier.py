from __future__ import annotations
import numpy as np
from PIL import Image
from .device import resolve_device

class OpenCLIPVerifier:
    def __init__(self, labels:list[str], model_name='ViT-B-32', pretrained='laion2b_s34b_b79k', device='auto', templates=None):
        import torch, open_clip
        self.torch=torch; self.device=resolve_device(device)
        self.model,_,self.preprocess=open_clip.create_model_and_transforms(model_name,pretrained=pretrained,device=self.device)
        self.model.eval(); self.tokenizer=open_clip.get_tokenizer(model_name)
        self.templates=templates or ['a clear photo of a {}','a close-up photo of a {}']
        self.labels=list(labels); self.text_features=self._encode_labels(self.labels)
    def _autocast(self):
        if self.device.startswith('cuda'): return self.torch.autocast(device_type='cuda',dtype=self.torch.float16)
        from contextlib import nullcontext; return nullcontext()
    def _encode_labels(self,labels):
        feats=[]
        with self.torch.no_grad(), self._autocast():
            for label in labels:
                texts=[t.format(label) for t in self.templates]
                toks=self.tokenizer(texts).to(self.device)
                f=self.model.encode_text(toks); f=f/f.norm(dim=-1,keepdim=True); f=f.mean(dim=0); f=f/f.norm()
                feats.append(f)
        return self.torch.stack(feats,dim=0)
    def classify(self,crop:Image.Image,candidate_labels:list[str]|None=None,top_k=5)->list[tuple[str,float]]:
        labels=candidate_labels or self.labels
        idx=[self.labels.index(x) for x in labels if x in self.labels]
        if not idx: return []
        image=self.preprocess(crop.convert('RGB')).unsqueeze(0).to(self.device)
        with self.torch.no_grad(), self._autocast():
            f=self.model.encode_image(image); f=f/f.norm(dim=-1,keepdim=True)
            tf=self.text_features[idx]
            probs=(100.0*(f@tf.T)).softmax(dim=-1)[0]
        vals,order=self.torch.topk(probs,min(top_k,len(idx)))
        return [(self.labels[idx[int(i)]],float(v)) for v,i in zip(vals.detach().cpu(),order.detach().cpu())]

class MockVerifier:
    def __init__(self,mapping=None): self.mapping=mapping or {}
    def classify(self,crop,candidate_labels=None,top_k=5):
        labels=candidate_labels or list(self.mapping)
        pairs=[(x,float(self.mapping.get(x,0.0))) for x in labels]
        return sorted(pairs,key=lambda x:x[1],reverse=True)[:top_k]
