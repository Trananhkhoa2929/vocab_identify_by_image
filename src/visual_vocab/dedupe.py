from __future__ import annotations
from .schemas import Detection

def iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1=max(ax1,bx1); iy1=max(ay1,by1); ix2=min(ax2,bx2); iy2=min(ay2,by2)
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    aa=max(0,ax2-ax1)*max(0,ay2-ay1); bb=max(0,bx2-bx1)*max(0,by2-by1)
    return inter/max(1e-9,aa+bb-inter)

def deduplicate(dets:list[Detection], threshold=0.72)->list[Detection]:
    kept=[]
    for d in sorted(dets,key=lambda x:x.score(),reverse=True):
        if any(iou(d.bbox,k.bbox)>=threshold and d.label==k.label for k in kept): continue
        kept.append(d)
    return kept
