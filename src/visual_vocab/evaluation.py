from __future__ import annotations
from dataclasses import dataclass
from .dedupe import iou

@dataclass
class EvalCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    localized: int = 0
    localized_correct_label: int = 0

    def metrics(self):
        p = self.tp / max(1, self.tp + self.fp)
        r = self.tp / max(1, self.tp + self.fn)
        f1 = 2*p*r/max(1e-12,p+r)
        cls = self.localized_correct_label/max(1,self.localized)
        return {'precision':p,'recall':r,'f1':f1,'label_accuracy_on_localized':cls,'tp':self.tp,'fp':self.fp,'fn':self.fn}

def evaluate_image(predictions, ground_truth, iou_threshold=0.5):
    """Greedy one-to-one matching. A TP requires IoU >= threshold AND correct canonical label."""
    counts=EvalCounts(); used=set()
    preds=sorted(predictions,key=lambda d:float(d.get('final_conf') or d.get('detector_conf') or 0),reverse=True)
    for p in preds:
        best_i=None; best_iou=0.0
        for i,g in enumerate(ground_truth):
            if i in used: continue
            v=iou(tuple(p['bbox']),tuple(g['bbox']))
            if v>best_iou: best_i,best_iou=i,v
        if best_i is not None and best_iou>=iou_threshold:
            used.add(best_i); counts.localized += 1
            if p['label']==ground_truth[best_i]['label']:
                counts.tp += 1; counts.localized_correct_label += 1
            else:
                counts.fp += 1; counts.fn += 1
        else:
            counts.fp += 1
    counts.fn += len(ground_truth)-len(used)
    return counts

def merge_counts(dst,src):
    for k in ('tp','fp','fn','localized','localized_correct_label'):
        setattr(dst,k,getattr(dst,k)+getattr(src,k))
    return dst
