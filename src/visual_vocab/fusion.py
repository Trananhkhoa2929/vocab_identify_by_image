from __future__ import annotations
import math
from .schemas import Detection

def bbox_area_ratio(box,image_size):
    w,h=image_size; x1,y1,x2,y2=box; return max(0,x2-x1)*max(0,y2-y1)/max(1,w*h)

def difficulty_score(det:Detection,image_size, small_ratio=0.012, target_conf=0.72, ambiguity_bonus=0.16, ambiguous=False):
    area=bbox_area_ratio(det.bbox,image_size)
    small=max(0.0,min(1.0,(small_ratio-area)/small_ratio))
    uncertainty=max(0.0,min(1.0,(target_conf-det.detector_conf)/target_conf))
    d=0.55*uncertainty+0.35*small+(ambiguity_bonus if ambiguous else 0.0)
    return float(max(0,min(1,d)))

def fuse_detection(det:Detection, verifier:list[tuple[str,float]], min_switch_prob=0.55, min_switch_margin=0.12):
    if det.detector_label is None:
        det.detector_label = det.label
    if not verifier:
        det.final_conf=det.detector_conf; return det
    top_label,top_prob=verifier[0]; second=verifier[1][1] if len(verifier)>1 else 0.0; margin=top_prob-second
    same=top_label==det.label
    if same:
        final=0.58*det.detector_conf+0.42*top_prob
        chosen=det.label
    elif top_prob>=min_switch_prob and margin>=min_switch_margin:
        chosen=top_label; final=0.38*det.detector_conf+0.62*top_prob
    else:
        chosen=det.label; final=0.72*det.detector_conf+0.28*top_prob
    det.verified=True; det.verifier_label=top_label; det.verifier_prob=float(top_prob); det.verifier_margin=float(margin); det.label=chosen; det.final_conf=float(max(0,min(1,final))); det.source='detector+clip'
    return det
