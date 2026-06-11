from __future__ import annotations
import numpy as np
from PIL import Image
from .schemas import Detection
from .device import resolve_device

class YOLOWorldDetector:
    def __init__(self, classes:list[str], model_name='yolov8s-worldv2.pt', device='auto', imgsz=640, conf=0.18, iou=0.55):
        from ultralytics import YOLOWorld
        self.device=resolve_device(device); self.imgsz=imgsz; self.conf=conf; self.iou=iou
        self.model=YOLOWorld(model_name); self.model.set_classes(classes); self.classes=classes
    def detect(self, image:Image.Image)->list[Detection]:
        results=self.model.predict(source=np.asarray(image.convert('RGB')), imgsz=self.imgsz, conf=self.conf, iou=self.iou, device=self.device, verbose=False)
        out=[]
        if not results: return out
        r=results[0]; names=r.names
        if r.boxes is None: return out
        xyxy=r.boxes.xyxy.detach().cpu().numpy(); confs=r.boxes.conf.detach().cpu().numpy(); clss=r.boxes.cls.detach().cpu().numpy().astype(int)
        for box,c,ci in zip(xyxy,confs,clss):
            x1,y1,x2,y2=[int(round(x)) for x in box.tolist()]
            label=str(names[int(ci)])
            out.append(Detection(label,(x1,y1,x2,y2),float(c),detector_label=label,final_conf=float(c)))
        return out

class MockDetector:
    def __init__(self,detections:list[Detection]|None=None): self.detections=detections or []
    def detect(self,image): return [Detection(**d.to_dict()) if isinstance(d,Detection) else d for d in self.detections]
