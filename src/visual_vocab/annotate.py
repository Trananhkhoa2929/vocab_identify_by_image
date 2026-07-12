from __future__ import annotations
from PIL import Image, ImageDraw, ImageFont
from .schemas import Detection

def annotate(image:Image.Image,detections:list[Detection])->Image.Image:
    im=image.convert('RGB').copy(); draw=ImageDraw.Draw(im); font=ImageFont.load_default()
    for i,d in enumerate(detections,1):
        x1,y1,x2,y2=d.bbox; draw.rectangle((x1,y1,x2,y2),outline='white',width=3)
        txt=f'{i}. {d.label} {d.score():.2f}'
        box=draw.textbbox((x1,y1),txt,font=font); tw=box[2]-box[0]; th=box[3]-box[1]
        ty=max(0,y1-th-6); draw.rectangle((x1,ty,x1+tw+6,ty+th+6),fill='black'); draw.text((x1+3,ty+3),txt,fill='white',font=font)
    return im
