from __future__ import annotations
import cv2, numpy as np
from PIL import Image
from .schemas import QualityReport

class ImageQualityValidator:
    def __init__(self, min_side=320, blur_min=45.0, brightness_min=25.0, brightness_max=235.0, contrast_min=18.0):
        self.min_side=min_side; self.blur_min=blur_min; self.brightness_min=brightness_min; self.brightness_max=brightness_max; self.contrast_min=contrast_min
    def validate(self, image: Image.Image) -> QualityReport:
        rgb=np.asarray(image.convert('RGB')); h,w=rgb.shape[:2]
        gray=cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        blur=float(cv2.Laplacian(gray,cv2.CV_64F).var())
        brightness=float(gray.mean()); contrast=float(gray.std())
        dark=float((gray<15).mean()); bright=float((gray>245).mean())
        warnings=[]
        if min(w,h)<self.min_side: warnings.append(f'image resolution is low ({w}x{h})')
        if blur<self.blur_min: warnings.append('image appears blurry')
        if brightness<self.brightness_min or dark>0.60: warnings.append('image is too dark')
        if brightness>self.brightness_max or bright>0.60: warnings.append('image is overexposed')
        if contrast<self.contrast_min: warnings.append('image has very low contrast')
        return QualityReport(not warnings,w,h,blur,brightness,contrast,dark,bright,warnings)
