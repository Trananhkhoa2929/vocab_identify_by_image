import numpy as np
from PIL import Image
from visual_vocab.quality import ImageQualityValidator

def test_dark_image_rejected():
    im=Image.fromarray(np.zeros((480,640,3),dtype=np.uint8)); r=ImageQualityValidator().validate(im); assert not r.ok and any('dark' in x for x in r.warnings)
