from visual_vocab.schemas import Detection
from visual_vocab.fusion import difficulty_score,fuse_detection

def test_small_low_conf_is_difficult():
    d=Detection('phone',(1,1,20,20),0.3); assert difficulty_score(d,(1000,1000),ambiguous=True)>0.5
def test_clip_can_switch_label_when_clear():
    d=Detection('remote control',(0,0,100,100),0.45); fuse_detection(d,[('smartphone',0.82),('remote control',0.20)]); assert d.label=='smartphone' and d.verified
