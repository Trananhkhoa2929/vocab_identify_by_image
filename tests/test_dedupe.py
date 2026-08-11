from visual_vocab.schemas import Detection
from visual_vocab.dedupe import deduplicate

def test_duplicate_boxes_removed():
    a=Detection('cup',(0,0,100,100),0.9,final_conf=0.9); b=Detection('cup',(2,2,99,99),0.8,final_conf=0.8); assert len(deduplicate([a,b]))==1
