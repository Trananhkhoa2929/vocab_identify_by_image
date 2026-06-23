from __future__ import annotations
from time import perf_counter
from PIL import Image
from .schemas import Detection
from .fusion import difficulty_score,fuse_detection
from .dedupe import deduplicate
from .annotate import annotate

class VisualVocabularyPipeline:
    def __init__(self,quality,detector,catalog,vocab_engine,verifier=None,verify_mode='adaptive',verify_threshold=0.36,crop_pad=0.22,min_switch_prob=0.55,min_switch_margin=0.12):
        self.quality=quality; self.detector=detector; self.catalog=catalog; self.vocab_engine=vocab_engine; self.verifier=verifier; self.verify_mode=verify_mode; self.verify_threshold=verify_threshold; self.crop_pad=crop_pad; self.min_switch_prob=min_switch_prob; self.min_switch_margin=min_switch_margin
    def _crop(self,image:Image.Image,box):
        w,h=image.size; x1,y1,x2,y2=box; bw=x2-x1; bh=y2-y1; px=int(bw*self.crop_pad); py=int(bh*self.crop_pad)
        return image.crop((max(0,x1-px),max(0,y1-py),min(w,x2+px),min(h,y2+py)))
    def analyze(self,image:Image.Image,native_lang='vi',target_lang='en'):
        t0=perf_counter(); q=self.quality.validate(image); t1=perf_counter()
        dets=self.detector.detect(image); t2=perf_counter()
        if not dets: q.warnings.append('no supported object was detected')
        for d in dets:
            ambiguous=len(self.catalog.candidate_labels(d.label,global_fallback=False))>1
            d.difficulty=difficulty_score(d,image.size,ambiguous=ambiguous)
            do_verify=self.verifier is not None and (self.verify_mode=='all' or (self.verify_mode=='adaptive' and d.difficulty>=self.verify_threshold))
            if do_verify:
                candidates=self.catalog.candidate_labels(d.label,global_fallback=True)
                preds=self.verifier.classify(self._crop(image,d.bbox),candidates,top_k=5)
                fuse_detection(d,preds,self.min_switch_prob,self.min_switch_margin)
        t3=perf_counter(); dets=deduplicate(dets); notes=self.vocab_engine.build(dets,native_lang,target_lang); annotated=annotate(image,dets); t4=perf_counter()
        return {
            'quality':q.to_dict(), 'detections':[d.to_dict() for d in dets], 'notes':[n.to_dict() for n in notes], 'annotated':annotated,
            'timing_ms':{'quality':(t1-t0)*1000,'detector':(t2-t1)*1000,'verification':(t3-t2)*1000,'notes_and_draw':(t4-t3)*1000,'total':(t4-t0)*1000},
            'verified_count':sum(1 for d in dets if d.verified)
        }
