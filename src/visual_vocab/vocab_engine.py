from __future__ import annotations
from collections import Counter
from .schemas import Detection,StudyNote
from .phonetics import phonemize_words

EXAMPLE_TEMPLATES={
'en':'This is a {w}.','vi':'Đây là {w}.','zh':'这是{w}。','ja':'これは{w}です。','ko':'이것은 {w}입니다。','fr':"C'est un/une {w}.",'de':'Das ist ein/eine {w}.','es':'Esto es un/una {w}.'}

class VocabularyEngine:
    def __init__(self,catalog,translator): self.catalog=catalog; self.translator=translator
    def build(self,detections:list[Detection],native_lang='vi',target_lang='en')->list[StudyNote]:
        best={}; counts=Counter()
        for d in detections:
            label=self.catalog.canonical(d.label); counts[label]+=1
            if label not in best or d.score()>best[label].score(): best[label]=d
        labels=list(best); targets=[self.translator.translate(x,target_lang) for x in labels]
        ipas=phonemize_words(targets,target_lang)
        notes=[]
        for label,target,ipa in zip(labels,targets,ipas):
            native=self.translator.translate(label,native_lang)
            ex=EXAMPLE_TEMPLATES.get(target_lang,'This is a {w}.').format(w=target)
            notes.append(StudyNote(label,native,target,ipa,'noun',ex,best[label].score(),counts[label]))
        return notes
