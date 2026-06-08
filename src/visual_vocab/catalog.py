from __future__ import annotations
from pathlib import Path
import json

class ObjectCatalog:
    def __init__(self, path: str | Path):
        data=json.loads(Path(path).read_text(encoding='utf-8'))
        self.entries=data['objects']; self.by_label={e['label']:e for e in self.entries}
        self.alias_to_label={}
        for e in self.entries:
            self.alias_to_label[e['label']]=e['label']
            for a in e.get('aliases',[]): self.alias_to_label[a]=e['label']
        self.confusion_groups=data.get('confusion_groups',[])
    @property
    def labels(self): return list(self.by_label)
    def canonical(self,label:str)->str: return self.alias_to_label.get(label,label)
    def entry(self,label:str)->dict: return self.by_label.get(self.canonical(label),{"label":label,"translations":{"en":label}})
    def translation(self,label:str,lang:str)->str|None: return self.entry(label).get('translations',{}).get(lang)
    def candidate_labels(self,label:str, global_fallback=True)->list[str]:
        c=self.canonical(label)
        for group in self.confusion_groups:
            if c in group: return list(dict.fromkeys([c,*group]))
        return self.labels if global_fallback else [c]
