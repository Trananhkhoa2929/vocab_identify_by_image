from __future__ import annotations

class CatalogTranslator:
    def __init__(self,catalog): self.catalog=catalog
    def translate(self,label:str,lang:str)->str:
        return self.catalog.translation(label,lang) or label.replace('_',' ')
