from __future__ import annotations
LANG_TO_ESPEAK={'en':'en-us','vi':'vi','zh':'cmn','ja':'ja','ko':'ko','fr':'fr-fr','de':'de','es':'es'}

def phonemize_words(words:list[str],lang:str)->list[str]:
    try:
        from phonemizer import phonemize
        code=LANG_TO_ESPEAK.get(lang,lang)
        out=phonemize(words,language=code,backend='espeak',strip=True,with_stress=True,preserve_punctuation=True,njobs=1)
        return [str(x).strip() for x in out]
    except Exception:
        return ['' for _ in words]
