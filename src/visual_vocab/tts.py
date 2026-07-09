from __future__ import annotations
from pathlib import Path
import hashlib, subprocess

VOICE_MAP={'en':'en-US-AriaNeural','vi':'vi-VN-HoaiMyNeural','zh':'zh-CN-XiaoxiaoNeural','ja':'ja-JP-NanamiNeural','ko':'ko-KR-SunHiNeural'}
ESPEAK_MAP={'en':'en-us','vi':'vi','zh':'cmn','ja':'ja','ko':'ko'}

class TTSService:
    def __init__(self,cache_dir='audio_cache',mode='auto'):
        self.cache=Path(cache_dir); self.cache.mkdir(parents=True,exist_ok=True); self.mode=mode
    def synthesize(self,text:str,lang='en')->str:
        key=hashlib.sha1(f'{lang}|{text}'.encode()).hexdigest()[:16]
        if self.mode in ('auto','edge'):
            path=self.cache/f'{key}.mp3'
            if path.exists(): return str(path)
            try:
                import edge_tts
                edge_tts.Communicate(text,VOICE_MAP.get(lang,'en-US-AriaNeural')).save_sync(str(path)); return str(path)
            except Exception:
                if self.mode=='edge': raise
        path=self.cache/f'{key}.wav'
        if path.exists(): return str(path)
        subprocess.run(['espeak-ng','-v',ESPEAK_MAP.get(lang,lang),'-w',str(path),text],check=True,capture_output=True)
        return str(path)
