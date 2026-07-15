from __future__ import annotations
from io import BytesIO
from PIL import Image
from fastapi import FastAPI,UploadFile,File,Query
from .factory import build_pipeline

def create_app(config=None,device='auto'):
    app=FastAPI(title='Visual Vocabulary Assistant'); pipe=build_pipeline(config,device=device)
    @app.get('/health')
    def health(): return {'ok':True}
    @app.post('/analyze')
    async def analyze(file:UploadFile=File(...),native_lang:str=Query('vi'),target_lang:str=Query('en')):
        image=Image.open(BytesIO(await file.read())).convert('RGB'); r=pipe.analyze(image,native_lang,target_lang); r.pop('annotated',None); return r
    return app
