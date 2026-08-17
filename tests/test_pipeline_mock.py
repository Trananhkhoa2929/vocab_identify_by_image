import numpy as np
from PIL import Image
from pathlib import Path
from visual_vocab.schemas import Detection
from visual_vocab.quality import ImageQualityValidator
from visual_vocab.catalog import ObjectCatalog
from visual_vocab.detector import MockDetector
from visual_vocab.verifier import MockVerifier
from visual_vocab.translator import CatalogTranslator
from visual_vocab.vocab_engine import VocabularyEngine
from visual_vocab.pipeline import VisualVocabularyPipeline

def test_pipeline_end_to_end_mock():
    root=Path(__file__).resolve().parents[1]; c=ObjectCatalog(root/'data/vocab/object_catalog.json'); v=VocabularyEngine(c,CatalogTranslator(c)); d=MockDetector([Detection('remote control',(20,20,120,120),0.4,final_conf=0.4)]); clip=MockVerifier({'smartphone':0.85,'remote control':0.10}); p=VisualVocabularyPipeline(ImageQualityValidator(100,0,0,255,0),d,c,v,clip,'all',0.0); r=p.analyze(Image.fromarray(np.full((300,300,3),128,dtype=np.uint8)),'vi','en'); assert r['detections'][0]['label']=='smartphone'; assert r['notes'][0]['target_text']=='smartphone'
