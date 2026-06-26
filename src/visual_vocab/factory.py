from __future__ import annotations
from pathlib import Path
from .config import load_config
from .catalog import ObjectCatalog
from .quality import ImageQualityValidator
from .detector import YOLOWorldDetector
from .verifier import OpenCLIPVerifier
from .translator import CatalogTranslator
from .vocab_engine import VocabularyEngine
from .pipeline import VisualVocabularyPipeline

def build_pipeline(config_path=None,device='auto',verify_mode=None):
    cfg=load_config(config_path); root=Path(__file__).resolve().parents[2]
    catalog=ObjectCatalog(root/cfg['catalog']['path'])
    quality=ImageQualityValidator(**cfg['quality'])
    detector=YOLOWorldDetector(catalog.labels,device=device,**cfg['detector'])
    verifier=None
    mode=verify_mode or cfg['verification']['mode']
    if mode!='off': verifier=OpenCLIPVerifier(catalog.labels,device=device,**cfg['verification']['model'])
    vocab=VocabularyEngine(catalog,CatalogTranslator(catalog))
    return VisualVocabularyPipeline(
        quality,detector,catalog,vocab,verifier,mode,
        cfg['verification']['difficulty_threshold'],cfg['verification']['crop_pad'],
        cfg['verification'].get('min_switch_prob',0.55),
        cfg['verification'].get('min_switch_margin',0.12),
    )
