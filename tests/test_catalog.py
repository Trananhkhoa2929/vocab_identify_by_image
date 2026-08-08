from pathlib import Path
from visual_vocab.catalog import ObjectCatalog

def test_catalog_translation_and_candidates():
    c=ObjectCatalog(Path(__file__).resolve().parents[1]/'data/vocab/object_catalog.json'); assert c.translation('bottle','vi')=='chai'; assert 'smartphone' in c.candidate_labels('remote control')
