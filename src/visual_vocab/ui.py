from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from PIL import Image
from .factory import build_pipeline
from .tts import TTSService

LANGS = {'Vietnamese':'vi','English':'en','Chinese':'zh','Japanese':'ja','Korean':'ko'}

def build_app(config=None, device='auto'):
    import gradio as gr
    pipe = build_pipeline(config, device=device)
    tts = TTSService(Path(__file__).resolve().parents[2] / 'audio_cache')
    def decode_state(raw):
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {}
        return raw if isinstance(raw, dict) else {}

    def analyze(image, native_name, target_name, mode):
        if image is None:
            return None, {}, pd.DataFrame(), '', {}, '{}'
        pipe.verify_mode = {'Adaptive':'adaptive','Verify all':'all','Detector only':'off'}[mode]
        r = pipe.analyze(Image.fromarray(image), LANGS[native_name], LANGS[target_name])
        rows = []
        for i, d in enumerate(r['detections'], 1):
            rows.append({
                '#': i,
                'object': d['label'],
                'confidence': round(d['final_conf'] or d['detector_conf'], 3),
                'verified': d['verified'],
                'difficulty': round(d['difficulty'], 3),
            })
        md = []
        for i, n in enumerate(r['notes'], 1):
            ipa = f" · /{n['target_ipa']}/" if n['target_ipa'] else ''
            md.append(
                f"### {i}. {n['target_text']}\n"
                f"**{n['native_text']}**{ipa}  \n"
                f"{n['example']}  \n"
                f"Confidence: {n['confidence']:.2f}"
            )
        state_data = {
            'result': {k:v for k,v in r.items() if k != 'annotated'},
            'notes': r['notes'],
            'target_lang': LANGS[target_name],
        }
        return r['annotated'], r['quality'], pd.DataFrame(rows), '\n\n'.join(md), r['timing_ms'], json.dumps(state_data, ensure_ascii=False)

    def choose_from_click(state_data, evt: gr.SelectData | None = None):
        try:
            state_data = decode_state(state_data)
            x, y = evt.index
            for i, d in enumerate(state_data['result']['detections'], 1):
                x1, y1, x2, y2 = d['bbox']
                if x1 <= x <= x2 and y1 <= y <= y2:
                    return i
        except Exception:
            pass
        return 1

    def export_notes(state_data):
        try:
            import tempfile
            state_data = decode_state(state_data)
            path = Path(tempfile.gettempdir()) / 'visual_vocab_study_notes.json'
            path.write_text(json.dumps(state_data['notes'], ensure_ascii=False, indent=2), encoding='utf-8')
            return str(path)
        except Exception:
            return None

    def speak(index, state_data):
        try:
            state_data = decode_state(state_data)
            n = state_data['notes'][int(index)-1]
            return tts.synthesize(n['target_text'], state_data['target_lang'])
        except Exception:
            return None

    with gr.Blocks(title='Visual Vocabulary Learning Assistant') as demo:
        # Keep this component inside Blocks so Gradio registers it before the
        # event handlers refer to its component ID.  State itself is avoided
        # because Gradio 6.17's State post-processing is broken in this venv.
        state = gr.Textbox(value='{}', visible=False, label='Internal state')
        gr.Markdown('# Visual Vocabulary Learning Assistant\nUpload/take a photo → detect objects → verify difficult objects → create vocabulary notes.')
        with gr.Row():
            image = gr.Image(type='numpy', sources=['upload','webcam'], label='Photo')
            annotated = gr.Image(label='Annotated result')
        with gr.Row():
            native = gr.Dropdown(list(LANGS), value='Vietnamese', label='Native language')
            target = gr.Dropdown(list(LANGS), value='English', label='Learning language')
            mode = gr.Radio(['Adaptive','Verify all','Detector only'], value='Adaptive', label='Verification mode')
        btn = gr.Button('Analyze', variant='primary')
        quality = gr.JSON(label='Image validation')
        timing = gr.JSON(label='Timing')
        table = gr.Dataframe(label='Objects')
        notes = gr.Markdown()
        with gr.Row():
            idx = gr.Number(value=1, precision=0, label='Vocabulary item # (or click its box)')
            speak_btn = gr.Button('Pronounce selected word')
            export_btn = gr.Button('Export study notes')
            audio = gr.Audio(label='Pronunciation')
            download = gr.File(label='Study notes JSON')
        btn.click(analyze, [image,native,target,mode], [annotated,quality,table,notes,timing,state])
        annotated.select(choose_from_click, [state], [idx])
        speak_btn.click(speak, [idx,state], audio)
        export_btn.click(export_notes, [state], download)
    return demo
