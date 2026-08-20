# Visual Vocabulary Learning Assistant

A photo-to-vocabulary learning system focused on **Computer Vision first**:

1. validate image quality,
2. open-vocabulary object detection with YOLO-World,
3. selectively verify only difficult/ambiguous objects with OpenCLIP,
4. merge final labels,
5. generate bilingual vocabulary notes + IPA-like phonemization + pronunciation audio,
6. show an annotated visual result.

## Main idea
The research/engineering contribution is **Adaptive Object Verification**. A heavy verifier is not run on every detection. A difficulty score based on detector confidence, object size, and ambiguity decides which crops need a second look.

## Quick run
```bash
sudo apt update && sudo apt install -y espeak-ng
python3 -m venv .venv
source .venv/bin/activate
# Install a ROCm PyTorch build first on AMD; then:
pip install -e '.[dev]'
python scripts/preflight.py
python scripts/run_ui.py --device auto
```
Open `http://127.0.0.1:7860`.

## CLI
```bash
python scripts/infer_image.py photo.jpg --native vi --target en --device auto --mode adaptive
```
Outputs: `results/annotated.jpg` + `results/last_result.json`.

## Ablation / speed benchmark
```bash
python scripts/benchmark_pipeline.py ./your_test_images --device auto --modes off adaptive all
```
Compare detector-only vs adaptive verification vs verify-all.

Read `FEATURE_REFERENCE_MAP.md` for the exact open-source references used to design each subsystem.

## Accuracy + speed evaluation on your own labeled photos
Create `labels.json`:
```json
[
  {"file":"desk1.jpg","objects":[
    {"label":"laptop","bbox":[100,80,500,390]},
    {"label":"mouse","bbox":[530,300,620,370]}
  ]}
]
```
Then run the same images through detector-only, adaptive and verify-all:
```bash
python scripts/evaluate_labeled.py labels.json --image-root ./images --device auto
```
Output: `results/accuracy_speed_ablation.json` with precision, recall, F1, label accuracy, FPS and mean number of verified crops.
