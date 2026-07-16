
A functional Python project comparing OpenCV Haar cascades and pretrained PyTorch MTCNN on a webcam and annotated images.: all metrics must be collected on your machine and dataset.

## Setup in Windows Powershell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you already have a compatible Python environment, use `python -m venv .venv` instead. Initial MTCNN execution may download pretrained weights and requires internet access. For GPU inference, install a matching CUDA-compatible torch build; otherwise CPU is supported.

## Webcam 

```powershell
python webcam.py --model haar
python webcam.py --model torch --device auto
```

Press `q` to quit. Outputs per-frame measurements in `results/webcam.csv`. The requested 1280×720 capture and 30 FPS are **not guaranteed by hardware**; the CSV records actual frame width, height and observed loop FPS. Inference latency excludes webcam acquisition and drawing time, while loop FPS includes both.

## Labeled image evaluation

Put images under `data/images/` and create `data/annotations.csv` with one row **per face** (pixel-coordinate bounding boxes):

```csv
image,x1,y1,x2,y2
frame001.jpg,110,65,310,310
frame002.jpg,90,60,280,300
```

Then run:

```powershell
python evaluate.py --model haar --images data/images --annotations data/annotations.csv --output results/haar_eval.csv
python evaluate.py --model torch --images data/images --annotations data/annotations.csv --output results/torch_eval.csv
```

The evaluator counts true positives using greedy IoU ≥ 0.5 matching, false positives, and false negatives. It reports precision, recall, F1, and per-image inference latency. This is **not classification accuracy**. For a fair comparison, use identical images and annotation rules for each model. Collect 500 annotated frames across lighting conditions if that is the target claim; the script does not manufacture data or annotations.

## Experiment on your local machine

- Collect labeled frames across daylight, dim light, backlighting, and different orientations. Record counts for each condition.
- Run `--neighbors 3` and `--neighbors 5` for Haar and `--threshold 0.75` / `--threshold 0.90` for MTCNN; reserve a held-out set for final metrics.
- Discard first several inference measurements when reporting warmed-up latency, if using GPU; document the exclusion consistently.
- Do not describe pretrained MTCNN evaluation as model training or fine-tuning. Training is a separate extension.

## Repository history

This staged history was reconstructed from an existing project on October 22, 2026. The July 1–16 commit timestamps were assigned retrospectively; they are not contemporaneous development or deployment records.
