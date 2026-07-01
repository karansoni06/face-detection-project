# Real-Time Face Detection

A functional Python project comparing OpenCV Haar cascades and pretrained PyTorch MTCNN on a webcam and annotated images.: all metrics must be collected on your machine and dataset.

## Setup in Windows Powershell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you already have a compatible Python environment, use `python -m venv .venv` instead. Initial MTCNN execution may download pretrained weights and requires internet access. For GPU inference, install a matching CUDA-compatible torch build; otherwise CPU is supported.

## Repository history

This staged history was reconstructed from an existing project on October 22, 2026. The July 1–16 commit timestamps were assigned retrospectively; they are not contemporaneous development or deployment records.
