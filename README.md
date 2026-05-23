# Retail Robustness Studio

Academic demo for the MobileNetV3Small vs TinyMobileViT Fruits-360 project.

## Purpose

This app showcases the exported `.pkl` artifact as a usable research system:

- upload a produce image
- compare MobileNetV3Small and TinyMobileViT predictions
- apply Gaussian noise, motion blur, or occlusion
- inspect clean and corrupted robustness metrics
- benchmark model latency and model size for edge-readiness discussion

## Run

Use the CUDA Python environment:

```powershell
& 'C:\Users\naosh\Documents\Codes\ebook2audiobook\python_env\python.exe' app.py
```

Then open:

```text
http://127.0.0.1:7860
```

## Smoke Test

```powershell
& 'C:\Users\naosh\Documents\Codes\ebook2audiobook\python_env\python.exe' smoke_test.py
```

## Artifact

Default artifact path:

```text
D:\Codes 2\Project\fruits360_mobilenetv3_mobilevit_artifact.pkl
```

The artifact stores model state dictionaries, class labels, normalization, clean metrics, robustness metrics, and training metadata.

## Defense Framing

This is an academic demo, not a production checkout system. The strongest defense claim:

> TinyMobileViT is best for clean images and occlusion, while MobileNetV3Small is more stable under Gaussian noise and motion blur. Model choice depends on expected retail failure mode.
