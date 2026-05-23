from __future__ import annotations

import io
import math
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F


APP_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT = Path(r"D:\Codes 2\Project\fruits360_mobilenetv3_mobilevit_artifact.pkl")
ARTIFACT_PATH = DEFAULT_ARTIFACT if DEFAULT_ARTIFACT.exists() else APP_DIR.parent / "fruits360_mobilenetv3_mobilevit_artifact.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, groups=1, act="silu"):
        padding = (kernel - 1) // 2
        if act == "hswish":
            activation = nn.Hardswish(inplace=True)
        elif act == "relu":
            activation = nn.ReLU(inplace=True)
        elif act == "none":
            activation = nn.Identity()
        else:
            activation = nn.SiLU(inplace=True)
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            activation,
        )


class SqueezeExcite(nn.Module):
    def __init__(self, channels, squeeze_factor=4):
        super().__init__()
        squeeze = max(8, channels // squeeze_factor)
        self.fc1 = nn.Conv2d(channels, squeeze, 1)
        self.fc2 = nn.Conv2d(squeeze, channels, 1)

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.relu(self.fc1(scale), inplace=True)
        scale = F.hardsigmoid(self.fc2(scale), inplace=True)
        return x * scale


class MobileNetV3Block(nn.Module):
    def __init__(self, in_ch, exp_ch, out_ch, kernel, stride, use_se, act):
        super().__init__()
        layers = []
        if exp_ch != in_ch:
            layers.append(ConvBNAct(in_ch, exp_ch, kernel=1, act=act))
        layers.append(ConvBNAct(exp_ch, exp_ch, kernel=kernel, stride=stride, groups=exp_ch, act=act))
        if use_se:
            layers.append(SqueezeExcite(exp_ch))
        layers.append(ConvBNAct(exp_ch, out_ch, kernel=1, act="none"))
        self.block = nn.Sequential(*layers)
        self.use_residual = stride == 1 and in_ch == out_ch

    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_residual else out


class MobileNetV3Small(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.25):
        super().__init__()
        cfgs = [
            (16, 16, 16, 3, 2, True, "relu"),
            (16, 72, 24, 3, 2, False, "relu"),
            (24, 88, 24, 3, 1, False, "relu"),
            (24, 96, 40, 5, 2, True, "hswish"),
            (40, 240, 40, 5, 1, True, "hswish"),
            (40, 240, 40, 5, 1, True, "hswish"),
            (40, 120, 48, 5, 1, True, "hswish"),
            (48, 144, 48, 5, 1, True, "hswish"),
            (48, 288, 96, 5, 2, True, "hswish"),
            (96, 576, 96, 5, 1, True, "hswish"),
            (96, 576, 96, 5, 1, True, "hswish"),
        ]
        layers = [ConvBNAct(3, 16, kernel=3, stride=2, act="hswish")]
        for args in cfgs:
            layers.append(MobileNetV3Block(*args))
        layers.append(ConvBNAct(96, 576, kernel=1, act="hswish"))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(576, 768),
            nn.Hardswish(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(768, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class MV2Block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, expansion=2):
        super().__init__()
        hidden = int(in_ch * expansion)
        self.use_residual = stride == 1 and in_ch == out_ch
        self.block = nn.Sequential(
            ConvBNAct(in_ch, hidden, kernel=1, act="silu"),
            ConvBNAct(hidden, hidden, kernel=3, stride=stride, groups=hidden, act="silu"),
            ConvBNAct(hidden, out_ch, kernel=1, act="none"),
        )

    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_residual else out


class TinyTransformer(nn.Module):
    def __init__(self, dim, depth=2, heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

    def forward(self, x):
        return self.encoder(x)


class MobileViTBlock(nn.Module):
    def __init__(self, in_ch, transformer_dim, depth=2, heads=4, dropout=0.1):
        super().__init__()
        self.local = nn.Sequential(
            ConvBNAct(in_ch, in_ch, kernel=3, act="silu"),
            ConvBNAct(in_ch, transformer_dim, kernel=1, act="silu"),
        )
        self.transformer = TinyTransformer(transformer_dim, depth=depth, heads=heads, dropout=dropout)
        self.project = ConvBNAct(transformer_dim, in_ch, kernel=1, act="silu")
        self.fuse = ConvBNAct(in_ch * 2, in_ch, kernel=3, act="silu")

    def forward(self, x):
        residual = x
        y = self.local(x)
        b, c, h, w = y.shape
        tokens = y.flatten(2).transpose(1, 2)
        tokens = self.transformer(tokens)
        y = tokens.transpose(1, 2).reshape(b, c, h, w)
        y = self.project(y)
        return self.fuse(torch.cat([residual, y], dim=1))


class TinyMobileViT(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.20):
        super().__init__()
        self.stem = ConvBNAct(3, 16, kernel=3, stride=2, act="silu")
        self.stage1 = nn.Sequential(MV2Block(16, 32, stride=1, expansion=2), MV2Block(32, 48, stride=2, expansion=2))
        self.mvit1 = MobileViTBlock(48, transformer_dim=64, depth=2, heads=4, dropout=dropout)
        self.stage2 = MV2Block(48, 80, stride=2, expansion=2)
        self.mvit2 = MobileViTBlock(80, transformer_dim=96, depth=2, heads=4, dropout=dropout)
        self.stage3 = nn.Sequential(MV2Block(80, 128, stride=2, expansion=2), ConvBNAct(128, 192, kernel=1, act="silu"))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(192),
            nn.Dropout(dropout),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.mvit1(x)
        x = self.stage2(x)
        x = self.mvit2(x)
        x = self.stage3(x)
        return self.head(x)


def load_artifact() -> dict:
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(f"Artifact not found: {ARTIFACT_PATH}")
    with ARTIFACT_PATH.open("rb") as f:
        return pickle.load(f)


ARTIFACT = load_artifact()
NUM_CLASSES = int(ARTIFACT["num_classes"])
IMAGE_SIZE = int(ARTIFACT["hyperparameters"]["image_size"])
MEAN = torch.tensor(ARTIFACT["normalization"]["mean"], dtype=torch.float32).view(3, 1, 1)
STD = torch.tensor(ARTIFACT["normalization"]["std"], dtype=torch.float32).view(3, 1, 1)
IDX_TO_CLASS = {int(k): v for k, v in ARTIFACT["idx_to_class"].items()}


def load_models(device: torch.device = DEVICE) -> Dict[str, nn.Module]:
    factories = {
        "MobileNetV3Small": lambda: MobileNetV3Small(NUM_CLASSES, dropout=0.25),
        "TinyMobileViT": lambda: TinyMobileViT(NUM_CLASSES, dropout=0.20),
    }
    models = {}
    for name, factory in factories.items():
        model = factory()
        state = ARTIFACT["models"][name]["state_dict"]
        model.load_state_dict(state, strict=True)
        model.to(device)
        model.eval()
        models[name] = model
    return models


MODELS = load_models(DEVICE)


def motion_blur_image(im: Image.Image, level: int) -> Image.Image:
    if level <= 0:
        return im
    radius = {1: 2, 2: 4, 3: 6}.get(int(level), 4)
    arr = np.asarray(im.convert("RGB"), dtype=np.float32)
    h, w, _ = arr.shape
    padded = np.pad(arr, ((0, 0), (radius, radius), (0, 0)), mode="edge")
    acc = np.zeros_like(arr)
    for offset in range(-radius, radius + 1):
        start = radius + offset
        acc += padded[:, start : start + w, :]
    return Image.fromarray(np.clip(acc / (2 * radius + 1), 0, 255).astype(np.uint8))


def gaussian_noise_image(im: Image.Image, level: int, seed: int = 42) -> Image.Image:
    if level <= 0:
        return im
    arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    sigma = {1: 0.05, 2: 0.10, 3: 0.18}.get(int(level), 0.10)
    rng = np.random.default_rng(seed + int(level))
    noise = rng.normal(0.0, sigma, arr.shape)
    return Image.fromarray((np.clip(arr + noise, 0.0, 1.0) * 255).astype(np.uint8))


def occlusion_image(im: Image.Image, level: int, seed: int = 42) -> Image.Image:
    if level <= 0:
        return im
    im = im.convert("RGB").copy()
    draw = ImageDraw.Draw(im)
    w, h = im.size
    frac = {1: 0.18, 2: 0.30, 3: 0.42}.get(int(level), 0.30)
    box_w = int(w * frac)
    box_h = int(h * frac)
    rng = np.random.default_rng(seed + int(level))
    x0 = int(rng.integers(0, max(1, w - box_w + 1)))
    y0 = int(rng.integers(0, max(1, h - box_h + 1)))
    draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=(35, 35, 35))
    return im


def apply_corruption(im: Image.Image, corruption: str, level: int) -> Image.Image:
    corruption = (corruption or "Clean").lower().replace(" ", "_")
    level = int(level)
    if corruption in {"clean", "none"} or level == 0:
        return im.convert("RGB")
    if corruption == "gaussian_noise":
        return gaussian_noise_image(im, level)
    if corruption == "motion_blur":
        return motion_blur_image(im, level)
    if corruption == "occlusion":
        return occlusion_image(im, level)
    return im.convert("RGB")


def preprocess(im: Image.Image) -> torch.Tensor:
    im = ImageOps.fit(im.convert("RGB"), (IMAGE_SIZE, IMAGE_SIZE), method=Image.Resampling.BICUBIC)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr.transpose(2, 0, 1))
    return ((x - MEAN) / STD).unsqueeze(0)


@torch.no_grad()
def predict_one(model: nn.Module, image: Image.Image, top_k: int = 5) -> pd.DataFrame:
    x = preprocess(image).to(next(model.parameters()).device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu()
    values, indices = torch.topk(probs, k=min(top_k, len(probs)))
    rows = []
    for rank, (idx, prob) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
        rows.append({"rank": rank, "class": IDX_TO_CLASS[int(idx)], "confidence": round(float(prob), 4)})
    return pd.DataFrame(rows)


def research_winner(corruption: str, level: int) -> str:
    df = pd.DataFrame(ARTIFACT["robustness_metrics"])
    key = corruption.lower().replace(" ", "_")
    level = int(level)
    if key in {"none", "clean"} or level == 0:
        key, level = "clean", 0
    match = df[(df["corruption"] == key) & (df["level"].astype(int) == level)]
    if match.empty:
        return "No matching study row."
    row = match.sort_values("accuracy", ascending=False).iloc[0]
    return f"{row['model']} won this condition in the study: accuracy {row['accuracy']:.2%}, macro F1 {row['f1_macro']:.2%}."


def predict_both(image: Image.Image, corruption: str, level: int, top_k: int = 5):
    if image is None:
        raise gr.Error("Upload an image first.")
    corrupted = apply_corruption(image, corruption, level)
    results = {name: predict_one(model, corrupted, top_k) for name, model in MODELS.items()}
    top_mobile = results["MobileNetV3Small"].iloc[0]
    top_vit = results["TinyMobileViT"].iloc[0]
    agree = top_mobile["class"] == top_vit["class"]
    if agree:
        verdict = f"Agreement: both models predict **{top_mobile['class']}**."
    else:
        chosen = top_mobile if top_mobile["confidence"] >= top_vit["confidence"] else top_vit
        verdict = (
            f"Disagreement: MobileNetV3Small predicts **{top_mobile['class']}** "
            f"({top_mobile['confidence']:.2%}); TinyMobileViT predicts **{top_vit['class']}** "
            f"({top_vit['confidence']:.2%}). Higher-confidence output: **{chosen['class']}**."
        )
    summary = (
        f"### Prediction Summary\n\n"
        f"{verdict}\n\n"
        f"**Study context:** {research_winner(corruption, level)}\n\n"
        f"Note: Fruits-360 uses centered produce on clean backgrounds, so real retail photos may be out-of-distribution."
    )
    return corrupted, summary, results["MobileNetV3Small"], results["TinyMobileViT"]


def clean_metrics_table() -> pd.DataFrame:
    df = pd.DataFrame(ARTIFACT["clean_test_metrics"])
    return df[["model", "accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovr_macro"]].sort_values("f1_macro", ascending=False)


def robustness_table() -> pd.DataFrame:
    df = pd.DataFrame(ARTIFACT["robustness_metrics"])
    cols = ["model", "corruption", "level", "accuracy", "f1_macro", "accuracy_drop", "f1_macro_drop"]
    return df[cols].sort_values(["corruption", "level", "model"])


def degradation_table() -> pd.DataFrame:
    df = robustness_table()
    return (
        df[df["corruption"] != "clean"]
        .groupby(["model", "corruption"])[["accuracy_drop", "f1_macro_drop"]]
        .agg(["mean", "max"])
        .round(4)
        .reset_index()
    )


def state_dict_mb(model_name: str) -> float:
    state = ARTIFACT["models"][model_name]["state_dict"]
    total = sum(v.numel() * v.element_size() for v in state.values() if torch.is_tensor(v))
    return total / (1024 * 1024)


@torch.no_grad()
def benchmark_models(repeats: int = 50, batch_size: int = 1) -> pd.DataFrame:
    repeats = int(max(5, min(repeats, 300)))
    batch_size = int(max(1, min(batch_size, 64)))
    rows = []
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    for device in devices:
        local_models = load_models(device)
        x = torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
        for name, model in local_models.items():
            for _ in range(8):
                _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(repeats):
                _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            ms_per_image = elapsed * 1000 / (repeats * batch_size)
            rows.append(
                {
                    "model": name,
                    "device": device.type,
                    "params": ARTIFACT["models"][name]["params"],
                    "state_dict_mb": round(state_dict_mb(name), 2),
                    "batch_size": batch_size,
                    "repeats": repeats,
                    "ms_per_image": round(ms_per_image, 3),
                    "images_per_second": round(1000 / ms_per_image, 2) if ms_per_image else math.inf,
                }
            )
    return pd.DataFrame(rows).sort_values(["device", "ms_per_image"])


def benchmark_summary(repeats: int, batch_size: int):
    df = benchmark_models(repeats, batch_size)
    gpu = df[df["device"] == "cuda"]
    if not gpu.empty:
        fastest = gpu.sort_values("ms_per_image").iloc[0]
    else:
        fastest = df.sort_values("ms_per_image").iloc[0]
    text = (
        f"Fastest measured model on available hardware: **{fastest['model']}** "
        f"on **{fastest['device']}** at **{fastest['ms_per_image']} ms/image**. "
        "Use this table in the report to answer edge-readiness critique."
    )
    return df, text


def dashboard_markdown() -> str:
    clean = clean_metrics_table()
    best_clean = clean.iloc[0]
    return f"""
# Retail Robustness Studio

Academic demo for **MobileNetV3Small vs TinyMobileViT** on Fruits-360 domain shift.

**Artifact:** `{ARTIFACT_PATH}`

**Runtime:** `{DEVICE}` {"(" + torch.cuda.get_device_name(0) + ")" if torch.cuda.is_available() else ""}

## Main Finding

- Best clean model: **{best_clean["model"]}**
- Clean accuracy: **{best_clean["accuracy"]:.2%}**
- Clean macro F1: **{best_clean["f1_macro"]:.2%}**
- Occlusion winner: **TinyMobileViT**
- Gaussian noise winner: **MobileNetV3Small**
- Motion blur winner: **MobileNetV3Small**

This demo is meant for the project defense: it shows prediction behavior, corruption sensitivity, and edge-style latency.
"""


def build_demo():
    with gr.Blocks(title="Retail Robustness Studio") as demo:
        gr.Markdown(dashboard_markdown())
        with gr.Tab("Dashboard"):
            gr.Markdown("## Clean Test Metrics")
            gr.Dataframe(clean_metrics_table(), interactive=False)
            gr.Markdown("## Robustness Metrics")
            gr.Dataframe(robustness_table(), interactive=False)
            gr.Markdown("## Degradation Summary")
            gr.Dataframe(degradation_table(), interactive=False)

        with gr.Tab("Upload + Corruption Lab"):
            with gr.Row():
                with gr.Column(scale=1):
                    image = gr.Image(type="pil", label="Upload produce image")
                    corruption = gr.Dropdown(
                        ["Clean", "Gaussian Noise", "Motion Blur", "Occlusion"],
                        value="Clean",
                        label="Domain shift",
                    )
                    level = gr.Slider(0, 3, value=0, step=1, label="Corruption intensity")
                    top_k = gr.Slider(1, 10, value=5, step=1, label="Top K predictions")
                    run = gr.Button("Run Comparison", variant="primary")
                with gr.Column(scale=1):
                    corrupted = gr.Image(type="pil", label="Model input preview")
                    summary = gr.Markdown()
            with gr.Row():
                mobile_out = gr.Dataframe(label="MobileNetV3Small top predictions", interactive=False)
                vit_out = gr.Dataframe(label="TinyMobileViT top predictions", interactive=False)
            run.click(predict_both, inputs=[image, corruption, level, top_k], outputs=[corrupted, summary, mobile_out, vit_out])

        with gr.Tab("Edge Benchmark"):
            gr.Markdown("Run local latency test. Use GPU row for live-checkout demo claim; CPU row for fallback claim.")
            with gr.Row():
                repeats = gr.Slider(10, 300, value=50, step=10, label="Repeats")
                batch_size = gr.Slider(1, 64, value=1, step=1, label="Batch size")
            bench_btn = gr.Button("Benchmark Models", variant="primary")
            bench_table = gr.Dataframe(label="Latency and size table", interactive=False)
            bench_text = gr.Markdown()
            bench_btn.click(benchmark_summary, inputs=[repeats, batch_size], outputs=[bench_table, bench_text])

        with gr.Tab("Report Text"):
            gr.Markdown(
                """
## Prototype System Text for Report

Retail Robustness Studio is a local academic demonstration system developed from the exported `.pkl` artifact. It allows users to upload produce images, apply simulated domain-shift corruptions, compare MobileNetV3Small and TinyMobileViT predictions, and measure inference latency on available hardware. This system connects the experimental findings to the smart-retail setting by showing that model choice depends on the expected deployment failure mode: TinyMobileViT is stronger for clean images and partial occlusion, while MobileNetV3Small is more stable under Gaussian noise and motion blur.
"""
            )
    return demo


if __name__ == "__main__":
    build_demo().launch(server_name="127.0.0.1", server_port=7860, show_error=True)
