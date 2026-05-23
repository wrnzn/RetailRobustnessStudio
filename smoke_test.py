from PIL import Image, ImageDraw

import app


def make_demo_image():
    im = Image.new("RGB", (160, 160), "white")
    draw = ImageDraw.Draw(im)
    draw.ellipse((35, 30, 125, 125), fill=(220, 30, 30), outline=(120, 0, 0), width=3)
    draw.rectangle((78, 20, 88, 40), fill=(80, 120, 30))
    return im


def main():
    image = make_demo_image()
    corrupted, summary, mobile, vit = app.predict_both(image, "Clean", 0, 5)
    assert corrupted.size[0] > 0
    assert len(mobile) == 5
    assert len(vit) == 5
    assert {"rank", "class", "confidence"}.issubset(mobile.columns)
    bench = app.benchmark_models(repeats=5, batch_size=1)
    assert not bench.empty
    assert {"model", "device", "ms_per_image"}.issubset(bench.columns)
    print("SMOKE_OK")
    print(summary.replace("\n", " ")[:400])
    print(bench.to_string(index=False))


if __name__ == "__main__":
    main()
