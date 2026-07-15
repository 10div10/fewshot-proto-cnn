"""
Generates simple synthetic product-like images for testing the few-shot
pipeline end-to-end without needing external image downloads (useful in
network-restricted environments, and for fast, deterministic CI/local tests).

Each "class" is a distinct shape+color combination so the CNN embeddings are
genuinely separable -- this is a stand-in for real product photos.
"""
import os

from PIL import Image, ImageDraw

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_data")

CLASSES = {
    "red_bottle": {"shape": "rectangle", "color": (200, 30, 30)},
    "blue_mug": {"shape": "ellipse", "color": (30, 60, 200)},
    "green_box": {"shape": "rectangle", "color": (30, 160, 60)},
}

IMAGES_PER_CLASS = 8


def _draw_variant(shape: str, color: tuple, seed: int) -> Image.Image:
    img = Image.new("RGB", (224, 224), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)
    # Vary position/size slightly per image so embeddings aren't identical,
    # mimicking the natural variation across real reference photos.
    offset = (seed % 5) * 6
    box = [40 + offset, 40 + offset, 184 - offset, 184 - offset]
    if shape == "rectangle":
        draw.rectangle(box, fill=color)
    else:
        draw.ellipse(box, fill=color)
    return img


def generate():
    os.makedirs(OUT_DIR, exist_ok=True)
    for class_name, spec in CLASSES.items():
        class_dir = os.path.join(OUT_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(IMAGES_PER_CLASS):
            img = _draw_variant(spec["shape"], spec["color"], seed=i)
            img.save(os.path.join(class_dir, f"{class_name}_{i}.jpg"))
    print(f"Generated {len(CLASSES)} classes x {IMAGES_PER_CLASS} images in {OUT_DIR}")


if __name__ == "__main__":
    generate()
