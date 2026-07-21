"""Render the compact post-repair Waterbirds comparison used by project_report.tex."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROWS = [
    ("SimCLR baseline", 48.04, 2.82, "#315a9b"),
    ("AUG q=0.50", 47.62, 7.31, "#c46a18"),
    ("AUG q=0.75", 47.60, 4.16, "#c46a18"),
    ("AUG q=0.90", 46.09, 2.34, "#c46a18"),
    ("AUG q=0.95", 46.93, 6.10, "#c46a18"),
    ("REG lambda=0.001", 46.47, 4.72, "#2f7d68"),
    ("REG lambda=0.01", 43.46, 5.46, "#2f7d68"),
    ("REG lambda=0.1", 48.49, 4.37, "#2f7d68"),
    ("REG lambda=1.0", 43.84, 5.00, "#2f7d68"),
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def main() -> None:
    width, height = 1800, 1080
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    label_font = font(30)
    tick_font = font(25)
    value_font = font(26, bold=True)

    x0, x1 = 390, 1640
    lo, hi = 36.0, 58.0

    def px(value: float) -> float:
        return x0 + (value - lo) / (hi - lo) * (x1 - x0)

    draw.text((x0, 18), "Worst-group validation accuracy (%, mean +/- sample SD; n=3)", fill="#526075", font=tick_font)
    for tick in [36, 40, 44, 48, 52, 56, 58]:
        x = px(tick)
        draw.line((x, 74, x, 984), fill="#d9dee8", width=2)
        label = str(tick)
        box = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((x - (box[2] - box[0]) / 2, 995), label, fill="#526075", font=tick_font)

    baseline_x = px(48.04)
    for y in range(74, 985, 18):
        draw.line((baseline_x, y, baseline_x, min(y + 10, 984)), fill="#315a9b", width=3)

    y = 120
    for label, mean, sd, color in ROWS:
        label_box = draw.textbbox((0, 0), label, font=label_font)
        draw.text((x0 - 28 - (label_box[2] - label_box[0]), y - 18), label, fill="#172033", font=label_font)
        left, right = px(mean - sd), px(mean + sd)
        draw.line((left, y, right, y), fill="#526075", width=5)
        draw.line((left, y - 12, left, y + 12), fill="#526075", width=5)
        draw.line((right, y - 12, right, y + 12), fill="#526075", width=5)
        r = 11
        draw.ellipse((px(mean) - r, y - r, px(mean) + r, y + r), fill=color)
        value = f"{mean:.2f}"
        draw.text((min(right + 20, width - 110), y - 18), value, fill="#172033", font=value_font)
        y += 98

    draw.text((x0, 1040), "Final epoch-1000 summaries; seeds 1, 3, and 4.", fill="#526075", font=tick_font)
    out = Path(__file__).resolve().parents[1] / "figures" / "current_postrepair_wg.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out, optimize=True)
    print(out)


if __name__ == "__main__":
    main()
