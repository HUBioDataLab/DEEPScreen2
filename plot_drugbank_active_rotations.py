#!/usr/bin/env python3
"""Plot exact active-rotation distributions for DrugBank repurposing runs."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
WIDTH = 1786
HEIGHT = 826
PLOT_LEFT = 168
PLOT_TOP = 92
PLOT_RIGHT = 1658
PLOT_BOTTOM = 734
Y_MAX = 550

FONT_REGULAR = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
FONT_BOLD = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")

PLOTS = (
    (
        ROOT / "predictions_CHEMBL301.csv",
        "CDK2 (CHEMBL301)",
        ROOT / "drugbank_CDK2_active_rotations_distribution.png",
    ),
    (
        ROOT / "predictions_CHEMBL4282.csv",
        "AKT1 (CHEMBL4282)",
        ROOT / "drugbank_AKT1_active_rotations_distribution.png",
    ),
)


def load_counts(csv_path: Path) -> Counter[int]:
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    required = {
        "molecule_chembl_id",
        "active_rotations",
        "total_rotations",
        "complete_rotations",
    }
    if not rows or not required.issubset(rows[0]):
        missing = required.difference(rows[0] if rows else {})
        raise ValueError(f"{csv_path.name}: missing columns {sorted(missing)}")

    molecule_ids = [row["molecule_chembl_id"] for row in rows]
    if len(molecule_ids) != len(set(molecule_ids)):
        raise ValueError(f"{csv_path.name}: duplicate molecule IDs found")

    if any(int(row["total_rotations"]) != 36 for row in rows):
        raise ValueError(f"{csv_path.name}: expected 36 rotations for every molecule")
    if any(row["complete_rotations"].lower() != "true" for row in rows):
        raise ValueError(f"{csv_path.name}: incomplete rotation set found")

    active_rotations = [int(row["active_rotations"]) for row in rows]
    if any(value < 0 or value > 36 for value in active_rotations):
        raise ValueError(f"{csv_path.name}: active rotation count outside 0..36")

    return Counter(active_rotations)


def centered_text(
    draw: ImageDraw.ImageDraw,
    center_x: float,
    y: float,
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: str,
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text((center_x - (bbox[2] - bbox[0]) / 2, y), text, font=font, fill=fill)


def render_plot(counts: Counter[int], target: str, output_path: Path) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#FFFFFF")
    draw = ImageDraw.Draw(image)

    title_font = ImageFont.truetype(str(FONT_BOLD), 28)
    axis_font = ImageFont.truetype(str(FONT_BOLD), 23)
    tick_font = ImageFont.truetype(str(FONT_REGULAR), 18)

    plot_width = PLOT_RIGHT - PLOT_LEFT
    plot_height = PLOT_BOTTOM - PLOT_TOP
    band_width = plot_width / 36
    bar_width = band_width * 0.80

    title = f"Distribution of Active Rotations of DrugBank Repurposing Molecules — {target}"
    centered_text(draw, WIDTH / 2, 31, title, title_font, "#111111")

    for tick_value in range(0, 501, 100):
        y = PLOT_BOTTOM - (tick_value / Y_MAX) * plot_height
        if tick_value > 0:
            dash_length = 7
            gap_length = 5
            x = PLOT_LEFT
            while x < PLOT_RIGHT:
                draw.line(
                    (x, y, min(x + dash_length, PLOT_RIGHT), y),
                    fill="#E4E4E4",
                    width=1,
                )
                x += dash_length + gap_length

        draw.line((PLOT_LEFT - 7, y, PLOT_LEFT, y), fill="#333333", width=2)
        label = str(tick_value)
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.text(
            (PLOT_LEFT - 17 - (bbox[2] - bbox[0]), y - (bbox[3] - bbox[1]) / 2 - 1),
            label,
            font=tick_font,
            fill="#222222",
        )

    for rotation in range(1, 37):
        center_x = PLOT_LEFT + (rotation - 0.5) * band_width
        value = counts[rotation]
        bar_top = PLOT_BOTTOM - (value / Y_MAX) * plot_height
        draw.rectangle(
            (
                round(center_x - bar_width / 2),
                round(bar_top),
                round(center_x + bar_width / 2),
                PLOT_BOTTOM,
            ),
            fill="#E95D48",
            outline="#333333",
            width=1,
        )

        draw.line(
            (center_x, PLOT_BOTTOM, center_x, PLOT_BOTTOM + 7),
            fill="#333333",
            width=2,
        )
        label = str(rotation)
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.text(
            (center_x - (bbox[2] - bbox[0]) / 2, PLOT_BOTTOM + 14),
            label,
            font=tick_font,
            fill="#222222",
        )

    draw.rectangle(
        (PLOT_LEFT, PLOT_TOP, PLOT_RIGHT, PLOT_BOTTOM),
        outline="#4A4A4A",
        width=1,
    )

    centered_text(
        draw,
        (PLOT_LEFT + PLOT_RIGHT) / 2,
        776,
        "Number of Active Rotations",
        axis_font,
        "#111111",
    )

    y_label = "Count of Molecules"
    y_bbox = draw.textbbox((0, 0), y_label, font=axis_font)
    y_image = Image.new(
        "RGBA",
        (y_bbox[2] - y_bbox[0] + 12, y_bbox[3] - y_bbox[1] + 12),
        (255, 255, 255, 0),
    )
    y_draw = ImageDraw.Draw(y_image)
    y_draw.text((6, 6 - y_bbox[1]), y_label, font=axis_font, fill="#111111")
    y_image = y_image.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    image.paste(
        y_image,
        (
            86 - y_image.width // 2,
            round((PLOT_TOP + PLOT_BOTTOM) / 2 - y_image.height / 2),
        ),
        y_image,
    )

    image.save(output_path, dpi=(200, 200), optimize=True)


def main() -> None:
    for csv_path, target, output_path in PLOTS:
        counts = load_counts(csv_path)
        render_plot(counts, target, output_path)
        print(f"Wrote {output_path.name}")


if __name__ == "__main__":
    main()
