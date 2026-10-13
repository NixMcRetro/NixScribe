#!/usr/bin/env python3
"""Generate NixScribe's blue waveform Dock icon as PNG + multi-size macOS ICNS.

This file is deliberately reproducible. Assets contain no licensed fonts or
third-party artwork, and the 1024px master is scaled by Pillow for macOS.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

SIZE = 1024
HERE = Path(__file__).resolve().parent


def create_icon() -> Image.Image:
    canvas = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))

    # Drop shadow under an Apple-style squircle.
    shadow_mask = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(shadow_mask).rounded_rectangle(
        (102, 98, 922, 912), radius=207, fill=190
    )
    soft_shadow = shadow_mask.filter(ImageFilter.GaussianBlur(34))
    canvas.paste(Image.new("RGBA", (SIZE, SIZE), (3, 20, 74, 100)),
                 (0, 0), soft_shadow)

    tile = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    gradient = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    pen = ImageDraw.Draw(gradient)
    for y in range(84, 926):
        f = (y - 84) / 842
        color = (
            round(27 + (11 - 27) * f),
            round(147 + (65 - 147) * f),
            round(254 + (202 - 254) * f),
            255,
        )
        pen.line((84, y, 940, y), fill=color, width=1)
    tile_mask = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(tile_mask).rounded_rectangle(
        (99, 88, 925, 914), radius=214, fill=255
    )
    tile.paste(gradient, (0, 0), tile_mask)
    canvas = Image.alpha_composite(canvas, tile)

    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.arc((114, 107, 911, 892), 188, 345,
             fill=(255, 255, 255, 72), width=12)
    # Horizontal center line links the seven rounded waveform bars.
    draw.rounded_rectangle((252, 464, 776, 495), radius=16,
                           fill=(240, 248, 255, 110))

    waveform = [
        (271, 390, 340, 568),
        (353, 330, 421, 627),
        (433, 266, 502, 682),
        (513, 221, 582, 730),
        (593, 292, 662, 654),
        (673, 348, 742, 610),
        (753, 405, 814, 559),
    ]
    for rect in waveform:
        draw.rounded_rectangle((rect[0] + 6, rect[1] + 10,
                                rect[2] + 6, rect[3] + 10),
                               radius=32, fill=(0, 32, 100, 37))
        draw.rounded_rectangle(rect, radius=32,
                               fill=(255, 255, 255, 248))

    # Two miniature transcript lines distinguish it from a generic recorder.
    draw.rounded_rectangle((365, 785, 660, 803), radius=9,
                           fill=(232, 245, 255, 176))
    draw.rounded_rectangle((425, 823, 600, 840), radius=9,
                           fill=(232, 245, 255, 112))
    return canvas


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    icon = create_icon()
    png = HERE / "NixScribe.png"
    icns = HERE / "NixScribe.icns"
    icon.save(png, format="PNG", optimize=True)
    icon.save(icns, format="ICNS")
    # Validate the file immediately, before PyInstaller processes it.
    with Image.open(icns) as check:
        check.verify()
    print(f"Created {png} and {icns}")


if __name__ == "__main__":
    main()
