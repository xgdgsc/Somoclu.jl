#!/usr/bin/env python3
"""Regenerate Somoclu.jl logo assets (hexagonal SOM + Julia 3 colors).

Requires: rsvg-convert (librsvg)

    python3 scripts/generate_logo.py
"""
import math
from pathlib import Path
from subprocess import run

ROOT = Path(__file__).resolve().parents[1]
ASSETS_DOCS = ROOT / "docs" / "src" / "assets"
ASSETS_ROOT = ROOT / "assets"

PURPLE, RED, GREEN, BLUE = "#9558B2", "#CB3C33", "#389826", "#4063D8"
W = H = 512
CX, CY = W / 2, H / 2 + 6


def hex_corners(cx, cy, size):
    return [
        (
            cx + size * math.cos(math.radians(60 * i - 30)),
            cy + size * math.sin(math.radians(60 * i - 30)),
        )
        for i in range(6)
    ]


def axial_to_pixel(q, r, size):
    return size * (math.sqrt(3) * q + math.sqrt(3) / 2 * r), size * (1.5 * r)


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(
        *(max(0, min(255, int(round(c * 255)))) for c in rgb)
    )


def lerp(a, b, t):
    return a + (b - a) * t


def mix_rgb(c1, c2, t):
    return tuple(lerp(a, b, t) for a, b in zip(c1, c2))


def axial_dist(a, b):
    aq, ar = a
    bq, br = b
    return (abs(aq - bq) + abs(aq + ar - bq - br) + abs(ar - br)) / 2.0


def darken_hex(h, t=0.22):
    return rgb_to_hex(mix_rgb(hex_to_rgb(h), (0, 0, 0), t))


def build_svg() -> str:
    R, size = 3, 44
    cells = [
        (q, r)
        for q in range(-R, R + 1)
        for r in range(-R, R + 1)
        if max(abs(q), abs(r), abs(-q - r)) <= R
    ]
    centers = {"purple": (0, -2), "red": (-2, 2), "green": (2, 0)}
    center_colors = {"purple": PURPLE, "red": RED, "green": GREEN}
    center_rgb = {k: hex_to_rgb(v) for k, v in center_colors.items()}

    def cell_color(q, r):
        ds = {n: axial_dist((q, r), c) for n, c in centers.items()}
        T = 0.55
        exps = {n: math.exp(-d / T) for n, d in ds.items()}
        Z = sum(exps.values())
        w = {n: exps[n] / Z for n in exps}
        winner = max(w, key=w.get)
        if w[winner] > 0.62:
            boost = (w[winner] - 0.62) / 0.38
            for n in w:
                w[n] = lerp(w[n], 1.0 if n == winner else 0.0, 0.55 * boost)
            Z = sum(w.values())
            w = {n: w[n] / Z for n in w}
        rgb = [0.0, 0.0, 0.0]
        for n, ww in w.items():
            for i in range(3):
                rgb[i] += ww * center_rgb[n][i]
        rgb = [min(1.0, c * 1.04 + 0.02) for c in rgb]
        return rgb_to_hex(tuple(rgb))

    parts = [
        f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="Somoclu.jl logo">
  <title>Somoclu.jl</title>
  <desc>Hexagonal self-organizing map with Julia purple, red, and green codebook prototypes</desc>
  <defs>
    <filter id="soft" x="-25%" y="-25%" width="150%" height="150%">
      <feDropShadow dx="0" dy="1.5" stdDeviation="2.5" flood-color="#1a1a1a" flood-opacity="0.18"/>
    </filter>
    <radialGradient id="hi" cx="35%" cy="30%" r="65%">
      <stop offset="0%" stop-color="#ffffff" stop-opacity="0.45"/>
      <stop offset="100%" stop-color="#ffffff" stop-opacity="0"/>
    </radialGradient>
  </defs>
'''
    ]

    cells_sorted = sorted(
        cells,
        key=lambda qr: (
            axial_to_pixel(qr[0], qr[1], size)[1],
            axial_to_pixel(qr[0], qr[1], size)[0],
        ),
    )
    parts.append('  <g id="som-lattice" filter="url(#soft)">\n')
    for q, r in cells_sorted:
        x, y = axial_to_pixel(q, r, size)
        px, py = CX + x, CY + y
        fill = cell_color(q, r)
        stroke = darken_hex(fill, 0.2)
        pts = hex_corners(px, py, size * 0.96)
        d = "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in pts) + " Z"
        parts.append(
            f'    <path d="{d}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="1.5" stroke-linejoin="round"/>\n'
        )
    parts.append("  </g>\n")

    parts.append('  <g id="codebook-prototypes">\n')
    dot_r = size * 0.52
    for name, (q, r) in centers.items():
        x, y = axial_to_pixel(q, r, size)
        px, py = CX + x, CY + y
        col = center_colors[name]
        parts.append(
            f'    <circle cx="{px:.2f}" cy="{py:.2f}" r="{dot_r + 3.5:.2f}" fill="#ffffff"/>\n'
        )
        parts.append(
            f'    <circle cx="{px:.2f}" cy="{py:.2f}" r="{dot_r:.2f}" fill="{col}"/>\n'
        )
        parts.append(
            f'    <circle cx="{px:.2f}" cy="{py:.2f}" r="{dot_r:.2f}" fill="url(#hi)"/>\n'
        )
    parts.append("  </g>\n")

    parts.append('  <g id="data-samples">\n')
    samples = [
        (-95, 0.93, "purple"),
        (-72, 0.89, "purple"),
        (205, 0.91, "red"),
        (230, 0.87, "red"),
        (35, 0.91, "green"),
        (58, 0.87, "green"),
    ]
    map_outer = size * (R * 1.52 + 0.85)
    for ang, rf, key in samples:
        a = math.radians(ang)
        px = CX + map_outer * rf * math.cos(a)
        py = CY + map_outer * rf * math.sin(a)
        parts.append(
            f'    <circle cx="{px:.2f}" cy="{py:.2f}" r="7" '
            f'fill="{center_colors[key]}" opacity="0.8"/>\n'
        )
    parts.append("  </g>\n")

    ring_r = size * (R + 1.5)
    parts.append(
        f'  <circle cx="{CX}" cy="{CY}" r="{ring_r:.2f}" fill="none" '
        f'stroke="{BLUE}" stroke-width="2.75" stroke-dasharray="9 12" opacity="0.28"/>\n'
    )
    parts.append("</svg>\n")
    return "".join(parts)


def main():
    ASSETS_DOCS.mkdir(parents=True, exist_ok=True)
    ASSETS_ROOT.mkdir(parents=True, exist_ok=True)
    svg_path = ASSETS_DOCS / "logo.svg"
    svg_path.write_text(build_svg())
    print(f"wrote {svg_path.relative_to(ROOT)}")

    for name, px in [("logo.png", 512), ("logo-128.png", 128), ("logo-64.png", 64)]:
        dest = ASSETS_DOCS / name
        run(
            [
                "rsvg-convert",
                "-w",
                str(px),
                "-h",
                str(px),
                "--background-color=transparent",
                str(svg_path),
                "-o",
                str(dest),
            ],
            check=True,
        )
        print(f"wrote {dest.relative_to(ROOT)}")

    for name in ("logo.svg", "logo.png", "logo-128.png"):
        run(["cp", str(ASSETS_DOCS / name), str(ASSETS_ROOT / name)], check=True)
    print(f"copied into {ASSETS_ROOT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
