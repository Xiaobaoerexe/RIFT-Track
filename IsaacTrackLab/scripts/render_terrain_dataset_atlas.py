#!/usr/bin/env python3
"""Render plan-view QA maps and 5x4 family atlases for a terrain dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from isaac_track_lab.task.terrain_routes import terrain_height


PALETTES = {
    "city": ((39, 46, 53), (188, 198, 205)),
    "park": ((24, 70, 39), (190, 211, 118)),
    "forest_trail": ((42, 39, 27), (137, 156, 84)),
}
ROUTE_COLORS = ((255, 74, 64), (0, 210, 255), (255, 202, 48), (255, 91, 196), (138, 255, 93))


def lerp_color(low: tuple[int, int, int], high: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return tuple(round(a + (b - a) * t) for a, b in zip(low, high))


def render_map(map_config: dict, manifest: dict, routes: list[dict], size: int = 360) -> Image.Image:
    family = map_config["family"]
    sample = 144
    heights = [
        terrain_height(map_config, -90.0 + x * 180.0 / (sample - 1), 90.0 - y * 180.0 / (sample - 1))
        for y in range(sample) for x in range(sample)
    ]
    low_h, high_h = min(heights), max(heights)
    low_color, high_color = PALETTES[family]
    pixels = [lerp_color(low_color, high_color, (h - low_h) / max(high_h - low_h, 1e-9)) for h in heights]
    terrain = Image.new("RGB", (sample, sample))
    terrain.putdata(pixels)
    terrain = terrain.resize((size, size), Image.Resampling.BILINEAR).convert("RGBA")
    draw = ImageDraw.Draw(terrain, "RGBA")

    def xy(x_m: float, y_m: float) -> tuple[float, float]:
        return ((x_m + 90.0) / 180.0 * size, (90.0 - y_m) / 180.0 * size)

    surface_colors = {
        "sidewalk": (177, 164, 145, 235), "road": (47, 51, 55, 245),
        "park_path": (202, 185, 139, 235), "forest_trail": (113, 78, 45, 235),
    }
    for corridor in map_config.get("surface_corridors", []):
        points = [xy(float(point[0]), float(point[1])) for point in corridor["points_m"]]
        width = max(2, round(float(corridor["width_m"]) / 180.0 * size))
        draw.line(points, fill=surface_colors.get(corridor.get("kind"), (180,180,180,220)),
                  width=width, joint="curve")

    for curb in map_config.get("curb_segments", []):
        cx, cy = map(float, curb["center_m"]); length = float(curb["length_m"])
        yaw = float(curb.get("yaw_deg", 0.0))
        if abs(yaw) < 45.0:
            left, right = xy(cx-length*.5, cy), xy(cx+length*.5, cy)
        else:
            left, right = xy(cx, cy-length*.5), xy(cx, cy+length*.5)
        draw.line((left, right), fill=(232, 230, 218, 230), width=1)

    for zone in map_config.get("challenge_zones", []):
        cx, cy = map(float, zone["center_m"]); width, height = map(float, zone["size_m"])
        left, top = xy(cx - width * 0.5, cy + height * 0.5)
        right, bottom = xy(cx + width * 0.5, cy - height * 0.5)
        fill = (210, 38, 45, 105) if not zone["traversable"] else (255, 145, 0, 42)
        outline = (255, 50, 55, 230) if not zone["traversable"] else (255, 196, 65, 180)
        if zone.get("shape") == "ellipse":
            draw.ellipse((left, top, right, bottom), fill=(20, 112, 170, 190),
                         outline=(142, 211, 238, 240), width=2)
        else:
            draw.rectangle((left, top, right, bottom), fill=fill, outline=outline, width=2)

    for feature in map_config.get("traversable_step_features", []):
        px, py = xy(*map(float, feature["center_m"]))
        draw.rectangle((px-3, py-3, px+3, py+3), fill=(255, 142, 35, 240),
                       outline=(25, 20, 10, 230), width=1)

    for obstacle in manifest["obstacles"]:
        px, py = xy(float(obstacle["x_m"]), float(obstacle["y_m"]))
        radius = max(1.5, float(obstacle["radius_m"]) / 180.0 * size)
        color = (14, 17, 20, 210) if obstacle["asset"] == "procedural_building" else (245, 245, 238, 175)
        draw.ellipse((px-radius, py-radius, px+radius, py+radius), fill=color)

    for index, route in enumerate(routes[:5]):
        points = [xy(float(p[0]), float(p[1])) for p in route["points_m"]]
        draw.line(points, fill=ROUTE_COLORS[index], width=3, joint="curve")
        if points:
            sx, sy = points[0]; ex, ey = points[-1]
            draw.ellipse((sx-4, sy-4, sx+4, sy+4), fill=(255,255,255,245), outline=ROUTE_COLORS[index], width=2)
            draw.rectangle((ex-4, ey-4, ex+4, ey+4), fill=ROUTE_COLORS[index])
    draw.rectangle((0, 0, size-1, size-1), outline=(255, 255, 255, 180), width=2)
    return terrain.convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/terrain_dataset_v2_catalog.json")
    parser.add_argument("--scene-root", type=Path, default=ROOT / "assets/scenes/terrain_dataset_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/terrain_dataset_v2/atlas")
    args = parser.parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    scene_root, output = args.scene_root.resolve(), args.output.resolve()
    map_output = output / "maps"
    map_output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    rendered: dict[str, list[tuple[dict, Image.Image]]] = {family: [] for family in PALETTES}
    for map_config in config["maps"]:
        map_dir = scene_root / map_config["id"]
        manifest = json.loads((map_dir / "scene_manifest.json").read_text(encoding="utf-8"))
        dataset = json.loads((map_dir / "routes/generated_routes.json").read_text(encoding="utf-8"))
        image = render_map(map_config, manifest, dataset["routes"])
        canvas = Image.new("RGB", (360, 390), (20, 23, 27))
        canvas.paste(image, (0, 30))
        ImageDraw.Draw(canvas).text(
            (9, 9), f"{map_config['id']}  obstacles={manifest['obstacle_count']}  routes=20",
            fill=(244, 246, 248), font=font,
        )
        canvas.save(map_output / f"{map_config['id']}.png", quality=95)
        rendered[map_config["family"]].append((map_config, canvas))
    for family, items in rendered.items():
        atlas = Image.new("RGB", (5 * 360, 4 * 390 + 46), (13, 15, 18))
        draw = ImageDraw.Draw(atlas)
        draw.text((15, 15), f"{config['batch_id']} — {family} — 20 maps — first 5/20 routes shown per map",
                  fill=(245, 247, 250), font=font)
        for index, (_, image) in enumerate(items):
            atlas.paste(image, ((index % 5) * 360, 46 + (index // 5) * 390))
        atlas_path = output / f"{family}_atlas.png"
        atlas.save(atlas_path, quality=95)
        print("TERRAIN_ATLAS_OK", family, atlas_path)


if __name__ == "__main__":
    main()
