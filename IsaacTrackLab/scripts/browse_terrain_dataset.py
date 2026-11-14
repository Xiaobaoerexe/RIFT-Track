#!/usr/bin/env python3
"""Interactively browse all terrain-dataset maps and their 20 routes in one Isaac Sim session."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/terrain_dataset_v3_realistic.json"
DEFAULT_SCENE_ROOT = ROOT / "assets/scenes/terrain_dataset_v3_realistic"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENE_ROOT)
parser.add_argument("--map", dest="initial_map", default="city_01", help="initial map id")
parser.add_argument("--route", dest="initial_route", type=int, default=1, help="initial route number, 1-20")
parser.add_argument("--show-all-routes", action="store_true")
parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
AppLauncher.add_app_launcher_args(parser)
# Isaac Lab 3 resolves the window mode from the selected visualizer.  Without
# an explicit Kit visualizer it forces headless execution even when
# AppLauncher receives headless=False.
parser.set_defaults(visualizer=["kit"])
args = parser.parse_args()
app_launcher = AppLauncher(args, headless=args.smoke_test, enable_cameras=True)
simulation_app = app_launcher.app

import carb
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom

if not args.smoke_test:
    import omni.appwindow
    import omni.kit.viewport.utility
    import omni.ui as ui
    from omni.kit.viewport.utility.camera_state import ViewportCameraState


ROUTE_COLORS = (
    Gf.Vec3f(1.00, 0.18, 0.10), Gf.Vec3f(0.00, 0.78, 1.00),
    Gf.Vec3f(1.00, 0.72, 0.05), Gf.Vec3f(0.38, 1.00, 0.22),
    Gf.Vec3f(1.00, 0.18, 0.72), Gf.Vec3f(0.55, 0.36, 1.00),
)


class TerrainDatasetBrowser:
    def __init__(self) -> None:
        self.config_path = args.config.resolve()
        self.scene_root = args.scene_root.resolve()
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.maps = self.config["maps"]
        ids = [item["id"] for item in self.maps]
        if args.initial_map not in ids:
            raise ValueError(f"unknown map id {args.initial_map}; first ids: {ids[:5]}")
        if not 1 <= args.initial_route <= 20:
            raise ValueError("--route must be in 1..20")
        self.map_index = ids.index(args.initial_map)
        self.route_index = args.initial_route - 1
        self.show_all_routes = bool(args.show_all_routes)
        self.show_collisions = False
        self.routes: list[dict] = []
        self.manifest: dict = {}

        context = omni.usd.get_context()
        context.new_stage()
        for _ in range(3):
            simulation_app.update()
        self.stage = context.get_stage()
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        world = UsdGeom.Xform.Define(self.stage, "/World")
        self.stage.SetDefaultPrim(world.GetPrim())
        if not args.smoke_test:
            self._create_camera()
        self._create_ui()
        if not args.smoke_test:
            self._subscribe_keyboard()
        self.load_map()

    @property
    def map_config(self) -> dict:
        return self.maps[self.map_index]

    def _create_camera(self) -> None:
        camera = UsdGeom.Camera.Define(self.stage, "/World/InspectionCamera")
        camera.CreateFocalLengthAttr(25.0)
        camera.CreateHorizontalApertureAttr(36.0)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 2000.0))
        camera.GetPrim().CreateAttribute(
            "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d,
            custom=True, variability=Sdf.VariabilityUniform,
        ).Set(Gf.Vec3d(0.0, 0.0, -10.0))
        self.viewport = omni.kit.viewport.utility.get_active_viewport()
        if self.viewport is not None:
            self.viewport.set_active_camera("/World/InspectionCamera")
            self.camera_state = ViewportCameraState("/World/InspectionCamera", self.viewport)
            self.camera_oblique()
        else:
            self.camera_state = None

    def _create_ui(self) -> None:
        if args.smoke_test:
            self.window = None
            self.map_label = self.route_label = self.metric_label = None
            return
        self.window = ui.Window("Terrain Dataset Browser", width=430, height=280)
        with self.window.frame, ui.VStack(spacing=5):
            self.map_label = ui.Label("", height=24)
            self.route_label = ui.Label("", height=24)
            self.metric_label = ui.Label("", word_wrap=True, height=48)
            with ui.HStack(height=32):
                ui.Button("Prev map", clicked_fn=lambda: self.change_map(-1))
                ui.Button("Next map", clicked_fn=lambda: self.change_map(1))
            with ui.HStack(height=32):
                ui.Button("Prev route", clicked_fn=lambda: self.change_route(-1))
                ui.Button("Next route", clicked_fn=lambda: self.change_route(1))
            with ui.HStack(height=32):
                ui.Button("City", clicked_fn=lambda: self.jump_family("city"))
                ui.Button("Park", clicked_fn=lambda: self.jump_family("park"))
                ui.Button("Forest", clicked_fn=lambda: self.jump_family("forest_trail"))
            with ui.HStack(height=32):
                ui.Button("Selected/all routes [A]", clicked_fn=self.toggle_all_routes)
                ui.Button("Collision proxies [C]", clicked_fn=self.toggle_collisions)
            with ui.HStack(height=32):
                ui.Button("Top", clicked_fn=self.camera_top)
                ui.Button("Oblique [F]", clicked_fn=self.camera_oblique)
                ui.Button("Ground", clicked_fn=self.camera_ground)
            ui.Label("Keys: LEFT/RIGHT map | UP/DOWN route | A routes | C collision | F camera", height=24)
        self.window.visible = True
        asyncio.ensure_future(self._dock_ui_window())

    async def _dock_ui_window(self) -> None:
        """Dock controls into the Property area so the panel cannot open off-screen."""
        import omni.kit.app

        for _ in range(12):
            custom = ui.Workspace.get_window("Terrain Dataset Browser")
            property_window = ui.Workspace.get_window("Property")
            if custom is not None and property_window is not None:
                custom.dock_in(property_window, ui.DockPosition.SAME, 1.0)
                custom.visible = True
                custom.focus()
                return
            await omni.kit.app.get_app().next_update_async()

    def _subscribe_keyboard(self) -> None:
        self.input = carb.input.acquire_input_interface()
        self.keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self.keyboard_sub = self.input.subscribe_to_keyboard_events(self.keyboard, self._on_key)

    def _on_key(self, event) -> bool:
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True
        actions = {
            "LEFT": lambda: self.change_map(-1), "RIGHT": lambda: self.change_map(1),
            "UP": lambda: self.change_route(-1), "DOWN": lambda: self.change_route(1),
            "A": self.toggle_all_routes, "C": self.toggle_collisions,
            "F": self.camera_oblique,
        }
        action = actions.get(event.input.name)
        if action is not None:
            action()
        return True

    def load_map(self) -> None:
        item = self.map_config
        map_dir = self.scene_root / item["id"]
        manifest_path = map_dir / "scene_manifest.json"
        route_path = map_dir / "routes/generated_routes.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        route_dataset = json.loads(route_path.read_text(encoding="utf-8"))
        self.routes = route_dataset["routes"]
        if len(self.routes) != 20:
            raise RuntimeError(f"{item['id']} has {len(self.routes)} routes, expected 20")
        usd_path = map_dir / self.manifest["usd"]
        self.stage.RemovePrim("/World/Map")
        map_prim = self.stage.DefinePrim("/World/Map", "Xform")
        map_prim.GetReferences().AddReference(str(usd_path))
        self.show_collisions = False
        self.author_routes()
        self.update_labels()
        print(
            "TERRAIN_BROWSER_MAP", f"{self.map_index+1:02d}/60", item["id"],
            f"route={self.route_index+1:02d}/20", usd_path, flush=True,
        )

    def author_routes(self) -> None:
        self.stage.RemovePrim("/World/RouteInspection")
        UsdGeom.Xform.Define(self.stage, "/World/RouteInspection")
        indices = range(len(self.routes)) if self.show_all_routes else (self.route_index,)
        for index in indices:
            selected = index == self.route_index
            route = self.routes[index]
            curve = UsdGeom.BasisCurves.Define(
                self.stage, f"/World/RouteInspection/Route_{index+1:02d}"
            )
            curve.CreateTypeAttr("linear")
            curve.CreateCurveVertexCountsAttr([len(route["points_m"])])
            curve.CreatePointsAttr([
                Gf.Vec3f(float(x), float(y), float(z) + (0.42 if selected else 0.30))
                for x, y, z in route["points_m"]
            ])
            curve.CreateWidthsAttr([0.34 if selected else 0.10])
            curve.SetWidthsInterpolation("constant")
            color = ROUTE_COLORS[index % len(ROUTE_COLORS)] if selected else Gf.Vec3f(0.55, 0.60, 0.64)
            curve.CreateDisplayColorAttr([color])
            curve.GetPrim().CreateAttribute("route:index", Sdf.ValueTypeNames.Int).Set(index + 1)
            curve.GetPrim().CreateAttribute("route:length_m", Sdf.ValueTypeNames.Double).Set(
                float(route["length_m"])
            )
        selected_route = self.routes[self.route_index]
        self._marker("Start", selected_route["points_m"][0], Gf.Vec3f(0.10, 1.0, 0.20))
        self._marker("End", selected_route["points_m"][-1], Gf.Vec3f(1.0, 0.12, 0.08))
        self.update_labels()

    def _marker(self, name: str, point: list[float], color: Gf.Vec3f) -> None:
        marker = UsdGeom.Sphere.Define(self.stage, f"/World/RouteInspection/{name}")
        marker.CreateRadiusAttr(0.85)
        marker.CreateDisplayColorAttr([color])
        UsdGeom.XformCommonAPI(marker).SetTranslate(
            Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]) + 1.0)
        )

    def update_labels(self) -> None:
        if self.map_label is None or not self.routes:
            return
        item, route = self.map_config, self.routes[self.route_index]
        self.map_label.text = f"Map {self.map_index+1:02d}/60: {item['id']}  |  {item['family']}"
        self.route_label.text = (
            f"Route {self.route_index+1:02d}/20  |  "
            f"{'all routes visible' if self.show_all_routes else 'selected route only'}"
        )
        self.metric_label.text = (
            f"Length {route['length_m']:.2f} m   max slope {route['maximum_slope_deg']:.2f} deg   "
            f"max turn {route['maximum_turn_deg']:.2f} deg\n"
            f"Min obstacle clearance {route['minimum_obstacle_clearance_m']:.2f} m   "
            f"obstacles {self.manifest['obstacle_count']}"
        )

    def change_map(self, delta: int) -> None:
        self.map_index = (self.map_index + delta) % len(self.maps)
        self.load_map()

    def change_route(self, delta: int) -> None:
        self.route_index = (self.route_index + delta) % len(self.routes)
        self.author_routes()
        print(
            "TERRAIN_BROWSER_ROUTE", self.map_config["id"],
            f"{self.route_index+1:02d}/20", flush=True,
        )

    def jump_family(self, family: str) -> None:
        self.map_index = next(i for i, item in enumerate(self.maps) if item["family"] == family)
        self.load_map()

    def toggle_all_routes(self) -> None:
        self.show_all_routes = not self.show_all_routes
        self.author_routes()

    def toggle_collisions(self) -> None:
        self.show_collisions = not self.show_collisions
        collision_root = self.stage.GetPrimAtPath("/World/Map/Collision")
        if collision_root:
            token = UsdGeom.Tokens.inherited if self.show_collisions else UsdGeom.Tokens.invisible
            for prim in Usd.PrimRange(collision_root):
                if prim.IsA(UsdGeom.Imageable):
                    UsdGeom.Imageable(prim).GetVisibilityAttr().Set(token)
        print("TERRAIN_BROWSER_COLLISIONS", self.show_collisions, flush=True)

    def _set_camera(self, eye: Gf.Vec3d, target: Gf.Vec3d) -> None:
        if self.camera_state is not None:
            self.camera_state.set_position_world(eye, True)
            self.camera_state.set_target_world(target, True)
            self.viewport.set_active_camera("/World/InspectionCamera")

    def camera_top(self) -> None:
        self._set_camera(Gf.Vec3d(0.0, -0.2, 235.0), Gf.Vec3d(0.0, 0.0, 0.0))

    def camera_oblique(self) -> None:
        self._set_camera(Gf.Vec3d(145.0, -175.0, 135.0), Gf.Vec3d(0.0, 0.0, 2.0))

    def camera_ground(self) -> None:
        self._set_camera(Gf.Vec3d(-84.0, -105.0, 17.0), Gf.Vec3d(0.0, 5.0, 2.0))

    def close(self) -> None:
        if getattr(self, "keyboard_sub", None) is not None:
            self.input.unsubscribe_to_keyboard_events(self.keyboard, self.keyboard_sub)


def main() -> None:
    browser = TerrainDatasetBrowser()
    try:
        for _ in range(12 if args.smoke_test else 0):
            simulation_app.update()
        if args.smoke_test:
            terrain = browser.stage.GetPrimAtPath("/World/Map/Terrain/Heightfield")
            route = browser.stage.GetPrimAtPath(f"/World/RouteInspection/Route_{args.initial_route:02d}")
            if not terrain or not route:
                raise RuntimeError("browser failed to compose terrain or route")
            print("TERRAIN_BROWSER_SMOKE_OK", args.initial_map, args.initial_route)
            return
        while simulation_app.is_running():
            simulation_app.update()
    finally:
        browser.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
