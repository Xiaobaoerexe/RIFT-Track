"""Seeded, collision-aware trajectory generation for outdoor terrain maps."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
from math import atan, atan2, ceil, degrees, exp, hypot, pi, sin, cos, sqrt, tan, radians
import random
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class Obstacle2D:
    obstacle_id: str
    x_m: float
    y_m: float
    radius_m: float


@dataclass(frozen=True)
class GeneratedTerrainRoute:
    map_id: str
    seed: int
    algorithm: str
    points_m: tuple[tuple[float, float, float], ...]
    length_m: float
    elevation_gain_m: float
    maximum_slope_deg: float
    maximum_step_height_m: float
    maximum_turn_deg: float
    minimum_obstacle_clearance_m: float
    challenge_zones_visited: tuple[str, ...]
    target_reachable: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def terrain_height(map_config: dict[str, Any], x_m: float, y_m: float) -> float:
    """Analytic terrain shared by the USD builder and route planner."""

    terrain = map_config["terrain"]
    amplitude = float(terrain["amplitude_m"])
    frequency = float(terrain["frequency"])
    profile = terrain["profile"]
    # The original six-map batch predates per-map terrain phases.  Preserve its
    # exact analytic surface unless a terrain seed is explicitly authored.
    seed = int(terrain.get("seed", 0))
    if "seed" in terrain:
        phase_a = (seed * 0.754877666) % 6.283185307
        phase_b = (seed * 0.569840296 + 1.7) % 6.283185307
        phase_c = (seed * 0.438579021 + 3.1) % 6.283185307
    else:
        phase_a = phase_b = phase_c = 0.0
    wave = (
        0.48 * sin(frequency * x_m + phase_a)
        + 0.32 * cos(1.37 * frequency * y_m + phase_b)
        + 0.20 * sin(0.71 * frequency * (x_m + y_m) + phase_c)
    )
    if profile == "rolling_forest":
        height = amplitude * wave
    elif profile == "rocky_canyon":
        height = amplitude * (0.55 * wave + 0.08 * sin(0.19 * x_m) * cos(0.13 * y_m))
    elif profile == "alpine_slope":
        height = 0.055 * (x_m + 90.0) + amplitude * 0.55 * wave
    elif profile == "wetland":
        height = amplitude * 0.45 * wave - 0.15 * cos(0.08 * x_m) * cos(0.07 * y_m)
    elif profile == "desert_dunes":
        height = amplitude * (0.65 * sin(frequency * (x_m + 0.45 * y_m) + phase_a) + 0.35 * cos(0.73 * frequency * y_m + phase_b))
    elif profile == "coastal_scrub":
        height = 0.012 * (x_m + 90.0) + amplitude * (0.55 * wave + 0.12 * sin(0.16 * y_m + phase_c))
    elif profile == "rugged_city":
        height = amplitude * (0.62 * wave + 0.10 * sin(0.11*x_m + phase_b) * cos(0.09*y_m + phase_a))
    elif profile == "rolling_park":
        height = amplitude * (0.74 * wave + 0.18 * sin(0.055*(x_m-y_m) + phase_c))
    elif profile == "forest_ridge":
        height = amplitude * (0.68 * wave + 0.16 * sin(0.075*x_m + phase_a) * sin(0.065*y_m + phase_b))
    else:
        raise ValueError(f"unknown terrain profile: {profile}")

    roughness = float(terrain.get("roughness_m", 0.0))
    micro_frequency = float(terrain.get("micro_frequency", 0.13))
    height += roughness * (
        0.48 * sin(micro_frequency*x_m + phase_b) * cos(0.83*micro_frequency*y_m + phase_c)
        + 0.31 * sin(1.57*micro_frequency*(x_m+y_m) + phase_a)
        + 0.21 * cos(2.11*micro_frequency*(x_m-0.37*y_m) + phase_c)
    )
    height += float(terrain.get("tilt_x", 0.0)) * x_m + float(terrain.get("tilt_y", 0.0)) * y_m
    for feature in terrain.get("features", []):
        cx, cy = map(float, feature["center_m"])
        dx, dy = x_m-cx, y_m-cy
        amplitude_f = float(feature["amplitude_m"])
        sigma = max(float(feature.get("sigma_m", 12.0)), 1e-3)
        kind = feature["kind"]
        if kind == "gaussian":
            height += amplitude_f * exp(-0.5 * (dx*dx + dy*dy) / (sigma*sigma))
        elif kind == "lake_basin":
            radius_x = max(float(feature.get("radius_x_m", sigma)), 1e-3)
            radius_y = max(float(feature.get("radius_y_m", sigma)), 1e-3)
            normalized = sqrt((dx / radius_x) ** 2 + (dy / radius_y) ** 2)
            if normalized < 1.0:
                # A compact, smooth bowl reaches the untouched terrain exactly at
                # the shoreline.  The water mesh sits inside the bowl instead of
                # intersecting arbitrary hills like a pasted rectangular plane.
                height -= abs(amplitude_f) * (1.0 - normalized * normalized) ** 2
        elif kind == "step_bump":
            radius_x = max(float(feature.get("radius_x_m", sigma)), 1e-3)
            radius_y = max(float(feature.get("radius_y_m", sigma)), 1e-3)
            angle = radians(float(feature.get("angle_deg", 0.0)))
            along = dx*cos(angle) + dy*sin(angle)
            across = -dx*sin(angle) + dy*cos(angle)
            normalized = (along / radius_x) ** 2 + (across / radius_y) ** 2
            if normalized < 1.0:
                height += abs(amplitude_f) * (1.0 - normalized) ** 2
        elif kind in {"ridge", "rut"}:
            angle = radians(float(feature.get("angle_deg", 0.0)))
            along = dx*cos(angle) + dy*sin(angle)
            across = -dx*sin(angle) + dy*cos(angle)
            wavelength = max(float(feature.get("wavelength_m", 18.0)), 1e-3)
            sign = -1.0 if kind == "rut" else 1.0
            height += sign * amplitude_f * exp(-0.5*(across/sigma)**2) * cos(along/wavelength)
        else:
            raise ValueError(f"unknown terrain feature: {kind}")
    return height


def terrain_mesh_height(
    map_config: dict[str, Any], x_m: float, y_m: float, *,
    grid_spacing_m: float = 1.0, terrain_buffer_m: float = 30.0,
) -> float:
    """Height of the authored piecewise-planar render/collision terrain.

    ``terrain_height`` is the continuous design function.  USD stores samples
    of that function on a regular grid and triangulates every cell.  On rough
    terrain, evaluating the design function between vertices can differ from
    the actual triangle by several centimetres, so runtime grounding must use
    this interpolated surface instead.
    """

    if grid_spacing_m <= 0.0 or terrain_buffer_m < 0.0:
        raise ValueError("terrain grid spacing must be positive and buffer non-negative")
    route_xmin, route_xmax, route_ymin, route_ymax = map(float, map_config["bounds_m"])
    xmin, xmax = route_xmin - terrain_buffer_m, route_xmax + terrain_buffer_m
    ymin, ymax = route_ymin - terrain_buffer_m, route_ymax + terrain_buffer_m
    x = min(xmax, max(xmin, float(x_m)))
    y = min(ymax, max(ymin, float(y_m)))
    cell_x = min(int((x - xmin) // grid_spacing_m), round((xmax - xmin) / grid_spacing_m) - 1)
    cell_y = min(int((y - ymin) // grid_spacing_m), round((ymax - ymin) / grid_spacing_m) - 1)
    x0 = xmin + cell_x * grid_spacing_m
    y0 = ymin + cell_y * grid_spacing_m
    tx = (x - x0) / grid_spacing_m
    ty = (y - y0) / grid_spacing_m
    h00 = terrain_height(map_config, x0, y0)
    h10 = terrain_height(map_config, x0 + grid_spacing_m, y0)
    h01 = terrain_height(map_config, x0, y0 + grid_spacing_m)
    h11 = terrain_height(map_config, x0 + grid_spacing_m, y0 + grid_spacing_m)
    if ty <= tx:
        return h00 + tx * (h10 - h00) + ty * (h11 - h10)
    return h00 + tx * (h11 - h01) + ty * (h01 - h00)


class TerrainRouteGenerator:
    """Generate reproducible routes with slope, obstacle, and zone constraints."""

    def __init__(
        self,
        map_config: dict[str, Any],
        defaults: dict[str, Any],
        obstacles: Iterable[Obstacle2D | dict[str, Any]] = (),
    ) -> None:
        self.map = map_config
        self.defaults = defaults
        self.map_id = str(map_config["id"])
        self.bounds = tuple(map(float, map_config["bounds_m"]))
        self.resolution = float(defaults["grid_resolution_m"])
        self.spacing = float(defaults["resample_spacing_m"])
        self.clearance = float(defaults["robot_clearance_m"])
        self.robot_maximum_slope_deg = float(defaults["maximum_slope_deg"])
        self.target_maximum_slope_deg = float(
            map_config.get("target_maximum_slope_deg", defaults.get("target_maximum_slope_deg", self.robot_maximum_slope_deg))
        )
        self.maximum_slope_deg = min(self.robot_maximum_slope_deg, self.target_maximum_slope_deg)
        self.minimum_length_m = float(defaults["minimum_length_m"])
        self.maximum_length_m = float(defaults["maximum_length_m"])
        self.maximum_attempts = int(defaults["maximum_attempts"])
        self.maximum_turn_deg = float(defaults.get("maximum_turn_deg", 150.0))
        self.obstacles = tuple(
            item if isinstance(item, Obstacle2D) else Obstacle2D(
                obstacle_id=str(item["id"]),
                x_m=float(item["x_m"]),
                y_m=float(item["y_m"]),
                radius_m=float(item["radius_m"]),
            )
            for item in obstacles
        )
        self._cell_valid_cache: dict[tuple[int, int], bool] = {}
        self._cell_height_cache: dict[tuple[int, int], float] = {}
        self._edge_valid_cache: dict[tuple[tuple[int, int], tuple[int, int]], bool] = {}
        self._cell_zone_cost_cache: dict[tuple[int, int], float] = {}
        if self.resolution <= 0.0 or self.spacing <= 0.0 or self.clearance < 0.0:
            raise ValueError("route resolutions must be positive and clearance non-negative")

    def generate(self, seed: int) -> GeneratedTerrainRoute:
        rng = random.Random(seed)
        for _ in range(self.maximum_attempts):
            gate_sets = self.map.get("route_gate_sets", [self.map["route_gates"]])
            anchors = [self._sample_gate(gate, rng) for gate in rng.choice(gate_sets)]
            if not all(self._point_valid(*point) for point in anchors):
                continue
            grid_path: list[tuple[float, float]] = []
            try:
                for start, goal in zip(anchors, anchors[1:]):
                    # Simplify each leg independently so every sampled gate remains a
                    # hard waypoint.  Simplifying the concatenated path could legally
                    # cut across an intermediate gate and collapse a 300 m route into
                    # a much shorter start-to-finish chord.
                    section = self._simplify(self._astar(start, goal))
                    if grid_path:
                        section = section[1:]
                    grid_path.extend(section)
            except RuntimeError:
                continue
            resampled = self._resample(grid_path)
            length = self._length(resampled)
            # Revalidate authored samples.  Each A* section is valid, but joining
            # and resampling multiple gate legs can otherwise cut a narrow trail
            # corner by a fraction of a metre.
            samples_valid = all(self._point_valid(point[0], point[1]) for point in resampled)
            turn_valid = self._maximum_turn_deg(resampled) <= self.maximum_turn_deg + 1e-6
            if (
                samples_valid
                and turn_valid
                and self.minimum_length_m <= length <= self.maximum_length_m
            ):
                return self._summarize(seed, resampled)
        raise RuntimeError(
            f"could not generate a valid {self.map_id} route in {self.maximum_attempts} attempts"
        )

    def _sample_gate(self, gate: Sequence[float], rng: random.Random) -> tuple[float, float]:
        cx, cy, width, height = map(float, gate)
        margin = max(self.clearance, self.resolution)
        return (
            rng.uniform(cx - width * 0.5 + margin, cx + width * 0.5 - margin),
            rng.uniform(cy - height * 0.5 + margin, cy + height * 0.5 - margin),
        )

    def _to_cell(self, point: Sequence[float]) -> tuple[int, int]:
        xmin, _, ymin, _ = self.bounds
        return (
            round((float(point[0]) - xmin) / self.resolution),
            round((float(point[1]) - ymin) / self.resolution),
        )

    def _to_world(self, cell: Sequence[int]) -> tuple[float, float]:
        xmin, _, ymin, _ = self.bounds
        return (xmin + int(cell[0]) * self.resolution, ymin + int(cell[1]) * self.resolution)

    def _point_valid(self, x_m: float, y_m: float) -> bool:
        xmin, xmax, ymin, ymax = self.bounds
        if not (xmin + self.clearance <= x_m <= xmax - self.clearance):
            return False
        if not (ymin + self.clearance <= y_m <= ymax - self.clearance):
            return False
        corridors = self.map.get("navigation_corridors", [])
        if corridors and not any(self._inside_corridor(x_m, y_m, corridor) for corridor in corridors):
            return False
        for zone in self.map.get("challenge_zones", []):
            if not zone["traversable"] and self._inside_zone(x_m, y_m, zone, self.clearance):
                return False
        return all(
            hypot(x_m - obstacle.x_m, y_m - obstacle.y_m)
            >= obstacle.radius_m + self.clearance
            for obstacle in self.obstacles
        )

    @staticmethod
    def _inside_corridor(x_m: float, y_m: float, corridor: dict[str, Any]) -> bool:
        points = corridor["points_m"]
        half_width = float(corridor["width_m"]) * 0.5
        return any(
            TerrainRouteGenerator._point_segment_distance(
                x_m, y_m, float(left[0]), float(left[1]), float(right[0]), float(right[1])
            ) <= half_width
            for left, right in zip(points, points[1:])
        )

    @staticmethod
    def _point_segment_distance(
        px: float, py: float, ax: float, ay: float, bx: float, by: float,
    ) -> float:
        dx, dy = bx - ax, by - ay
        denominator = dx*dx + dy*dy
        if denominator <= 1e-12:
            return hypot(px-ax, py-ay)
        ratio = min(1.0, max(0.0, ((px-ax)*dx + (py-ay)*dy) / denominator))
        return hypot(px-(ax+ratio*dx), py-(ay+ratio*dy))

    @staticmethod
    def _inside_zone(x_m: float, y_m: float, zone: dict[str, Any], margin: float = 0.0) -> bool:
        cx, cy = map(float, zone["center_m"])
        width, height = map(float, zone["size_m"])
        return (
            abs(x_m - cx) <= width * 0.5 + margin
            and abs(y_m - cy) <= height * 0.5 + margin
        )

    def _zone_cost(self, x_m: float, y_m: float) -> float:
        return max(
            [1.0] + [
                float(zone["cost"])
                for zone in self.map.get("challenge_zones", [])
                if zone["traversable"] and self._inside_zone(x_m, y_m, zone)
            ]
        )

    def _cell_valid(self, cell: tuple[int, int]) -> bool:
        if cell not in self._cell_valid_cache:
            self._cell_valid_cache[cell] = self._point_valid(*self._to_world(cell))
        return self._cell_valid_cache[cell]

    def _cell_height(self, cell: tuple[int, int]) -> float:
        if cell not in self._cell_height_cache:
            self._cell_height_cache[cell] = terrain_height(self.map, *self._to_world(cell))
        return self._cell_height_cache[cell]

    def _cell_zone_cost(self, cell: tuple[int, int]) -> float:
        if cell not in self._cell_zone_cost_cache:
            self._cell_zone_cost_cache[cell] = self._zone_cost(*self._to_world(cell))
        return self._cell_zone_cost_cache[cell]

    def _edge_valid(self, left: tuple[int, int], right: tuple[int, int]) -> bool:
        key = (left, right) if left <= right else (right, left)
        if key not in self._edge_valid_cache:
            # Endpoints can both lie inside a curved semantic corridor while a
            # diagonal edge cuts just outside it.  Validate the complete edge so
            # A* cannot corner-cut a road or trail boundary.
            self._edge_valid_cache[key] = self._line_valid(
                self._to_world(left), self._to_world(right)
            )
        return self._edge_valid_cache[key]

    def _segment_obstacle_valid(
        self, start: Sequence[float], end: Sequence[float]
    ) -> bool:
        dx, dy = float(end[0]) - float(start[0]), float(end[1]) - float(start[1])
        denominator = dx * dx + dy * dy
        for obstacle in self.obstacles:
            if denominator <= 1e-12:
                closest_x, closest_y = float(start[0]), float(start[1])
            else:
                ratio = ((obstacle.x_m-float(start[0]))*dx + (obstacle.y_m-float(start[1]))*dy) / denominator
                ratio = min(1.0, max(0.0, ratio))
                closest_x = float(start[0]) + ratio * dx
                closest_y = float(start[1]) + ratio * dy
            if hypot(closest_x-obstacle.x_m, closest_y-obstacle.y_m) < obstacle.radius_m + self.clearance:
                return False
        for zone in self.map.get("challenge_zones", []):
            if zone["traversable"]:
                continue
            cx, cy = map(float, zone["center_m"])
            width, height = map(float, zone["size_m"])
            if self._segment_intersects_box(
                start, end, cx, cy,
                width * 0.5 + self.clearance,
                height * 0.5 + self.clearance,
            ):
                return False
        return True

    @staticmethod
    def _segment_intersects_box(
        start: Sequence[float], end: Sequence[float], cx: float, cy: float,
        half_width: float, half_height: float,
    ) -> bool:
        """Exact segment/AABB test so resampling cannot cut a forbidden corner."""
        x0, y0 = float(start[0]) - cx, float(start[1]) - cy
        dx, dy = float(end[0]) - float(start[0]), float(end[1]) - float(start[1])
        lower, upper = 0.0, 1.0
        for origin, delta, extent in ((x0, dx, half_width), (y0, dy, half_height)):
            if abs(delta) <= 1e-12:
                if abs(origin) <= extent:
                    continue
                return False
            enter = (-extent - origin) / delta
            leave = (extent - origin) / delta
            if enter > leave:
                enter, leave = leave, enter
            lower, upper = max(lower, enter), min(upper, leave)
            if lower > upper:
                return False
        return True

    def _astar(
        self, start_world: tuple[float, float], goal_world: tuple[float, float]
    ) -> list[tuple[float, float]]:
        start, goal = self._to_cell(start_world), self._to_cell(goal_world)
        open_heap: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score = {start: 0.0}
        neighbors = ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1))
        slope_limit = tan(radians(self.maximum_slope_deg))
        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return [self._to_world(cell) for cell in reversed(path)]
            current_world = self._to_world(current)
            current_height = self._cell_height(current)
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                world = self._to_world(neighbor)
                if not self._cell_valid(neighbor) or not self._edge_valid(current, neighbor):
                    continue
                planar = self.resolution * (2 ** 0.5 if dx and dy else 1.0)
                height = self._cell_height(neighbor)
                grade = abs(height - current_height) / planar
                if grade > slope_limit:
                    continue
                step_cost = planar * self._cell_zone_cost(neighbor) * (1.0 + 2.5 * grade)
                tentative = g_score[current] + step_cost
                if tentative >= g_score.get(neighbor, float("inf")):
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                heuristic = hypot(neighbor[0] - goal[0], neighbor[1] - goal[1]) * self.resolution
                heapq.heappush(open_heap, (tentative + heuristic, neighbor))
        raise RuntimeError("no collision-free path between route gates")

    def _line_valid(self, start: Sequence[float], end: Sequence[float]) -> bool:
        if not self._segment_obstacle_valid(start, end):
            return False
        distance = hypot(end[0] - start[0], end[1] - start[1])
        count = max(1, int(distance / (0.5 * self.resolution)))
        previous = (float(start[0]), float(start[1]))
        previous_height = terrain_height(self.map, *previous)
        for index in range(1, count + 1):
            ratio = index / count
            point = (
                float(start[0]) + ratio * (float(end[0]) - float(start[0])),
                float(start[1]) + ratio * (float(end[1]) - float(start[1])),
            )
            if not self._point_valid(*point):
                return False
            height = terrain_height(self.map, *point)
            planar = hypot(point[0] - previous[0], point[1] - previous[1])
            slope = degrees(atan(abs(height - previous_height) / max(planar, 1e-6)))
            if slope > self.maximum_slope_deg:
                return False
            previous, previous_height = point, height
        return True

    def _simplify(self, points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(points) <= 2:
            return list(points)
        output = [points[0]]
        index = 0
        while index < len(points) - 1:
            candidate = len(points) - 1
            while candidate > index + 1 and not self._line_valid(points[index], points[candidate]):
                candidate -= 1
            output.append(points[candidate])
            index = candidate
        return output

    def _resample(self, points: Sequence[tuple[float, float]]) -> list[tuple[float, float, float]]:
        output: list[tuple[float, float, float]] = []
        for start, end in zip(points, points[1:]):
            distance = hypot(end[0] - start[0], end[1] - start[1])
            # ceil guarantees the authored target never jumps farther than the
            # requested spacing, including short diagonal A* remnants.
            count = max(1, ceil(distance / self.spacing))
            for index in range(count):
                ratio = index / count
                x_m = start[0] + ratio * (end[0] - start[0])
                y_m = start[1] + ratio * (end[1] - start[1])
                point = (x_m, y_m, terrain_height(self.map, x_m, y_m))
                if not output or hypot(point[0]-output[-1][0], point[1]-output[-1][1]) > 1e-6:
                    output.append(point)
        x_m, y_m = points[-1]
        output.append((x_m, y_m, terrain_height(self.map, x_m, y_m)))
        return output

    @staticmethod
    def _length(points: Sequence[Sequence[float]]) -> float:
        return sum(
            hypot(right[0] - left[0], right[1] - left[1])
            for left, right in zip(points, points[1:])
        )

    @staticmethod
    def _maximum_turn_deg(points: Sequence[Sequence[float]]) -> float:
        """Return the largest planar heading change between adjacent samples.

        Near-180 degree changes normally mean that a gate was placed at a dead
        end and the generated actor immediately retraces its last step.  Such a
        route is collision-free, but is not a natural walking trajectory.
        """

        maximum = 0.0
        for first, middle, last in zip(points, points[1:], points[2:]):
            incoming = atan2(middle[1] - first[1], middle[0] - first[0])
            outgoing = atan2(last[1] - middle[1], last[0] - middle[0])
            delta = abs((outgoing - incoming + pi) % (2.0 * pi) - pi)
            maximum = max(maximum, degrees(delta))
        return maximum

    def _summarize(
        self, seed: int, points: Sequence[tuple[float, float, float]]
    ) -> GeneratedTerrainRoute:
        slopes, step_heights, elevation_gain = [], [], 0.0
        for left, right in zip(points, points[1:]):
            planar = hypot(right[0]-left[0], right[1]-left[1])
            dz = right[2]-left[2]
            slopes.append(degrees(atan(abs(dz) / max(planar, 1e-6))))
            step_heights.append(abs(dz))
            elevation_gain += max(0.0, dz)
        clearance = min(
            [1.0e9] + [
                hypot(point[0]-obstacle.x_m, point[1]-obstacle.y_m)-obstacle.radius_m
                for point in points for obstacle in self.obstacles
            ]
        )
        visited = tuple(
            zone["id"] for zone in self.map.get("challenge_zones", [])
            if any(self._inside_zone(point[0], point[1], zone) for point in points)
        )
        return GeneratedTerrainRoute(
            map_id=self.map_id,
            seed=seed,
            algorithm=str(self.defaults["algorithm"]),
            points_m=tuple(points),
            length_m=self._length(points),
            elevation_gain_m=elevation_gain,
            maximum_slope_deg=max(slopes, default=0.0),
            maximum_step_height_m=max(step_heights, default=0.0),
            maximum_turn_deg=self._maximum_turn_deg(points),
            minimum_obstacle_clearance_m=clearance,
            challenge_zones_visited=visited,
            target_reachable=max(slopes, default=0.0) <= self.target_maximum_slope_deg + 1e-6,
        )
