"""Actual Isaac Sim backend for the StandardTrackingEnv contract.

Isaac imports are deliberately lazy: AppLauncher must be started by the caller
before :meth:`IsaacTrackingBackend.create` is invoked.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from math import acos, cos, degrees, radians, sin
from pathlib import Path
from typing import Any

import numpy as np
import torch

from isaac_track_lab.control import Go2LocomotionPolicy, POLICY_JOINT_NAMES
from isaac_track_lab.sim2real import (
    D455PostProcessor,
    SensorControlTimeline,
    Sim2RealProfile,
    TemporalCameraVibration,
    build_episode_randomization,
)
from isaac_track_lab.sim2real.events import ScenarioEventRuntime
from isaac_track_lab.task import (
    DisturbanceEvent, TerrainTargetFollower, build_scenario_plan, terrain_mesh_height,
    traversable_feature_surface_height,
)
from isaac_track_lab.task.motion import inside_route_corridor

from .core import (
    ActorObservation,
    BackendTransition,
    EpisodeRequest,
    PlanarVelocityAction,
    PrivilegedSignals,
)


ROOT = Path(__file__).resolve().parents[3]
REPRESENTATIVE_MAPS = ("city_01", "park_01", "forest_trail_01")
ACTOR_FIELDS = ActorObservation.keys()
PRIVILEGED_FIELDS = tuple(PrivilegedSignals.__dataclass_fields__)


@dataclass(frozen=True)
class IsaacBackendOptions:
    map_id: str = "city_01"
    route_index: int = 1
    device: str = "cuda:0"
    num_envs: int = 1
    offline_high_quality: bool = False
    output_root: Path = ROOT / "outputs/sim2real_training"


class IsaacTrackingBackend:
    """Single-environment Isaac implementation with batched sensor tensors.

    ``num_envs`` is retained in the scene and camera configuration for future
    vectorization. Target/event orchestration intentionally rejects values
    other than one until per-environment character Fabric views are validated.
    """

    def __init__(self) -> None:
        raise RuntimeError("use IsaacTrackingBackend.create after AppLauncher starts")

    @classmethod
    def create(
        cls,
        profile: Sim2RealProfile,
        options: IsaacBackendOptions,
        *,
        simulation_app: Any,
    ) -> "IsaacTrackingBackend":
        if options.num_envs != 1:
            raise NotImplementedError(
                "character/event orchestration is validated for one environment; "
                "camera and observation tensors retain a batch dimension for future vectorization"
            )
        self = object.__new__(cls)
        self.profile = profile
        self.options = options
        self.simulation_app = simulation_app
        self._import_isaac()
        self._load_static_inputs()
        self._build_stage_and_scene()
        self.request: EpisodeRequest | None = None
        self.elapsed_s = 0.0
        self.next_camera_s = 0.0
        self.last_processed = None
        self.template_rgb = None
        self.event_runtime = ScenarioEventRuntime(())
        self.event_history: list[dict[str, Any]] = []
        self.randomization = None
        self.timeline = None
        self.sensor = None
        self.vibration = None
        self.last_sensor_metadata: dict[str, Any] = {}
        self.sensor_metadata_history: list[dict[str, Any]] = []
        self.randomization_apply_log: dict[str, Any] = {}
        self._pending_inference_ms: float | None = None
        self.current_active_events: list[tuple[Any, float, Any]] = []
        self.capture_callback = None
        return self

    def set_capture_callback(self, callback: Any | None) -> None:
        """Register a consumer for every 30 Hz degraded RGB-D capture."""

        self.capture_callback = callback

    def _import_isaac(self) -> None:
        import omni.kit.commands
        import omni.usd
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, UsdSkel

        import isaaclab.sim as sim_utils
        from isaaclab.assets import ArticulationCfg, RigidObjectCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sensors import CameraCfg, ContactSensorCfg
        from isaaclab.utils import configclass
        import isaaclab.utils.math as math_utils
        from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
        from isaaclab_physx.renderers import (
            IsaacRtxRendererCfg, IsaacRtxRendererGlobalSettingsCfg,
        )
        from isaaclab_physx.sim.views import FabricFrameView

        self.api = {
            "commands": omni.kit.commands, "usd_context": omni.usd.get_context(),
            "Gf": Gf, "Sdf": Sdf, "Usd": Usd, "UsdGeom": UsdGeom, "UsdLux": UsdLux,
            "UsdPhysics": UsdPhysics, "UsdShade": UsdShade,
            "UsdSkel": UsdSkel,
            "sim_utils": sim_utils, "ArticulationCfg": ArticulationCfg,
            "RigidObjectCfg": RigidObjectCfg, "InteractiveScene": InteractiveScene,
            "InteractiveSceneCfg": InteractiveSceneCfg, "CameraCfg": CameraCfg,
            "ContactSensorCfg": ContactSensorCfg, "configclass": configclass,
            "math_utils": math_utils, "UNITREE_GO2_CFG": UNITREE_GO2_CFG,
            "IsaacRtxRendererCfg": IsaacRtxRendererCfg,
            "IsaacRtxRendererGlobalSettingsCfg": IsaacRtxRendererGlobalSettingsCfg,
            "FabricFrameView": FabricFrameView,
        }

    def _load_static_inputs(self) -> None:
        catalog = json.loads((ROOT / "configs/terrain_dataset_v3_realistic.json").read_text())
        maps = {item["id"]: item for item in catalog["maps"]}
        if self.options.map_id not in maps:
            raise ValueError(f"unknown realistic terrain map {self.options.map_id!r}")
        self.map_config = maps[self.options.map_id]
        self.map_dir = ROOT / "assets/scenes/terrain_dataset_v3_realistic" / self.options.map_id
        self.manifest = json.loads((self.map_dir / "scene_manifest.json").read_text())
        routes = json.loads((self.map_dir / "routes/generated_routes.json").read_text())["routes"]
        if not 1 <= self.options.route_index <= len(routes):
            raise ValueError("route index out of range")
        self.route = routes[self.options.route_index - 1]
        self.route_points = np.asarray(self.route["points_m"], dtype=np.float64)
        self.scenario_config = json.loads((ROOT / "configs/scenarios.json").read_text())
        self.scenarios = {row["id"]: row for row in self.scenario_config["scenarios"]}
        smoke_path = ROOT / "configs/sim2real_smoke_scenario.json"
        self.smoke_scenario_config = json.loads(smoke_path.read_text())
        smoke_scenario = self.smoke_scenario_config["scenario"]
        self.scenarios[smoke_scenario["id"]] = smoke_scenario
        self.locomotion_root = (
            ROOT.parent / "MaSTrack/robot_runtime/models/locomotion_candidates/kaiwu2025_v6-2_124004"
        )

    def _build_stage_and_scene(self) -> None:
        a = self.api
        stage = a["usd_context"].get_stage()
        self.stage = stage
        map_prim = stage.DefinePrim("/World/Map", "Xform")
        map_prim.GetReferences().AddReference(str(self.map_dir / self.manifest["usd"]))
        self._create_lights(stage)
        self._spawn_character(stage, "/World/TargetHuman", str(ROOT / "assets/external/nvidia/people/F_Business_02.usd"))
        self._create_capsule(stage, "/World/TargetCollider", role="target_human_collision")
        for _ in range(20):
            self.simulation_app.update()
        self.target_motion_controller = self._setup_character_animation(
            "/World/TargetHuman", phase_offset=0
        )

        start = self.route_points[0].copy()
        start[2] = terrain_mesh_height(self.map_config, float(start[0]), float(start[1]))
        heading = np.arctan2(
            self.route_points[min(3, len(self.route_points) - 1), 1] - start[1],
            self.route_points[min(3, len(self.route_points) - 1), 0] - start[0],
        )
        go2_cfg = a["UNITREE_GO2_CFG"].replace(prim_path="{ENV_REGEX_NS}/Robot")
        go2_cfg.init_state.pos = (float(start[0]), float(start[1]), float(start[2]) + 0.50)
        go2_cfg.init_state.rot = (
            0.0, 0.0, float(sin(heading / 2)), float(cos(heading / 2))
        )
        go2_cfg.actuators["base_legs"].stiffness = 20.0
        go2_cfg.actuators["base_legs"].damping = 0.5

        renderer = self._renderer_config()
        d455 = self.profile.d455
        camera_cfg = a["CameraCfg"](
            prim_path="{ENV_REGEX_NS}/Robot/base/D455",
            update_latest_camera_pose=True,
            update_period=1.0 / d455.rgb_hz,
            height=d455.height, width=d455.width,
            data_types=["rgb", "distance_to_image_plane"],
            renderer_cfg=renderer,
            spawn=a["sim_utils"].PinholeCameraCfg(
                focal_length=2.34708, horizontal_aperture=3.84, vertical_aperture=2.88,
                focus_distance=400.0, clipping_range=(0.1, 30.0),
                distortion=a["sim_utils"].OpenCvPinholeDistortionCfg(
                    fx=d455.fx, fy=d455.fy, cx=d455.cx, cy=d455.cy,
                    image_size=(d455.width, d455.height), apply_lens_distortion=False,
                ),
            ),
            offset=a["CameraCfg"].OffsetCfg(
                pos=d455.body_relative_translation_m,
                rot=d455.body_relative_quaternion_xyzw_ros,
                convention="ros",
            ),
        )

        configclass = a["configclass"]
        InteractiveSceneCfg = a["InteractiveSceneCfg"]
        ArticulationCfg = a["ArticulationCfg"]
        RigidObjectCfg = a["RigidObjectCfg"]
        ContactSensorCfg = a["ContactSensorCfg"]

        @configclass
        class TrainingSceneCfg(InteractiveSceneCfg):
            robot: ArticulationCfg = go2_cfg
            target_collider: RigidObjectCfg = RigidObjectCfg(
                prim_path="/World/TargetCollider", spawn=None,
            )
            d455 = camera_cfg
            feet_contact = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/.*_foot", update_period=0.005,
                history_length=2, track_air_time=True,
            )
            base_contact = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/base", update_period=0.005,
                history_length=2,
            )

        self.sim = a["sim_utils"].SimulationContext(
            a["sim_utils"].SimulationCfg(dt=0.005, device=self.options.device)
        )
        self.scene = a["InteractiveScene"](
            TrainingSceneCfg(num_envs=self.options.num_envs, env_spacing=4.0)
        )
        self.sim.reset()
        self.robot = self.scene["robot"]
        self.camera = self.scene["d455"]
        self.target_collider = self.scene["target_collider"]
        self.target_visual = a["FabricFrameView"]("/World/TargetHuman", device=self.options.device)
        self.target_visual.get_world_poses()
        self.joint_ids, joint_names = self.robot.find_joints(
            list(POLICY_JOINT_NAMES), preserve_order=True
        )
        if tuple(joint_names) != POLICY_JOINT_NAMES:
            raise RuntimeError(f"Go2 joint order mismatch: {joint_names}")
        self.foot_ids, foot_names = self.robot.find_bodies(".*_foot", preserve_order=True)
        if len(self.foot_ids) != 4:
            raise RuntimeError(f"expected four feet, got {foot_names}")
        self.locomotion = Go2LocomotionPolicy(
            self.locomotion_root / "exported/policy.onnx",
            self.locomotion_root / "params/deploy.yaml",
        )
        self.policy_steps = max(1, round(self.locomotion.step_dt / 0.005))

    def _renderer_config(self) -> Any:
        a = self.api
        if self.options.offline_high_quality:
            spp, gi, reflections, ao = 128, True, True, True
        else:
            spp, gi, reflections, ao = 4, False, False, False
        self.renderer_evidence = {
            "samples_per_pixel": spp,
            "global_illumination": gi,
            "reflections": reflections,
            "shadows": True,
            "ambient_occlusion": ao,
            "sampled_lighting": bool(self.options.offline_high_quality),
            "antialiasing": "TAA",
        }
        return a["IsaacRtxRendererCfg"](
            enable_scene_partitioning=False,
            global_settings=a["IsaacRtxRendererGlobalSettingsCfg"](
                antialiasing_mode="TAA", enable_dl_denoiser=True,
                samples_per_pixel=spp, enable_direct_lighting=True,
                enable_global_illumination=gi, enable_reflections=reflections,
                enable_shadows=True, enable_ambient_occlusion=ao,
                ambient_light_intensity=1.0,
                carb_settings={
                    "/rtx/rendermode": "RaytracedLighting",
                    "/rtx/directLighting/sampledLighting/enabled": bool(self.options.offline_high_quality),
                    "/rtx/hydra/readTransformsFromFabricInRenderDelegate": False,
                },
            ),
        )

    def _spawn_character(self, stage: Any, path: str, asset: str) -> None:
        Sdf = self.api["Sdf"]
        self.api["commands"].execute(
            "CreatePayloadCommand", usd_context=self.api["usd_context"],
            path_to=path, asset_path=asset, prim_path=Sdf.Path.emptyPath,
            instanceable=False, select_prim=False,
        )

    def _setup_character_animation(self, root_path: str, phase_offset: int) -> dict[str, Any]:
        Usd, UsdSkel, Gf = self.api["Usd"], self.api["UsdSkel"], self.api["Gf"]
        root = self.stage.GetPrimAtPath(root_path)
        skelroot = next(
            (prim for prim in Usd.PrimRange(root) if prim.IsA(UsdSkel.Root)), None
        )
        if skelroot is None:
            raise RuntimeError(f"person asset has no UsdSkel root: {root_path}")
        skeleton_prim = next(
            (prim for prim in Usd.PrimRange(skelroot) if prim.IsA(UsdSkel.Skeleton)), None
        )
        if skeleton_prim is None:
            raise RuntimeError(f"person asset has no skeleton: {root_path}")
        skeleton = UsdSkel.Skeleton(skeleton_prim)
        joints = list(skeleton.GetJointsAttr().Get())
        sequence = json.loads(
            (ROOT / "assets/external/nvidia/people/behavior_walk_sequence.json").read_text()
        )
        if sequence["joints"] != [str(joint) for joint in joints]:
            raise RuntimeError("walk sequence joints do not match person skeleton")
        frames = sequence["frames"][:29]
        animation_path = skeleton_prim.GetPath().AppendChild("Sim2RealWalkMotion")
        animation = UsdSkel.Animation.Define(self.stage, animation_path)
        animation.CreateJointsAttr().Set(joints)
        UsdSkel.BindingAPI.Apply(skeleton_prim).CreateAnimationSourceRel().SetTargets(
            [animation_path]
        )
        return {
            "frames": frames, "fps": float(sequence.get("fps", 30.0)),
            "phase_offset": int(phase_offset), "last_index": None,
            "translations": animation.CreateTranslationsAttr(),
            "rotations": animation.CreateRotationsAttr(),
            "scales": animation.CreateScalesAttr(), "Gf": Gf,
        }

    @staticmethod
    def _update_character_animation(
        controller: dict[str, Any], elapsed_s: float, *, playback_rate: float
    ) -> None:
        if playback_rate <= 1e-6 and controller["last_index"] is not None:
            return
        frames = controller["frames"]
        pose_index = (
            int(elapsed_s * controller["fps"] * playback_rate)
            + controller["phase_offset"]
        ) % len(frames)
        if pose_index == controller["last_index"]:
            return
        pose = frames[pose_index]
        Gf = controller["Gf"]
        controller["translations"].Set([Gf.Vec3f(*value) for value in pose["translations"]])
        controller["rotations"].Set([
            Gf.Quatf(value[0], Gf.Vec3f(*value[1:])) for value in pose["rotations"]
        ])
        controller["scales"].Set([Gf.Vec3h(*value) for value in pose["scales"]])
        controller["last_index"] = pose_index

    def _create_capsule(self, stage: Any, path: str, role: str) -> None:
        UsdGeom, UsdPhysics, Sdf = self.api["UsdGeom"], self.api["UsdPhysics"], self.api["Sdf"]
        capsule = UsdGeom.Capsule.Define(stage, path)
        capsule.CreateAxisAttr("Z"); capsule.CreateRadiusAttr(0.30); capsule.CreateHeightAttr(1.25)
        capsule.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        capsule.GetPrim().CreateAttribute("track:role", Sdf.ValueTypeNames.String).Set(role)
        UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(capsule.GetPrim()).CreateKinematicEnabledAttr(True)

    def _create_lights(self, stage: Any) -> None:
        UsdLux = self.api["UsdLux"]
        self.dome_light = UsdLux.DomeLight.Define(stage, "/World/Sim2RealDomeLight")
        self.sun_light = UsdLux.DistantLight.Define(stage, "/World/Sim2RealSunLight")
        self.sun_light.CreateAngleAttr(0.53)

    def reset(self, request: EpisodeRequest) -> BackendTransition:
        request.validate()
        self.request = request
        self.elapsed_s = 0.0
        episode_profile = self.profile.with_seed(request.seed)
        scenario = self.scenarios.get(request.scenario_id)
        if scenario is None:
            scenario = {
                "id": request.scenario_id, "duration_s": request.horizon_s,
                "interference": "none", "distractors": [0, 0],
                "forced_occlusions": [0, 0], "appearance_changes": [0, 0],
                "lighting_changes": [0, 0], "sensor_degradations": [0, 0],
            }
        scenario = dict(scenario)
        scenario["duration_s"] = request.horizon_s
        occlusion_config = self.scenario_config["occlusion_event"]
        disturbance_config = self.scenario_config["disturbance_event"]
        if request.scenario_id == self.smoke_scenario_config["scenario"]["id"]:
            occlusion_config = self.smoke_scenario_config["occlusion_event"]
            disturbance_config = self.smoke_scenario_config["disturbance_event"]
        plan = build_scenario_plan(
            scenario, occlusion_config, request.seed,
            disturbance_config,
            self.scenario_config["long_horizon"]["checkpoint_interval_s"],
        )
        if request.scenario_id == self.smoke_scenario_config["scenario"]["id"]:
            explicit = tuple(
                DisturbanceEvent(
                    kind=row["kind"], variant=row["variant"],
                    start_s=float(row["start_s"]), duration_s=float(row["duration_s"]),
                    recovery=row.get("recovery"),
                )
                for row in self.smoke_scenario_config["explicit_events"]
            )
            plan = replace(plan, disturbance_events=explicit)
        self.scenario_plan = plan
        self.randomization = build_episode_randomization(
            episode_profile, root=ROOT,
            distractor_range=(plan.distractor_count, plan.distractor_count),
            map_id=self.options.map_id,
        )
        self.randomization_apply_log = {}
        self.target_trajectory: list[dict[str, float | bool | str | None]] = []
        self.sensor = D455PostProcessor(episode_profile, self.robot.device)
        self.timeline = SensorControlTimeline(
            episode_profile.section("timing"), request.seed, episode_profile.enabled
        )
        self.vibration = TemporalCameraVibration(
            episode_profile.section("camera_vibration"), request.seed,
            episode_profile.enabled["camera_vibration"],
        )
        self.event_runtime = ScenarioEventRuntime(plan.disturbance_events)
        self._lighting_event_was_active = False
        self._appearance_event_was_active = False
        self.target_motion_time_s = 0.0
        self._spawn_distractors(self.randomization.distractor_assets)
        self._update_character_animation(
            self.target_motion_controller, 0.0,
            playback_rate=self.randomization.target_speed_mps / 0.82,
        )
        for controller in self.distractor_motion_controllers:
            self._update_character_animation(controller, 0.0, playback_rate=1.0)
        self._apply_episode_lighting()
        self._apply_episode_materials()
        self._apply_person_appearance()

        self.target_follower = TerrainTargetFollower(
            self.map_config, self.route["points_m"],
            nominal_speed_mps=self.randomization.target_speed_mps,
            minimum_speed_mps=min(1.0, self.randomization.target_speed_mps),
            maximum_speed_mps=max(1.5, self.randomization.target_speed_mps),
            traversal_speed_mps=min(0.65, self.randomization.target_speed_mps),
            terrain_grid_spacing_m=float(self.manifest["terrain_grid_spacing_m"]),
        )
        target_pose = self.target_follower.reset(3.0)
        self._write_target_pose(target_pose)
        start = self.route_points[0].copy()
        start[2] = terrain_mesh_height(self.map_config, float(start[0]), float(start[1]))
        heading = radians(target_pose.yaw_deg)
        robot_pose = self.robot.data.default_root_pose.torch.clone()
        robot_pose[0, :3] = torch.tensor(
            [target_pose.position_m[0] - 3.0 * cos(heading),
             target_pose.position_m[1] - 3.0 * sin(heading), start[2] + 0.50],
            device=self.robot.device,
        )
        robot_pose[0, 3:] = torch.tensor(
            [0.0, 0.0, sin(heading / 2), cos(heading / 2)], device=self.robot.device
        )
        self.robot.write_root_pose_to_sim_index(root_pose=robot_pose)
        self.robot.write_root_velocity_to_sim_index(
            root_velocity=self.robot.data.default_root_vel.torch.clone()
        )
        joints = self.robot.data.default_joint_pos.torch.clone()
        self.robot.write_joint_position_to_sim_index(position=joints)
        self.robot.write_joint_velocity_to_sim_index(
            velocity=self.robot.data.default_joint_vel.torch.clone()
        )
        self.robot.actuators.target_command.set_position_index(value=joints)
        self.scene.reset(); self.locomotion.reset(); self.sensor.reset()
        self.last_processed = None
        self.template_rgb = None
        self.sensor_metadata_history = []
        # Match the validated showcase: run the zero-command locomotion policy
        # for two seconds so the legs actively establish a standing stance.
        zero = np.zeros(3, dtype=np.float32)
        for step in range(400):
            self._low_level_step(zero, step)
        self.locomotion.reset()
        gravity = self.robot.data.projected_gravity_b.torch[0]
        settled_tilt = degrees(acos(float(torch.clamp(-gravity[2], -1.0, 1.0))))
        settled_root = self.robot.data.root_pos_w.torch[0]
        settled_terrain = terrain_mesh_height(
            self.map_config, float(settled_root[0]), float(settled_root[1])
        )
        self.reset_diagnostics = {
            "settle_duration_s": 400 * self.sim.get_physics_dt(),
            "base_world_z_m": float(settled_root[2]),
            "terrain_height_m": settled_terrain,
            "base_clearance_m": float(settled_root[2]) - settled_terrain,
            "base_tilt_deg": settled_tilt,
        }
        target_pose = self.target_follower.reset(3.0)
        self._write_target_pose(target_pose)
        self.elapsed_s = 0.0
        self.next_camera_s = 1.0 / self.profile.d455.rgb_hz
        # The camera rendered during settling. Publish that already-captured
        # frame far enough in the past that reset returns a causal t=0 sample.
        self._capture_if_new(capture_time_s=-0.1)
        obs = self._actor_observation()
        if self.template_rgb is None:
            self.template_rgb = self._make_template(obs.rgb)
            obs = self._actor_observation()
        return BackendTransition(obs, self._privileged(), 0.0)

    def step(self, action: PlanarVelocityAction) -> BackendTransition:
        if self.request is None or self.timeline is None:
            raise RuntimeError("backend reset must be called before step")
        action.validate()
        self.timeline.schedule_policy_command(
            self.elapsed_s, action.as_tuple(), measured_inference_ms=self._pending_inference_ms
        )
        self._pending_inference_ms = None
        policy_period = 1.0 / float(self.profile.section("timing")["policy_hz"])
        end = self.elapsed_s + policy_period
        while self.elapsed_s + 1e-9 < end:
            command = self.timeline.command(self.elapsed_s)
            self._advance_physics(command)
        return BackendTransition(self._actor_observation(), self._privileged(), self.elapsed_s)

    def set_measured_inference_latency_ms(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("inference latency must be non-negative")
        self._pending_inference_ms = float(value)

    def _advance_physics(self, command: tuple[float, float, float]) -> None:
        dt = self.sim.get_physics_dt()
        step_index = int(round(self.elapsed_s / dt))
        active = self.event_runtime.update(self.elapsed_s)
        self.current_active_events = active
        self._apply_active_events(active)
        camera_due = self.elapsed_s + 1e-9 >= self.next_camera_s
        if camera_due:
            pause_start = self.request.horizon_s * self.randomization.target_pause_start_fraction
            paused = pause_start <= self.elapsed_s < pause_start + self.randomization.target_pause_duration_s
            pose = (
                self.target_follower.hold()
                if paused
                else self.target_follower.step(1.0 / self.profile.d455.rgb_hz)
            )
            pose = self._person_behavior_pose(pose)
            if not paused:
                self.target_motion_time_s += 1.0 / self.profile.d455.rgb_hz
            pose_profile_weight = max(
                (
                    weight for event, weight, _ in self.current_active_events
                    if event.kind == "appearance_change"
                    and event.variant == "pose_profile_change"
                ),
                default=0.0,
            )
            self._update_character_animation(
                self.target_motion_controller, self.target_motion_time_s,
                playback_rate=(
                    pose.speed_mps / 0.82
                    * (1.0 + 0.5 * pose_profile_weight)
                ),
            )
            for controller in self.distractor_motion_controllers:
                self._update_character_animation(
                    controller, self.elapsed_s, playback_rate=1.0
                )
            self._write_target_pose(pose)
            self.target_trajectory.append({
                "time_s": self.elapsed_s,
                "x_m": pose.position_m[0], "y_m": pose.position_m[1],
                "z_m": pose.position_m[2], "yaw_deg": pose.yaw_deg,
                "paused": paused,
                "locomotion_mode": pose.locomotion_mode,
                "traversable_feature_id": pose.traversable_feature_id,
                "walkable_surface_z_m": pose.walkable_surface_z_m,
                "surface_clearance_m": pose.surface_clearance_m,
                "vertical_speed_mps": pose.vertical_speed_mps,
                "collision_penetration_detected": (
                    pose.position_m[2] + 1e-5 < pose.walkable_surface_z_m
                ),
            })
            self._apply_camera_vibration(1.0 / self.profile.d455.rgb_hz)
            self.next_camera_s += 1.0 / self.profile.d455.rgb_hz
        self._low_level_step(np.asarray(command, dtype=np.float32), step_index)
        self.elapsed_s += dt
        if camera_due:
            self._capture_if_new()

    def _low_level_step(self, command: np.ndarray, step_index: int) -> None:
        if step_index % self.policy_steps == 0:
            output = self.locomotion.step(
                self.robot.data.root_ang_vel_b.torch[0].cpu().numpy(),
                self.robot.data.projected_gravity_b.torch[0].cpu().numpy(), command,
                self.robot.data.joint_pos.torch[0, self.joint_ids].cpu().numpy(),
                self.robot.data.joint_vel.torch[0, self.joint_ids].cpu().numpy(),
            )
            self.robot.actuators.target_command.set_position_index(
                value=torch.tensor(output.joint_position_targets[None], device=self.robot.device),
                joint_ids=self.joint_ids,
            )
        self.scene.write_data_to_sim(); self.sim.step(); self.scene.update(self.sim.get_physics_dt())

    def _capture_if_new(self, capture_time_s: float | None = None) -> None:
        assert self.sensor is not None and self.timeline is not None
        rgb = self.camera.data.output["rgb"].torch[0, :, :, :3]
        depth = self.camera.data.output["distance_to_image_plane"].torch[0]
        angular = float(torch.linalg.vector_norm(self.robot.data.root_ang_vel_b.torch[0]))
        linear = float(torch.linalg.vector_norm(self.robot.data.root_lin_vel_w.torch[0, :2]))
        under, depth_dropout, blur_event, shot_event = 0.0, False, 0.0, 0.0
        for event, weight, _ in self.current_active_events:
            if event.kind == "sensor_degradation":
                if event.variant == "temporary_underexposure": under = max(under, weight)
                if event.variant == "depth_dropout": depth_dropout = depth_dropout or weight > 0.5
                if event.variant == "motion_blur": blur_event = max(blur_event, weight)
                if event.variant == "shot_noise": shot_event = max(shot_event, weight)
            if event.kind == "lighting_change" and event.variant == "exposure_shift":
                under = max(under, weight)
        processed = self.sensor.process(
            rgb, depth, motion_intensity=min(1.0, 0.35 * angular + 0.12 * linear + blur_event),
            underexposure_weight=under, force_depth_dropout=depth_dropout,
            shot_noise_weight=shot_event,
        )
        self.last_processed = processed
        self.last_sensor_metadata = processed.metadata
        self.sensor_metadata_history.append(dict(processed.metadata))
        timestamp = self.elapsed_s if capture_time_s is None else float(capture_time_s)
        self.timeline.capture("rgb", timestamp, processed.rgb)
        depth_payload = torch.stack((
            processed.depth_m.nan_to_num(0.0),
            processed.depth_valid_mask.to(torch.float32),
        ), dim=-1)
        self.timeline.capture("depth", timestamp, depth_payload)
        if self.capture_callback is not None:
            self.capture_callback(processed, timestamp, rgb, depth)

    def _actor_observation(self) -> ActorObservation:
        assert self.timeline is not None
        frames = self.timeline.observation(self.elapsed_s)
        if "rgb" not in frames or "depth" not in frames:
            if self.last_processed is None:
                raise RuntimeError("no causal RGB-D frame is available")
            rgb = self.last_processed.rgb
            depth = torch.stack((self.last_processed.depth_m.nan_to_num(0.0),
                                 self.last_processed.depth_valid_mask.float()), dim=-1)
        else:
            rgb, depth = frames["rgb"].payload, frames["depth"].payload
        if self.template_rgb is None:
            template = self._make_template(rgb)
        else:
            template = self.template_rgb
        force = self.scene["feet_contact"].data.net_forces_w
        foot_contacts = (0.0, 0.0, 0.0, 0.0)
        if force is not None:
            norms = torch.linalg.vector_norm(force.torch[0], dim=-1)
            foot_contacts = tuple(float(v >= 5.0) for v in norms[:4])
        return ActorObservation(
            rgb=rgb, target_template_rgb=template,
            joint_position=tuple(map(float, self.robot.data.joint_pos.torch[0, self.joint_ids].tolist())),
            joint_velocity=tuple(map(float, self.robot.data.joint_vel.torch[0, self.joint_ids].tolist())),
            base_angular_velocity=tuple(map(float, self.robot.data.root_ang_vel_b.torch[0].tolist())),
            projected_gravity=tuple(map(float, self.robot.data.projected_gravity_b.torch[0].tolist())),
            foot_contacts=foot_contacts, aligned_depth=depth,
        )

    def _make_template(self, rgb: Any) -> Any:
        image = torch.as_tensor(rgb)
        box = self._projected_target_bbox()
        if box is None:
            h, w = image.shape[:2]; box = (w * 0.4, h * 0.2, w * 0.6, h * 0.85)
        x1, y1, x2, y2 = map(int, box)
        crop = image[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2), :3]
        return crop.clone()

    def _privileged(self) -> PrivilegedSignals:
        target = self.target_pose.position_m
        root = self.robot.data.root_pos_w.torch[0]
        distance = float(np.hypot(float(root[0]) - target[0], float(root[1]) - target[1]))
        box = self._projected_target_bbox()
        area = 0.0 if box is None else max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]) / (
            self.profile.d455.width * self.profile.d455.height
        )
        gravity = self.robot.data.projected_gravity_b.torch[0]
        tilt = degrees(acos(float(torch.clamp(-gravity[2], -1.0, 1.0))))
        contact = self.scene["base_contact"].data.net_normal_forces_w
        base_impulse = 0.0
        if contact is not None:
            base_impulse = float(torch.linalg.vector_norm(contact.torch, dim=-1).max()) * self.sim.get_physics_dt()
        in_bounds = inside_route_corridor(
            (float(root[0]), float(root[1])), self.route_points, 5.0, 1.0
        )
        terrain_z = terrain_mesh_height(
            self.map_config, float(root[0]), float(root[1])
        )
        return PrivilegedSignals(
            target_distance_m=distance, target_bbox_area_ratio=area,
            base_contact_impulse_ns=base_impulse,
            base_height_m=float(root[2]) - terrain_z,
            base_tilt_deg=tilt, in_bounds=in_bounds,
        )

    def _projected_target_bbox(self) -> tuple[float, float, float, float] | None:
        target = torch.tensor(self.target_pose.position_m, device=self.robot.device, dtype=torch.float32)
        target[2] += 0.9
        relative = target - self.camera.data.pos_w.torch[0]
        camera_point = self.api["math_utils"].quat_apply_inverse(
            self.camera.data.quat_w_ros.torch[0], relative
        )
        z = float(camera_point[2])
        if z <= 0.1:
            return None
        d = self.profile.d455
        u = d.fx * float(camera_point[0]) / z + d.cx
        v = d.fy * float(camera_point[1]) / z + d.cy
        width = d.fx * 0.55 / z; height = d.fy * 1.75 / z
        x1, x2 = max(0.0, u - width / 2), min(d.width - 1.0, u + width / 2)
        y1, y2 = max(0.0, v - height / 2), min(d.height - 1.0, v + height / 2)
        if x2 <= x1 or y2 <= y1:
            return None
        depth = self.camera.data.output["distance_to_image_plane"].torch[0]
        ui, vi = int(np.clip(u, 0, d.width - 1)), int(np.clip(v, 0, d.height - 1))
        measured = float(depth[vi, ui])
        if np.isfinite(measured) and measured + 0.35 < z:
            return None
        return x1, y1, x2, y2

    def _write_target_pose(self, pose: Any) -> None:
        self.target_pose = pose
        yaw = radians(pose.yaw_deg + 90.0)
        position = torch.tensor([pose.position_m], dtype=torch.float32, device=self.robot.device)
        orientation = torch.tensor(
            [[0.0, 0.0, sin(yaw / 2), cos(yaw / 2)]],
            dtype=torch.float32, device=self.robot.device,
        )
        with self.target_visual.xform_world_space_writer() as writer:
            writer.set_poses(positions=position, orientations=orientation)
        collider = torch.tensor(
            [[pose.position_m[0], pose.position_m[1], pose.position_m[2] + 0.90,
              0.0, 0.0, sin(radians(pose.yaw_deg) / 2), cos(radians(pose.yaw_deg) / 2)]],
            dtype=torch.float32, device=self.robot.device,
        )
        self.target_collider.write_root_pose_to_sim_index(root_pose=collider)

    def _person_behavior_pose(self, pose: Any) -> Any:
        """Apply one smooth, terrain-grounded lateral/turn maneuver per episode."""

        from dataclasses import replace

        assert self.request is not None and self.randomization is not None
        # A lateral scripted maneuver while a foot is negotiating a narrow
        # curb/stair can place the character beside the collision geometry.
        # Keep the root on the authored route until the transition is clear.
        if pose.locomotion_mode != "walk":
            return pose
        start = 0.55 * self.request.horizon_s
        duration = max(0.75, min(2.5, 0.20 * self.request.horizon_s))
        local = min(1.0, max(0.0, (self.elapsed_s - start) / duration))
        # A half-sine shifts out and returns smoothly without teleportation.
        blend = sin(np.pi * local)
        lateral = self.randomization.target_lateral_shift_m * blend
        yaw_rad = radians(pose.yaw_deg)
        x = pose.position_m[0] - sin(yaw_rad) * lateral
        y = pose.position_m[1] + cos(yaw_rad) * lateral
        base_z = terrain_mesh_height(self.map_config, x, y)
        z = traversable_feature_surface_height(
            self.map_config, x, y, base_height_m=base_z
        )
        return replace(
            pose,
            position_m=(float(x), float(y), float(z + self.target_follower.ground_clearance_m)),
            yaw_deg=float(pose.yaw_deg + self.randomization.target_turn_deg * blend),
            walkable_surface_z_m=float(z),
            surface_clearance_m=0.0,
            vertical_speed_mps=0.0,
        )

    def _apply_camera_vibration(self, dt: float) -> None:
        assert self.vibration is not None and self.randomization is not None
        angular = float(torch.linalg.vector_norm(self.robot.data.root_ang_vel_b.torch[0]))
        force = self.scene["feet_contact"].data.net_forces_w
        contact_fraction = 1.0
        if force is not None:
            contact_fraction = float((torch.linalg.vector_norm(force.torch[0], dim=-1) >= 5.0).float().mean())
        trans, rpy = self.vibration.step(
            dt, base_angular_speed_rps=angular, foot_contact_fraction=contact_fraction
        )
        if not self.profile.enabled["camera_vibration"]:
            return
        bias_t = self.randomization.mount_translation_bias_m
        bias_r = self.randomization.mount_rpy_bias_deg
        rel_pos = torch.tensor(
            [[self.profile.d455.body_relative_translation_m[i] + bias_t[i] + trans[i]
              for i in range(3)]], device=self.robot.device,
        )
        base_pos = self.robot.data.root_pos_w.torch[0:1]
        base_quat = self.robot.data.root_quat_w.torch[0:1]
        world_pos = base_pos + self.api["math_utils"].quat_apply(base_quat, rel_pos)
        nominal_q = torch.tensor(
            [self.profile.d455.body_relative_quaternion_xyzw_ros], device=self.robot.device
        )
        delta = self.api["math_utils"].quat_from_euler_xyz(*[
            torch.tensor([radians(bias_r[i] + rpy[i])], device=self.robot.device)
            for i in range(3)
        ])
        world_q = self.api["math_utils"].quat_mul(base_quat, self.api["math_utils"].quat_mul(nominal_q, delta))
        self.camera.set_world_poses(world_pos, world_q, convention="ros")

    def _spawn_distractors(self, assets: tuple[str, ...]) -> None:
        self.distractor_views = []
        self.distractor_motion_controllers = []
        for index, asset in enumerate(assets):
            path = f"/World/Distractor_{index:02d}"
            if not self.stage.GetPrimAtPath(path):
                self._spawn_character(self.stage, path, asset)
                for _ in range(3): self.simulation_app.update()
            self.distractor_motion_controllers.append(
                self._setup_character_animation(path, phase_offset=3 * (index + 1))
            )
            view = self.api["FabricFrameView"](path, device=self.options.device)
            view.get_world_poses()
            route_index = min(
                len(self.route_points) - 1,
                max(1, round((index + 1) * len(self.route_points) / (len(assets) + 1))),
            )
            base = self.route_points[route_index]
            offset = 1.4 if index % 2 == 0 else -1.4
            prev = self.route_points[max(0, route_index - 1)]
            nxt = self.route_points[min(len(self.route_points) - 1, route_index + 1)]
            heading = float(np.arctan2(nxt[1] - prev[1], nxt[0] - prev[0]))
            x = float(base[0] - sin(heading) * offset)
            y = float(base[1] + cos(heading) * offset)
            z = terrain_mesh_height(self.map_config, x, y)
            position = torch.tensor([[x, y, z]], device=self.robot.device)
            yaw = heading + np.pi / 2.0
            orientation = torch.tensor(
                [[0.0, 0.0, sin(yaw / 2.0), cos(yaw / 2.0)]],
                device=self.robot.device,
            )
            with view.xform_world_space_writer() as writer:
                writer.set_poses(positions=position, orientations=orientation)
            self.distractor_views.append(view)
        self.randomization_apply_log["distractors_spawned"] = len(self.distractor_views)

    def _apply_episode_lighting(self) -> None:
        assert self.randomization is not None
        Gf = self.api["Gf"]
        r = self.randomization
        self.dome_light.CreateTextureFileAttr(r.hdri)
        self.dome_light.CreateIntensityAttr(r.hdri_intensity * 1000.0)
        dome_xform = self.api["UsdGeom"].Xformable(self.dome_light.GetPrim())
        dome_ops = dome_xform.GetOrderedXformOps()
        dome_op = dome_ops[0] if dome_ops else dome_xform.AddRotateZOp()
        dome_op.Set(float(r.hdri_rotation_deg))
        self.sun_light.CreateIntensityAttr(r.sun_intensity * 1000.0)
        self.sun_light.CreateColorAttr(Gf.Vec3f(*r.light_rgb))
        # A larger distant-light angle creates softer shadows. This maps the
        # configured dimensionless strength into an actual RTX shadow change.
        self.sun_light.GetAngleAttr().Set(float(0.2 + (1.0 - r.shadow_strength) * 3.0))
        rotation = self.api["UsdGeom"].Xformable(self.sun_light.GetPrim())
        op = rotation.AddRotateXYZOp() if not rotation.GetOrderedXformOps() else rotation.GetOrderedXformOps()[0]
        op.Set(Gf.Vec3f(90.0 - r.sun_elevation_deg, 0.0, r.sun_azimuth_deg))
        self.randomization_apply_log["lighting"] = r.to_dict() | {"applied": True}

    def _apply_episode_materials(self) -> None:
        assert self.randomization is not None
        if not self.profile.enabled["material_randomization"]:
            self.randomization_apply_log["materials"] = {"applied": False, "reason": "disabled"}
            return
        Gf, Sdf = self.api["Gf"], self.api["Sdf"]
        applied = []
        semantic_tokens = {
            "road": ("asphalt",), "sidewalk": ("brick_pavement", "rock_path"),
            "wall": ("brick_wall", "beige_wall"),
        }
        for semantic, spec in self.randomization.materials.items():
            for prim in self.stage.Traverse():
                path = str(prim.GetPath()).lower()
                if not any(token in path for token in semantic_tokens[semantic]):
                    continue
                file_attr = prim.GetAttribute("inputs:file")
                if file_attr and "diff" in prim.GetName().lower():
                    file_attr.Set(Sdf.AssetPath(spec["albedo"]))
                    scale = prim.GetAttribute("inputs:scale")
                    if scale:
                        tint = spec["tint_rgb"]
                        scale.Set(Gf.Vec4f(float(tint[0]), float(tint[1]), float(tint[2]), 1.0))
                    applied.append(str(prim.GetPath()))
                if "rough" in prim.GetName().lower():
                    scale = prim.GetAttribute("inputs:scale")
                    if scale:
                        rough = max(0.08, float(spec["roughness"]) * (1.0 - 0.65 * float(spec["wetness"])))
                        scale.Set(Gf.Vec4f(rough, rough, rough, 1.0))
                if "uvtransform" in prim.GetName().lower():
                    uv_scale = prim.GetAttribute("inputs:scale")
                    uv_rotation = prim.GetAttribute("inputs:rotation")
                    if uv_scale:
                        scale_value = float(spec["texture_scale"])
                        uv_scale.Set(Gf.Vec2f(scale_value, scale_value))
                    if uv_rotation:
                        uv_rotation.Set(float(spec["texture_rotation_deg"]))
        self.randomization_apply_log["materials"] = {
            "applied": bool(applied), "shader_count": len(applied),
            "shader_paths": applied[:50], "semantic_parameters": self.randomization.materials,
        }

    def _apply_person_appearance(self) -> None:
        assert self.randomization is not None
        if not self.profile.enabled["person_appearance_randomization"]:
            self.randomization_apply_log["person_appearance"] = {
                "applied": False, "reason": "disabled",
            }
            return
        Gf = self.api["Gf"]
        prim = self.stage.GetPrimAtPath("/World/TargetHuman")
        xform = self.api["UsdGeom"].Xformable(prim)
        scale_ops = [op for op in xform.GetOrderedXformOps() if op.GetOpType() == self.api["UsdGeom"].XformOp.TypeScale]
        op = scale_ops[0] if scale_ops else xform.AddScaleOp()
        s = self.randomization.target_scale
        op.Set(Gf.Vec3f(s, s, s))
        colored = 0
        shader_parameters = 0
        for child in self.stage.Traverse():
            if not str(child.GetPath()).startswith("/World/TargetHuman"):
                continue
            if child.IsA(self.api["UsdGeom"].Gprim):
                self.api["UsdGeom"].Gprim(child).CreateDisplayColorAttr([
                    Gf.Vec3f(*self.randomization.target_clothing_rgb)
                ]); colored += 1
            for name in ("inputs:diffuse_color_constant", "inputs:diffuse_tint"):
                attr = child.GetAttribute(name)
                if attr:
                    try:
                        attr.Set(Gf.Vec3f(*self.randomization.target_clothing_rgb))
                        shader_parameters += 1
                    except Exception:
                        pass
            for name in ("inputs:reflection_roughness_constant", "inputs:roughness"):
                attr = child.GetAttribute(name)
                if attr:
                    try:
                        attr.Set(float(self.randomization.target_roughness))
                        shader_parameters += 1
                    except Exception:
                        pass
        self.randomization_apply_log["person_appearance"] = {
            "applied": True, "asset": self.randomization.target_asset,
            "scale": s, "display_color_prim_count": colored,
            "shader_parameter_count": shader_parameters,
            "clothing_rgb": self.randomization.target_clothing_rgb,
            "roughness": self.randomization.target_roughness,
        }

    def _apply_active_events(self, active: list[tuple[Any, float, Any]]) -> None:
        lighting_active = any(event.kind == "lighting_change" for event, _, _ in active)
        if lighting_active or self._lighting_event_was_active:
            self._restore_episode_lighting_base()
        self._lighting_event_was_active = lighting_active
        appearance_active = any(event.kind == "appearance_change" for event, _, _ in active)
        if self._appearance_event_was_active and not appearance_active:
            self._apply_person_appearance()
        self._appearance_event_was_active = appearance_active
        if not active:
            return
        Gf = self.api["Gf"]
        for event, weight, record in active:
            if event.kind == "lighting_change":
                if not self.profile.enabled["lighting_randomization"]:
                    record.skip_reason = "lighting_randomization_disabled"
                    continue
                strength = float(self.profile.section("lighting")["event_strength"].get(event.variant, 0.5))
                if event.variant in ("sun_shadow_transition", "cloud_cover"):
                    self.sun_light.GetIntensityAttr().Set(
                        self.randomization.sun_intensity * 1000.0 * (1.0 - strength * weight)
                    )
                    if event.variant == "sun_shadow_transition":
                        base_angle = 0.2 + (1.0 - self.randomization.shadow_strength) * 3.0
                        self.sun_light.GetAngleAttr().Set(float(base_angle + 4.0 * weight))
                elif event.variant == "backlight":
                    self.sun_light.GetIntensityAttr().Set(
                        self.randomization.sun_intensity * 1000.0 * (1.0 + strength * weight)
                    )
                    sun_xform = self.api["UsdGeom"].Xformable(self.sun_light.GetPrim())
                    sun_xform.GetOrderedXformOps()[0].Set(Gf.Vec3f(
                        90.0 - self.randomization.sun_elevation_deg,
                        0.0,
                        self.randomization.sun_azimuth_deg + 180.0 * weight,
                    ))
                record.asset = str(self.randomization.hdri)
            elif event.kind == "appearance_change":
                if not self.profile.enabled["person_appearance_randomization"]:
                    record.skip_reason = "person_appearance_randomization_disabled"
                    continue
                if event.variant in ("outerwear_change", "partial_clothing_cover"):
                    # A smooth tint change is actor-visible and never changes truth labels.
                    tint = tuple(
                        min(1.0, c * (1.0 - 0.25 * weight) + 0.15 * weight)
                        for c in self.randomization.target_clothing_rgb
                    )
                    for child in self.stage.Traverse():
                        if (
                            str(child.GetPath()).startswith("/World/TargetHuman")
                            and child.IsA(self.api["UsdGeom"].Gprim)
                        ):
                            self.api["UsdGeom"].Gprim(child).GetDisplayColorAttr().Set(
                                [Gf.Vec3f(*tint)]
                            )
                elif event.variant == "scale_change":
                    prim = self.stage.GetPrimAtPath("/World/TargetHuman")
                    xform = self.api["UsdGeom"].Xformable(prim)
                    scale_ops = [
                        op for op in xform.GetOrderedXformOps()
                        if op.GetOpType() == self.api["UsdGeom"].XformOp.TypeScale
                    ]
                    op = scale_ops[0] if scale_ops else xform.AddScaleOp()
                    scale = self.randomization.target_scale * (1.0 + 0.08 * weight)
                    op.Set(Gf.Vec3f(scale, scale, scale))
                elif event.variant != "pose_profile_change":
                    record.skip_reason = f"unsupported_appearance_variant:{event.variant}"
                    continue
                record.asset = self.randomization.target_asset
            elif event.kind == "occlusion":
                if not self.profile.enabled["distractors_occlusion"]:
                    record.skip_reason = "distractors_occlusion_disabled"
                    continue
                if event.variant not in ("crossing_person", "static_occluder"):
                    record.skip_reason = f"unsupported_occluder_variant:{event.variant}"
                    continue
                self._execute_occlusion(event, weight, record)
            elif event.kind == "sensor_degradation":
                if not (
                    self.profile.enabled["rgb_degradation"]
                    or self.profile.enabled["depth_degradation"]
                ):
                    record.skip_reason = "rgb_and_depth_degradation_disabled"
                else:
                    record.asset = "D455PostProcessor.actor_input_path"

    def _restore_episode_lighting_base(self) -> None:
        assert self.randomization is not None
        r = self.randomization
        self.sun_light.GetIntensityAttr().Set(r.sun_intensity * 1000.0)
        self.sun_light.GetAngleAttr().Set(float(0.2 + (1.0 - r.shadow_strength) * 3.0))
        sun_xform = self.api["UsdGeom"].Xformable(self.sun_light.GetPrim())
        sun_xform.GetOrderedXformOps()[0].Set(self.api["Gf"].Vec3f(
            90.0 - r.sun_elevation_deg, 0.0, r.sun_azimuth_deg,
        ))

    def _execute_occlusion(self, event: Any, weight: float, record: Any) -> None:
        if not self.distractor_views:
            record.skip_reason = "no_distractor_asset_spawned"
            return
        robot = self.robot.data.root_pos_w.torch[0]
        target = torch.tensor(self.target_pose.position_m, device=self.robot.device)
        center = 0.5 * (robot + target)
        delta = target[:2] - robot[:2]
        length = torch.linalg.vector_norm(delta).clamp_min(1e-6)
        lateral = torch.tensor([-delta[1], delta[0]], device=self.robot.device) / length
        crossing = (
            (2.0 * weight - 1.0) * 1.2
            if event.variant == "crossing_person" else 0.0
        )
        xy = center[:2] + crossing * lateral
        z = terrain_mesh_height(self.map_config, float(xy[0]), float(xy[1]))
        position = torch.tensor([[float(xy[0]), float(xy[1]), z]], device=self.robot.device)
        yaw = torch.atan2(delta[1], delta[0]) + torch.pi / 2
        orientation = torch.tensor(
            [[0.0, 0.0, float(torch.sin(yaw / 2)), float(torch.cos(yaw / 2))]],
            device=self.robot.device,
        )
        with self.distractor_views[0].xform_world_space_writer() as writer:
            writer.set_poses(positions=position, orientations=orientation)
        record.asset = self.randomization.distractor_assets[0]
        if record.trajectory is None: record.trajectory = []
        record.trajectory.append({"time_s": self.elapsed_s, "x_m": float(xy[0]), "y_m": float(xy[1]), "z_m": z})

    def report(self) -> dict[str, Any]:
        assert self.timeline is not None and self.randomization is not None
        actor = set(ACTOR_FIELDS); privileged = set(PRIVILEGED_FIELDS)
        forbidden = {
            "target_world_pose", "robot_world_pose", "ground_truth_bbox",
            "route_progress", "semantic_segmentation", "simulator_object_id",
        }
        invalid_samples = [
            float(row["depth_invalid_fraction"])
            for row in self.sensor_metadata_history
            if "depth_invalid_fraction" in row
        ]
        dropout_samples = [
            bool(row.get("depth_dropout", False))
            for row in self.sensor_metadata_history
        ]
        ordered_invalid = sorted(invalid_samples)
        if ordered_invalid:
            p95_index = int(round(0.95 * (len(ordered_invalid) - 1)))
            invalid_summary = {
                "samples": invalid_samples,
                "mean": sum(invalid_samples) / len(invalid_samples),
                "p95": ordered_invalid[p95_index],
                "maximum": ordered_invalid[-1],
            }
        else:
            invalid_summary = {
                "samples": [], "mean": None, "p95": None, "maximum": None,
            }
        sensor_statistics = {
            "processed_frame_count": len(self.sensor_metadata_history),
            "depth_invalid_fraction": invalid_summary,
            "depth_dropout_count": sum(dropout_samples),
            "depth_dropout_rate": (
                sum(dropout_samples) / len(dropout_samples)
                if dropout_samples else 0.0
            ),
        }
        mode_counts: dict[str, int] = {}
        for row in self.target_trajectory:
            mode = str(row["locomotion_mode"])
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        target_clearances = [
            float(row["surface_clearance_m"]) for row in self.target_trajectory
        ]
        penetration_count = sum(
            bool(row["collision_penetration_detected"])
            for row in self.target_trajectory
        )
        return {
            "schema_version": 1,
            "backend": "IsaacTrackingBackend",
            "profile": self.profile.name,
            "seed": self.request.seed if self.request else self.profile.seed,
            "map_id": self.options.map_id,
            "route_index": self.options.route_index,
            "offline_high_quality": self.options.offline_high_quality,
            "renderer_settings": self.renderer_evidence,
            "d455_frozen": self.profile.d455.as_dict(),
            "enabled_randomization": self.profile.enabled,
            "sampled_randomization": self.randomization.to_dict(),
            "actual_runtime_application": self.randomization_apply_log,
            "target_behavior": {
                "speed_mps": self.randomization.target_speed_mps,
                "pause_start_fraction": self.randomization.target_pause_start_fraction,
                "pause_duration_s": self.randomization.target_pause_duration_s,
                "lateral_shift_m": self.randomization.target_lateral_shift_m,
                "turn_deg": self.randomization.target_turn_deg,
                "traversal_validation": {
                    "mode_counts": mode_counts,
                    "minimum_surface_clearance_m": (
                        min(target_clearances) if target_clearances else None
                    ),
                    "maximum_absolute_vertical_speed_mps": (
                        max(
                            abs(float(row["vertical_speed_mps"]))
                            for row in self.target_trajectory
                        )
                        if self.target_trajectory else None
                    ),
                    "collision_penetration_count": penetration_count,
                    "skeletal_foot_ik_enabled": False,
                    "validation_scope": "character_root_and_collision_capsule",
                },
                "trajectory": self.target_trajectory,
            },
            "reset_diagnostics": self.reset_diagnostics,
            "events": self.event_runtime.report(),
            "executed_event_count": sum(
                r["apply_count"] > 0 and r["skip_reason"] is None
                for r in self.event_runtime.report()
            ),
            "timing": self.timeline.report(),
            "last_sensor_metadata": self.last_sensor_metadata,
            "sensor_statistics": sensor_statistics,
            "actor_input_fields": sorted(actor),
            "privileged_input_fields": sorted(privileged),
            "truth_leakage_detected": bool(actor & (privileged | forbidden)),
        }

    def close(self) -> None:
        # App lifetime is owned by the launcher/caller.
        pass
