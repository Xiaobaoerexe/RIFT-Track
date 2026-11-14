# IsaacTrackLab

## RIFT-Track integration branch

This directory is the portable source snapshot of the RGB-D person-following
training environment on branch `codex/sim2real-training-env`. It intentionally
does not vendor local virtual environments, Isaac caches, generated USD stages,
licensed NVIDIA character binaries, downloaded Poly Haven assets, checkpoints,
videos, or training outputs.

Run `bash scripts/bootstrap_third_party.sh` to check out the exact Isaac Lab and
Unitree ROS source revisions. Then fetch/build external assets as documented in
`UPLOAD_MANIFEST.md` and `THIRD_PARTY.md`. The frozen D455 calibration and the
`sim2real_v1` runtime are part of this snapshot.

面向 **Unitree As2 + D455** 的城市户外人形目标跟踪仿真工程；当前动力学验证阶段按项目计划暂用 **Unitree Go2** 替代 As2。策略侧不使用语言输入；第一阶段采用：

`RGB + 目标模板 + 本体状态/可选深度 -> 机体速度指令 -> 四足低层运动控制器`

这保留了视觉到追踪动作的端到端训练能力，同时避免一开始就把像素到关节力矩、稳定行走和目标追踪三个难题绑在一起。仿真真值、语义分割和目标位姿只供训练 critic、奖励与评测使用，不进入部署时 actor。

## 当前状态

- 已锁定 Isaac Sim 6.1 + Isaac Lab `v3.0.0-EA` + Python 3.12。
- 已拉取固定版本的 Isaac Lab 和 Unitree 官方 `as2_description`。
- 305.176 m 场景已确立为唯一任务基线；旧五分钟最小城市路线及历史输出已移除。
- 已给出失败/成功判据、观测、动作、随机化和资产接入清单，并新增标准 `reset/step` 训练环境契约。
- 已将生产场景升级为 116 m × 102 m 的交错高清街区：305.176 m 路线、4 次主要转向、3 次障碍绕行、2K PBR/HDRI 和 91 个场景碰撞体。
- 当前任务只启用 1 个高质量目标人物、无干扰人物；目标以 1.25 m/s 自然行走，并带独立运动学胶囊碰撞体。
- D455 标称参数均为已实测冻结基准：640×480、fx/fy=391.18、cx=321.78、cy=239.94、机身相对平移 `[0.32, 0, 0.32]` m、ROS xyzw 四元数 `[0.54167522, -0.54167522, 0.45451948, -0.45451948]`、光心高 0.66 m、俯视 10°、RGB/Depth 30 Hz。
- 五类失败、回合成功、路径越界与找回指标已实现；含 sim-to-real 流水线在内的 61 项自动化测试通过。
- 已用真实 RTX RGB 流做过“实体遮挡—目标丢失—重新出现”诊断；遮挡资产已从当前默认场景移除。
- 已把正式展示的 `showcase_hybrid` 与不允许路线方位/真值重置的 `pure_e2e` 命令路径分离；纯模式 4 s Isaac 冒烟测试通过。
- 已接入 MaST `RGB + 初始模板 -> bbox/confidence`和CUDA ONNX推理。正式路线展示还使用预定目标路线方位前馈，并在无干扰人物前提下使用紫色背心检测重新初始化MaST；这两项必须与“纯RGB+模板 actor”指标分开报告。
- 已编译基于 Unitree 官方 SDK2 `a2::SportClient` 的实机速度桥；仿真视觉验收使用相同 `(vx, vy, yaw_rate)` 契约的平面速度替身，不等同于步态动力学验证。
- 已将 MaSTrack 项目现有 `kaiwu2025_v6-2_124004` Go2 ONNX 步态策略接入 Isaac Lab；执行真实关节/接触动力学，不再使用根节点平移替身。
- 已修复城市道路碰撞网格面绕序；不再使用隐藏支撑面。Go2基座距路面净高0.377 m，原生道路1.25 m/s固定指令实测1.240 m/s。
- 已将动态人物和胶囊从运行时USD位姿写入改为Fabric/PhysX张量更新，避免全场景重同步拖慢关节步态。
- 244.2 s正式回归已完整通过305.176 m：跟踪覆盖率99.10%，4次短暂缺失全部重捕获，跌倒/强碰撞/越界均为0，机器人平均平面速度1.230 m/s；清单与视频哈希已冻结为 `baseline_305m_v1`。
- 三段网络已具备独立冻结和端到端解冻接口。当前305 m默认场景仍不注入干扰或遮挡；本次只按用户明确指示新增独立的全地形地图批次，一小时实跑和任何动态事件仍需后续具体指示。
- 已按用户本阶段指示新增独立的户外全地形批次：森林、峡谷、雪坡、湿地、沙丘和海岸 6 张 180 m × 180 m 地图。每图包含 2K PBR/HDRI、显式摩擦/碰撞、静态障碍和带种子的避障轨迹生成器；这不修改冻结的 305 m 城市基线，也不自动加入遮挡事件或干扰人物。
- 已生成真实规划 v3 城市、公园、森林泥土山径数据集：每类 20 张、每图 20 条目标可达路线，共 60 张 USDC 和 1200 条路线。城市含道路/人行道/路缘，公园含真实湖盆/丘陵，森林含密集树木/岩石/倒木及可跨越低障碍；碰撞、摩擦、语义路径、湖岸净空、坡度和轨迹唯一性均已全量通过。检查方法见 `docs/REALISTIC_TERRAIN_DATASET_V3.md`。
- 已新增真实 Isaac `IsaacTrackingBackend` 与独立 `sim2real_v1` profile：D455 RGB-D 退化、时间相关振动、光照/材质/人物/遮挡随机化和传感器—推理—动作延迟全部接入 actor 实际输入或 Isaac USD/PhysX 运行时；`sim2real_nominal` 全关闭，冻结 showcase 默认不变。
- 已生成 city/park/forest 三场景 128 spp 离线 RGB-D 增广批次，逐帧同时保存原始 RTX、actor 输入、原始/退化深度、有效 mask 和时间戳；清单位于 `outputs/offline_high_quality_rgbd/sim2real_v1_seed10000_20261113T134524Z/manifest.json`。
- Isaac Sim 6.1 已通过 GPU/PhysX 验证，As2 已导入 USD；商业/受许可约束的城市、人物资产尚未选择。

## 快速检查

```bash
cd /home/abc/Documents/视觉跟踪/IsaacTrackLab
.venv/bin/python scripts/preflight.py
PYTHONPATH=source .venv/bin/python -m isaac_track_lab.spec
PYTHONPATH=source .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=source .venv/bin/python scripts/build_scenario_plans.py
python scripts/validate_baseline_305m.py
python scripts/validate_recovery_gate.py
PYTHONPATH=source python -m unittest tests.test_terrain_routes -v
```

Sim-to-real 实际 Isaac smoke 与三场景离线增广（从 Isaac Lab Python 环境执行）：

```bash
OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=source third_party/IsaacLab/.venv/bin/python \
  scripts/run_sim2real_training.py --training-profile sim2real_v1 --seed 1234 \
  --map city_01 --scenario-id smoke_sim2real_events --duration-s 1.0 --device cuda:0

OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=source third_party/IsaacLab/.venv/bin/python \
  scripts/render_three_scene_offline_rgbd.py --training-profile sim2real_v1 \
  --seed 1234 --duration-s 5.0 --device cuda:0
```

户外全地形批量构建和随机轨迹数据生成（从 Isaac Lab 环境执行）：

```bash
cd /home/abc/Documents/视觉跟踪/IsaacTrackLab/third_party/IsaacLab
OMNI_KIT_ACCEPT_EULA=yes PYTHONPATH=/home/abc/Documents/视觉跟踪/IsaacTrackLab/source \
  uv run --extra isaacsim python /home/abc/Documents/视觉跟踪/IsaacTrackLab/scripts/build_outdoor_map_batch.py
OMNI_KIT_ACCEPT_EULA=yes PYTHONPATH=/home/abc/Documents/视觉跟踪/IsaacTrackLab/source \
  uv run --extra isaacsim python /home/abc/Documents/视觉跟踪/IsaacTrackLab/scripts/generate_outdoor_routes.py --count 16
OMNI_KIT_ACCEPT_EULA=yes PYTHONPATH=/home/abc/Documents/视觉跟踪/IsaacTrackLab/source \
  uv run --extra isaacsim python /home/abc/Documents/视觉跟踪/IsaacTrackLab/scripts/validate_outdoor_map_batch.py
```

交错高清任务城市场景的路线、引用、摩擦和碰撞检查（从 Isaac Lab 环境执行）：

```bash
cd /home/abc/Documents/视觉跟踪/IsaacTrackLab/third_party/IsaacLab
uv run --extra isaacsim python /home/abc/Documents/视觉跟踪/IsaacTrackLab/scripts/validate_hd_route_city.py
OMNI_KIT_ACCEPT_EULA=yes PYTHONPATH=/home/abc/Documents/视觉跟踪/IsaacTrackLab/source \
  uv run --extra isaacsim python /home/abc/Documents/视觉跟踪/IsaacTrackLab/scripts/render_hd_route_city_preview.py \
  --device cuda:0 --output-dir outputs/hd_route_city_preview
```

305 m Go2 + MaST 闭环测试并录制1080p视频（从 Isaac Lab 环境执行）：

```bash
cd /home/abc/Documents/视觉跟踪/IsaacTrackLab/third_party/IsaacLab
OMNI_KIT_ACCEPT_EULA=yes PYTHONPATH=/home/abc/Documents/视觉跟踪/IsaacTrackLab/source \
  uv run --extra isaacsim python /home/abc/Documents/视觉跟踪/IsaacTrackLab/scripts/render_city_go2_showcase.py \
  --device cuda:0 --duration-s 244.2 --target-speed-mps 1.25 \
  --control-mode showcase_hybrid \
  --video-fps 5 --render-spp 16 --video-width 1920 --video-height 1080 \
  --output-dir outputs/baseline_305m/closed_loop --output-stem baseline_305m_official
```

这条命令复现的是带路线方位前馈和紫色背心找回的冻结展示，不是纯端到端成绩。检查无真值 actor 路径时使用 `--control-mode pure_e2e`，结果 JSON 必须同时满足：

```text
control_mode = pure_e2e
actor_truth_leakage_guard = true
privileged_actor_inputs = []
appearance_reinitializations = 0
appearance_recovery = disabled
```

As2 物理烟雾测试（从 Isaac Lab 环境执行）：

```bash
cd /home/abc/Documents/视觉跟踪/IsaacTrackLab/third_party/IsaacLab
OMNI_KIT_ACCEPT_EULA=yes PYTHONPATH=/home/abc/Documents/视觉跟踪/IsaacTrackLab/source \
  uv run --extra isaacsim python /home/abc/Documents/视觉跟踪/IsaacTrackLab/scripts/smoke_as2.py \
  --steps 1000 --device cuda:0
```

编译实机 As2 官方 SDK2 速度桥：

```bash
cd /home/abc/Documents/视觉跟踪/IsaacTrackLab
/home/abc/Documents/视觉跟踪/EVT-Bench/.conda/bin/cmake -S deployment -B build/deployment
/home/abc/Documents/视觉跟踪/EVT-Bench/.conda/bin/cmake --build build/deployment -j
build/deployment/as2_sport_bridge NETWORK_INTERFACE [/tmp/mastrack_motion_intent.sock]
```

## 目录

- `configs/task.json`：任务、观测、动作和终止条件。
- `configs/baseline_305m_v1.json`：冻结305 m展示的文件哈希、指标和局限。
- `configs/training_environment.json`：纯 actor/展示模式边界和阶段门。
- `configs/outdoor_map_batch.json`：6 张户外地图、摩擦、障碍、挑战区和轨迹门配置。
- `configs/scenarios.json`：5分钟/1小时 × 三种干扰等级。
- `configs/sim2real_nominal.json` / `configs/sim2real_v1.json`：全关闭基准与训练随机化 profile。
- `configs/assets.json`：机器人、城市、人物和遮挡物资产状态。
- `docs/ENVIRONMENT_SPEC.md`：完整环境设计。
- `docs/IMPLEMENTATION_PLAN.md`：从资产导入到训练/评测的实施顺序。
- `docs/HD_ROUTE_CITY_MANUAL_CHECK.md`：305 m 交错路线、碰撞、摩擦、人物和 D455 的手动检查步骤。
- `docs/GO2_LOCOMOTION_ENV_DIAGNOSIS.md`：脱离视觉任务的Go2固定指令、摩擦扫描、步幅和城市道路碰撞诊断。
- `docs/TRAINING_ENVIRONMENT_STAGES.md`：端到端训练环境的阶段状态和逐项检查方法。
- `docs/OUTDOOR_ALL_TERRAIN_BATCH.md`：全地形地图批量构建、随机轨迹与 Isaac Sim 手动检查。
- `docs/REALISTIC_TERRAIN_DATASET_V3.md`：真实规划 v3 的 60 张地图、1200 条路线、自动验收、二维图集和 Isaac Sim 逐图检查方法。
- `scripts/browse_terrain_dataset.py`：一次启动 Isaac Sim 后用按钮或方向键逐图、逐路线查看三维场景，并切换碰撞代理和相机。
- `scripts/run_sim2real_training.py`：真实 Isaac 后端训练/smoke 入口，支持 profile、seed 和九类消融开关。
- `scripts/render_three_scene_offline_rgbd.py`：city/park/forest 非覆盖式 128 spp 离线 RGB-D 增广。
- `docs/SIM2REAL_V1.md`：运行时模块、证据、命令和当前资产限制。
- `assets/scenes/outdoor_batch`：全地形 USD、场景清单和可复现轨迹数据。
- `outputs/baseline_305m/closed_loop`：正式闭环结果。
- `outputs/locomotion_benchmark`：独立运控JSON、50 Hz时序CSV和汇总表。
- `outputs/baseline_305m/diagnostics`：出生高度、位移与原始/预处理D455对照。
- `outputs/baseline_305m/scene_previews`：场景、路线和障碍物静态验收图。
- `third_party/IsaacLab`：固定到 `v3.0.0-EA`。
- `third_party/unitree_ros`：Unitree 官方 As2 URDF/mesh。
- `third_party/unitree_sdk2`：Unitree 官方 As2 实机控制 SDK。

阈值目前是工程初值，带 `calibration_required` 的碰撞、跌倒和可见性阈值必须通过 As2 实机参数与仿真基线跑数校准后冻结。
