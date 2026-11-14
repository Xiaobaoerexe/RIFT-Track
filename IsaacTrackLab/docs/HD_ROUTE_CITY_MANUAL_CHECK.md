# 交错街道 305 m 任务场景：手动检查指南

## 已接入的生产场景

当前任务脚本实际使用：

`/home/abc/Documents/视觉跟踪/IsaacTrackLab/assets/scenes/hd_route_city/hd_route_city.usda`

`hd_route_city.usda` 是当前唯一生产与任务基线。旧直街和五分钟最小城市基线已移除。以后手工改生产场景，应改此文件或它的构建器，而不是再打开 `World_CityDemopack.usd`。

## 一键自动验收

```bash
cd /home/abc/Documents/视觉跟踪/IsaacTrackLab/third_party/IsaacLab
uv run --extra isaacsim python \
  /home/abc/Documents/视觉跟踪/IsaacTrackLab/scripts/validate_hd_route_city.py
```

正确结果以 `HD_ROUTE_CITY_VALIDATION_OK` 开头，并应包含：

- `route_m=305.176`
- `speed_mps=1.25`
- `waypoints=27`
- `major_turns=4`
- `obstacle_deviations=3`
- `colliders=91`

## 在 Isaac Sim 中手动检查

1. 打开 `hd_route_city.usda`，确认 Stage 根节点是 `/HDStreet`。
2. 在 **Layer** 面板确认场景单位为 1 m、Up Axis 为 Z。
3. 展开 `/HDStreet/Ground`：应看到 3 条横向道路和 3 条纵向道路。
4. 展开 `/HDStreet/Route`：应有 `Waypoint_00` 到 `Waypoint_26`。选择 `/HDStreet/Route`，Property 中应看到：
   - `route:length_m = 305.175792...`
   - `route:target_speed_mps = 1.25`
   - `route:major_turn_count = 4`
5. 展开 `/HDStreet/Props`，确认障碍物视觉模型存在；再展开 `/HDStreet/Collision`，确认每个障碍物都有独立低成本碰撞代理。
6. 在右上角眼睛菜单打开 **Show By Type > Physics > Colliders**。道路、街区、建筑、屋顶、灯杆、树与 6 个障碍物都应显示碰撞轮廓。
7. 选择任一道路，例如 `/HDStreet/Ground/Road_H_0`：
   - 应有 `Physics Collision API`；
   - `material:binding:physics` 应指向 `/HDStreet/Looks/RoadPhysics`；
   - 物理材质静摩擦 0.95、动摩擦 0.82、恢复系数 0.02。
8. 选择 `BarrierRoute`、`CoveredCarRoute` 与 `PlanterRoute`。路线分别在三个位置横向偏移，不能穿过其碰撞代理。

## 查看路线和画面

```bash
cd /home/abc/Documents/视觉跟踪/IsaacTrackLab/third_party/IsaacLab
OMNI_KIT_ACCEPT_EULA=yes \
PYTHONPATH=/home/abc/Documents/视觉跟踪/IsaacTrackLab/source \
uv run --extra isaacsim python \
  /home/abc/Documents/视觉跟踪/IsaacTrackLab/scripts/render_hd_route_city_preview.py \
  --device cuda:0 --output-dir outputs/baseline_305m/scene_previews
```

重点查看 `route_plan.png`、`route_start.png`、`first_turn.png`、`covered_car_deviation.png` 和 `aerial_city.png`。本机 RTX 的 DLSS/NGX 当前不可用，原始实时帧仍可能出现颗粒；验收输出会做中值去火花处理。它不改变 D455 送入跟踪器前的几何投影。

## 目标人物与动态检查

正式闭环使用 `configs/task.json`。305 m 运行若触发目标过远、目标丢失、强碰撞、跌倒或越界，会立即停止；因此“跑满路线”和“完整场景巡检”不是同一个概念。

完整路线约需 `305.176 / 1.25 = 244.14 s`：

```bash
cd /home/abc/Documents/视觉跟踪/IsaacTrackLab/third_party/IsaacLab
OMNI_KIT_ACCEPT_EULA=yes \
PYTHONPATH=/home/abc/Documents/视觉跟踪/IsaacTrackLab/source \
uv run --extra isaacsim python \
  /home/abc/Documents/视觉跟踪/IsaacTrackLab/scripts/render_city_go2_showcase.py \
  --device cuda:0 --duration-s 244.2 --target-speed-mps 1.25 \
  --video-fps 5 --render-spp 16 --video-width 1920 --video-height 1080 \
  --output-dir outputs/baseline_305m/closed_loop \
  --output-stem baseline_305m_official
```

需要在 Isaac Sim 窗口中实时观看时增加 `--gui`。正式结果必须读取输出 JSON 的 `state`、`failure_reason` 和 `elapsed_s`，不能仅凭 MP4 是否生成来判定通过。

快速检查第一个转向和车辆绕行：

```bash
OMNI_KIT_ACCEPT_EULA=yes \
PYTHONPATH=/home/abc/Documents/视觉跟踪/IsaacTrackLab/source \
uv run --extra isaacsim python \
  /home/abc/Documents/视觉跟踪/IsaacTrackLab/scripts/render_city_go2_showcase.py \
  --device cuda:0 --duration-s 18 --route-start-distance-m 95 \
  --video-fps 5 --output-dir outputs/baseline_305m/diagnostics \
  --output-stem turn_obstacle_check
```

人物视觉骨架使用 NVIDIA Behavior walk 循环，并按 1.25 m/s 匹配播放；根节点沿等弧长路线移动，朝向按速度方向限速转动。另有一个不可见的运动学胶囊碰撞体跟随人物，场景内没有其他干扰人物。

## D455 参数如何核对

仿真已采用随附 Jetson 项目中 SafeFlow 正在使用的参数：640×480、`fx=fy=391.18`、`cx=321.78`、`cy=239.94`。安装光心高度 0.66 m、俯视 10°来自部署文档。

`configs/d455_intrinsics.json` 明确写有 `PLACEHOLDER ONLY`，所以其中 386 px 焦距没有使用。随附资料没有给出完整机身到相机的实测 6DoF 矩阵：当前前向偏移 0.32 m 沿用原仿真假设，z 偏移由 0.66 m 光心高度减去 Go2 约 0.34 m 基座高度得到。实机标定完成前，不应把该 x 偏移写成“已标定”。完整记录见 `configs/d455_sim_calibration.json`。

## Go2 出生高度和移动检查

道路顶面是 `z=0.20 m`，Go2初始根节点必须使用 `SURFACE_Z + 0.50 m`。生产脚本直接使用修复后的原生道路碰撞网格，不再创建隐藏支撑平面。验证器会检查可行驶网格的三角形几何法向必须朝上。启动日志应包含：

- `base_height_m` 约 0.577 m；
- `clearance_above_road_m` 约 0.377 m；
- JSON 中 `robot_forward_progress_m` 必须大于 0。

修复后的独立原生道路诊断中，6 s测量窗向前移动7.413 m，平均速度1.240 m/s，证明出生点与道路碰撞已修复。

## 画面噪声与视觉模型

颗粒在编码前的原始D455 PNG中已经存在，不是MP4压缩导致。主要来源是2K PBR材质高频纹理、实时RTX采样与640×480缩放的混叠；当前主机的DLSS/NGX又不可用。这些人工颗粒会影响小目标边缘和模板相似度。

脚本现在会同时保存 `*_d455_raw.png` 和 `*_d455_tracker_input.png`；后者是送入MaST前的单次3×3去脉冲结果。展示视频使用1080p第三人称相机、5×5去脉冲加轻度锐化和CRF 16编码；D455和展示链不混用。

## 已知动态边界

正式244.2 s回归已完整通过305.176 m：`state=success`，跟踪覆盖率99.10%，4次短暂缺失全部重捕获，平均平面速度1.230 m/s，最终跟随距离3.150 m，跌倒、强碰撞和越界均为0。人物根节点和胶囊已改用Fabric/PhysX张量位姿，没有直接改写Go2根节点。由于本场景明确无干扰人物，MaST发生IoU漂移时会用目标紫色背心先验重新初始化；JSON中的`appearance_reinitializations`必须保留用于透明审计，不得把本结果当作无外观恢复的纯MaST指标。
