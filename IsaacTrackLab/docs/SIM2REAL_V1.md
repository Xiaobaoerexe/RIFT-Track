# Sim-to-real v1 运行时

`sim2real_v1` 是独立训练 profile，不改变 `baseline_305m_v1` 或 showcase 默认行为。`sim2real_nominal` 关闭全部九类随机化，用于冻结环境对照。D455 标称内外参来自 `configs/d455_sim_calibration.json`，加载时逐值校验，不允许 profile 覆盖。

## 实际接入路径

- `IsaacTrackingBackend` 加载真实地图、碰撞、Go2、RTX D455、人物和低层 ONNX 步态，执行 200 Hz 物理、50 Hz 低层控制、30 Hz RGB-D 和 10 Hz actor 闭环。
- actor 只收到 RGB、模板、可选的 Depth+有效 mask 和本体状态。世界位姿、真值框、距离与边界只进入奖励、终止和评测。
- GPU tensor 后处理覆盖深度裁剪/噪声/量化/空洞/飞点/整帧 dropout/错位，以及 RGB 去 firefly、shot/read noise、曝光、gamma、白平衡、饱和度和运动模糊。
- 带时间戳的因果队列覆盖 RGB/Depth 独立采集和可用时间、Depth 额外延迟、推理/命令/执行延迟、丢帧重复旧帧、动作保持和错过策略更新。
- 每回合实际写入 HDRI、太阳、阴影、语义材质、人物尺度/材质与动画；事件在物理步上连续插值并记录资产、时间、次数和遮挡轨迹。

未由实测支持的范围均在 profile 中标为 `engineering_initial_not_measured`，应在采集真实 D455 与机器人日志后再校准。

## 命令

```bash
OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=source third_party/IsaacLab/.venv/bin/python \
  scripts/run_sim2real_training.py --training-profile sim2real_v1 --seed 1234 \
  --map city_01 --route 1 --scenario-id short_distractors_occlusion \
  --duration-s 300 --device cuda:0
```

任一模块可用 `--disable-rgb-degradation`、`--disable-depth-degradation`、`--disable-camera-vibration`、`--disable-lighting-randomization`、`--disable-material-randomization`、`--disable-person-appearance-randomization`、`--disable-distractors-occlusion`、`--disable-sensor-latency` 或 `--disable-action-latency` 消融。

短时事件验收报告：`outputs/sim2real_training/smoke_events_final_v3.json`。可用
`python scripts/validate_sim2real_report.py <report.json> --require-all-events`
自动检查冻结 D455、因果时间戳、逐帧延迟、Depth 无效率、事件证据和 actor 真值泄漏。
三场景 128 spp 离线增广清单：`outputs/offline_high_quality_rgbd/sim2real_v1_seed10000_20261113T134524Z/manifest.json`。

## 当前资产限制

本地许可清晰的人体网格目前只有 `F_Business_02.usd`，因此“多外观”已实现为尺度、服装纹理着色、粗糙度、颜色和动作相位变化，不宣称已有多个身份网格。人物骨骼采用本地 walk 序列；rig 专用足部 IK 尚未验收。正式训练前仍需用实测数据校准所有工程初值，并扩充许可清晰的人体身份资产。
