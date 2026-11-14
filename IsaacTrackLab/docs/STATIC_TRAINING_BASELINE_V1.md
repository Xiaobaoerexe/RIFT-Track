# 静态训练基线 v1

本阶段没有扩展地图，没有加入遮挡、干扰人或动态事件。它把现有 60 张城市、公园和森林地图固化为一个可重复检查的训练基线。

## 本次修正

- 公园路线不再把支路尽头作为必须到达的门点，消除“到点后立即接近 180° 原路折返”。
- 路线生成器新增最大瞬时转角门槛，不合格路线会自动重试。
- 新增人物根节点运动器：速度保持在 1.0–1.5 m/s，转弯时在区间内减速，用前视方向平滑朝向，并每帧按 USD 实际三角碰撞面重新计算高度。
- 新增统一静态验收：D455 冻结参数、碰撞数量、摩擦系数、路线贴地误差、转角、人物速度和角速度同时检查。

## 一键验收

在项目根目录执行：

```bash
env PYTHONPATH=source third_party/IsaacLab/.venv/bin/python scripts/validate_realistic_terrain_dataset.py
env PYTHONPATH=source third_party/IsaacLab/.venv/bin/python scripts/validate_static_training_baseline.py
```

两条命令都必须输出 `PASS`。详细结果位于：

- `outputs/terrain_dataset_v3_realistic/quality/quality_report.json`
- `outputs/terrain_dataset_v3_realistic/static_baseline_report.json`

## 逐张地图和路线检查

```bash
env PYTHONPATH=source third_party/IsaacLab/.venv/bin/python scripts/browse_terrain_dataset.py --config configs/terrain_dataset_v3_realistic.json --scene-root assets/scenes/terrain_dataset_v3_realistic
```

检查面板现在会显示路线长度、最大坡度、最大转角和最小障碍物间距。重点检查：

1. 路线是否贴合道路或小径，没有穿过湖泊、建筑、树木和大石头。
2. 公园路线在端点附近是否仍出现立即原路折返。
3. 路线的高度是否跟随山丘和地形起伏。
4. 切换碰撞显示后，障碍物碰撞代理是否与可见模型基本对齐。

## 当前不能误报为已完成的项目

- D455 内参、完整安装平移/四元数、光心高度、俯角和名义频率均为已实测冻结值；随机化只作用于观测退化或冻结外参之上的小扰动。
- 人物根节点已贴地，但骨骼足部 IK 还没有针对当前人物 rig 完成验收。
- Go2 在 60 张地图上的动力学通过率、足端滑移和连续一小时闭环仍未验证。
