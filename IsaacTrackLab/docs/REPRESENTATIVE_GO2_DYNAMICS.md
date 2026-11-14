# 代表地图 Go2 动力学验收

这是运控与环境适配性测试，不是视觉跟踪或端到端任务测试。路线几何仅用作特权真值的诊断速度指令。

## 已解决的问题

- 240 m × 240 m 单块、约 11.5 万三角形的地形碰撞网格在当前 PhysX 后端中未成功创建运行时 shape，因此 Go2 会直接穿过地面。
- 视觉地形保留为完整网格，物理地形改为 40 m 瓦片分别烹饪。全部 60 张地图已重建并通过结构验收。
- 人物根节点贴地、Go2 初始高度和足端穿透度量现在使用与 USD 一致的分片平面三角插值，而不是连续解析地形高度。
- 公园湖岸石块的非等比视觉缩放与碰撞体分离，消除 PhysX 形状不匹配警告。
- 道路和小径不再是抛高的第二层网格；材质现在直接绑定到可见地形的三角面子集。可见道路和 PhysX 地形使用相同顶点和三角分割，高度偏差由约 3.5 cm 降为 0。

## 运行验收

在项目根目录执行：

```bash
OMNI_KIT_ACCEPT_EULA=yes env PYTHONPATH=source third_party/IsaacLab/.venv/bin/python \
  scripts/validate_representative_go2_dynamics.py --map city_01 --route 1 --duration-s 12
OMNI_KIT_ACCEPT_EULA=yes env PYTHONPATH=source third_party/IsaacLab/.venv/bin/python \
  scripts/validate_representative_go2_dynamics.py --map park_01 --route 1 --duration-s 12
OMNI_KIT_ACCEPT_EULA=yes env PYTHONPATH=source third_party/IsaacLab/.venv/bin/python \
  scripts/validate_representative_go2_dynamics.py --map forest_trail_01 --route 1 --duration-s 12
python scripts/summarize_representative_go2_dynamics.py
```

汇总结果位于 `outputs/terrain_dataset_v3_realistic/dynamics/representative_dynamics_report.json`。三张代表图必须全部为 `PASS`。

## 当前结果

| 地图 | 路线进度 | 平均速度 | 横向误差 P95 | 足端接触比 | 滑移 P95 | 最大倾角 | 跌倒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| city_01 | 8.91 m | 0.87 m/s | 0.78 m | 96.7% | 0.12 m/s | 3.78° | 否 |
| park_01 | 12.05 m | 0.98 m/s | 0.90 m | 94.7% | 0.26 m/s | 3.87° | 否 |
| forest_trail_01 | 8.79 m | 0.80 m/s | 0.64 m | 95.0% | 0.55 m/s | 30.66° | 否 |

森林的倾角和滑移明显高于城市和公园，符合其崎岖设计，也是后续扩展到 60 图动力学验收时应重点关注的指标。

## 范围边界

本报告仅证明三张代表地图上的真实 PhysX 接触和 Go2 运控可工作。它不等于 60 图全量动力学验收，也不包含相机、视觉跟踪、目标跟随、任务终止或一小时闭环。

## 实时人物预览

人物运动器已接入 Isaac Sim 实时循环：根节点跟随实际三角地形，骨骼播放行走循环，且同步移动不可见的运动学胶囊碰撞体。预览中没有干扰人和事件。

```bash
OMNI_KIT_ACCEPT_EULA=yes env PYTHONPATH=source third_party/IsaacLab/.venv/bin/python \
  scripts/render_live_target_route_preview.py --map park_01 --route 1 --duration-s 8
```

输出视频为 `outputs/terrain_dataset_v3_realistic/live_target/park_01_route_01_live_target.mp4`。渲染器使用稳定直接光照，以 1920×1080 渲染后面积降采样到 1280×720，用于减少当前驱动不支持 DLSS-RR 时的颗粒和地表高频混叠。当前公园回归路段的脚部穿模已通过消除可见—物理高度差修复；针对崎岖森林地形的通用骨骼足部 IK 仍未验收。
