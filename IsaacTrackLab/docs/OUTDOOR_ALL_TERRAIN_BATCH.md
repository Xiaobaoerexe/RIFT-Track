# 户外全地形批次：生成、检查与使用

这一批地图用于避障和全地形数据生产，不替换已冻结的 305 m 城市回归基线。当前只包含静态地形、静态障碍和物理挑战结构；没有遮挡事件、干扰人物或自动扩展到一小时路线。

## 地图组成

| 地图 | 主要地形 | 挑战结构 | 默认摩擦（静/动） |
|---|---|---|---|
| `forest_trail` | 森林泥土起伏地 | 泥地、树根区、树木、倒木、树桩、岩石 | 0.92 / 0.78 |
| `rocky_canyon` | 岩石峡谷与碎石坡 | 狭窄通道、碎石区、大小岩块 | 0.88 / 0.72 |
| `alpine_slope` | 雪地连续坡面 | 横坡、低台阶、裸岩 | 0.70 / 0.55 |
| `wetland_boardwalk` | 湿地与泥泞地 | 不可通行深水、可碰撞栈道、泥盆 | 0.82 / 0.58 |
| `desert_dunes` | 沙丘与缓沟 | 软沙区、干沟、散落岩石 | 0.72 / 0.50 |
| `coastal_scrub` | 海岸砂石缓坡 | 不可通行潮沟、潮蚀岩棚、漂木 | 0.84 / 0.66 |

每图的任务/轨迹有效区为 180 m × 180 m，外围再生成 30 m 连续地形缓冲，避免机器人相机看到突兀的方形边缘。地面使用 2 m 高度网格、2K PBR 漫反射/粗糙度/法线材质、2K HDRI 和方向光。地形网格、全部障碍代理、栈道和台阶均有碰撞；每张地图使用独立基础物理材质，泥地、碎石、横坡、栈道等挑战区还通过面子集绑定各自的局部摩擦材质，而不只是改变画面颜色或规划代价。

## 一次性生成

以下命令都从固定的 Isaac Lab 环境执行：

```bash
cd /home/abc/Documents/视觉跟踪/IsaacTrackLab/third_party/IsaacLab

OMNI_KIT_ACCEPT_EULA=yes uv run --extra isaacsim python \
  /home/abc/Documents/视觉跟踪/IsaacTrackLab/scripts/fetch_outdoor_batch_assets.py

OMNI_KIT_ACCEPT_EULA=yes PYTHONPATH=/home/abc/Documents/视觉跟踪/IsaacTrackLab/source \
  uv run --extra isaacsim python \
  /home/abc/Documents/视觉跟踪/IsaacTrackLab/scripts/build_outdoor_map_batch.py

OMNI_KIT_ACCEPT_EULA=yes PYTHONPATH=/home/abc/Documents/视觉跟踪/IsaacTrackLab/source \
  uv run --extra isaacsim python \
  /home/abc/Documents/视觉跟踪/IsaacTrackLab/scripts/generate_outdoor_routes.py --count 16

OMNI_KIT_ACCEPT_EULA=yes PYTHONPATH=/home/abc/Documents/视觉跟踪/IsaacTrackLab/source \
  uv run --extra isaacsim python \
  /home/abc/Documents/视觉跟踪/IsaacTrackLab/scripts/validate_outdoor_map_batch.py
```

只重建一张地图可使用 `build_outdoor_map_batch.py --map forest_trail`。需要更多数据时运行 `generate_outdoor_routes.py --count 100 --seed-offset 10000`；不要复用相同的 `seed-offset`，否则会得到完全相同的路线。

## 随机轨迹算法

算法入口为 `isaac_track_lab.task.terrain_routes.TerrainRouteGenerator`。每条路线依次经过地图的多个任务门区，但门内起终点由种子随机采样。规划器在 2 m 栅格上使用 8 邻域代价 A*，同时检查：

- 机器人中心到障碍碰撞代理至少 1.2 m；
- 局部坡度不超过 28°；
- 不进入深水或潮沟等不可通行区域；
- 泥地、碎石、横坡等区域具有不同通行代价；
- 结果保留全部中间门点，按约 1 m 间距重采样；
- 总长必须在 220–430 m 之间。

同一地图、障碍清单和种子会逐点复现相同路线。输出位于每张地图的 `routes/generated_routes.json`，其中包含三维点、长度、累计爬升、最大坡度、最小障碍净距和经过的挑战区。`generated_routes.usda` 用于在 Isaac Sim 中叠加查看彩色轨迹。

## Isaac Sim 手动检查

1. 打开 `assets/scenes/outdoor_batch/<地图ID>/<地图ID>.usda`，确认 Stage 默认 Prim 为 `/OutdoorMap`。
2. 在 Stage 树检查 `/OutdoorMap/Terrain/Heightfield`，Physics 面板应显示 Collision；材质应引用对应 2K diffuse、roughness 和 OpenGL normal。
3. 展开 `/OutdoorMap/Collision`。这些代理默认不可见，但都应带 Collision API；可临时把 Visibility 改为 `inherited` 检查是否与树木、倒木和岩石对齐，检查后改回 `invisible`。
4. 检查 `/OutdoorMap/Looks/TerrainPhysics` 的静摩擦与动摩擦是否和上表一致。
5. 把同目录 `routes/generated_routes.usda` 作为额外 Stage 打开或子层加入，确认路线没有穿过碰撞代理、深水或潮沟。
6. 用 Physics Debug Visualisation 显示碰撞体，分别在平地、坡地、栈道/台阶和障碍旁出生 Go2。先做低速直行与原地转向，再做完整路线；出现足端下陷时先检查地面碰撞网格和出生高度，不要直接提高摩擦系数掩盖问题。

结构校验通过只证明 USD、碰撞、材质引用和路线几何满足契约，不等同于 Go2 已在全部地图上完成动力学回归。每张图仍需要单独跑出生稳定、固定速度、转向、坡面和完整轨迹测试后，才能加入训练采样池。
