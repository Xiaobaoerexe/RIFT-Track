# 真实规划全地形数据集 v3

本数据集把“随机散布资产”改为“先规划真实空间结构，再在结构内做受约束变化”。它包含城市、公园、森林泥土山径三类场景，每类 20 张地图，每张地图 20 条路线，共 60 张 USDC 和 1200 条路线。当前版本不包含干扰人物、动态遮挡或其他事件。

## 已实现的场景约束

- 城市：3×3 交错道路网、沥青路面、人行道、带路口缺口的物理路缘、按街区布置的建筑、避开路口并沿路停放的车辆和街道设施。
- 公园：真实地形湖盆、水平椭圆水面、自动岸线净空、碰撞岸石、两处以上丘陵、环形路和交叉步道、树木、巨石、长椅及嵌入式小石块。
- 森林：主山径和支路、山脊/沟壑/泥地、每图至少 90 棵树、岩石和倒木；路线上有低木头、嵌入石块和小凸起，属于可跨越训练特征。
- 普通障碍物参与路线避障；低木头、小石块和小凸起保留碰撞，但不作为禁行障碍，路线会有意经过它们。
- 所有路线必须位于语义道路或步道内。A* 会检查完整边段，不能在弯道处切出路径。

## 输出位置

- 规划目录：`configs/terrain_dataset_v3_realistic.json`
- 三维地图：`assets/scenes/terrain_dataset_v3_realistic/<地图ID>/<地图ID>.usdc`
- 每图 20 条路线：`assets/scenes/terrain_dataset_v3_realistic/<地图ID>/routes/generated_routes.json`
- 自动验收报告：`outputs/terrain_dataset_v3_realistic/quality/quality_report.json`
- 表格报告：`outputs/terrain_dataset_v3_realistic/quality/quality_report.csv`
- 二维单图和图集：`outputs/terrain_dataset_v3_realistic/atlas/`

## 最方便的逐图三维检查

在终端运行：

```bash
cd /home/abc/Documents/视觉跟踪/IsaacTrackLab
./scripts/open_terrain_browser.sh
```

浏览器现在默认打开 v3。也可以直接指定地图和路线：

```bash
./scripts/open_terrain_browser.sh --map park_01 --route 1
./scripts/open_terrain_browser.sh --map forest_trail_01 --route 7 --show-all-routes
```

操作方式：

- `←/→`：上一张/下一张地图；`↑/↓`：上一条/下一条路线。
- `A`：单路线/全部 20 条路线；`C`：碰撞代理；`F`：斜视相机。
- 面板按钮可直接跳到 City、Park、Forest，并提供俯视、斜视和贴地相机。

首次打开会等待材质编译。终端里的 DLSS、MaterialX 或缺少菜单扩展警告不等于启动失败；看到 `TERRAIN_BROWSER_MAP` 且窗口出现地图即为正常。

## 二维快速筛查

先查看三张总览图，再进入单图目录：

- `outputs/terrain_dataset_v3_realistic/atlas/city_atlas.png`
- `outputs/terrain_dataset_v3_realistic/atlas/park_atlas.png`
- `outputs/terrain_dataset_v3_realistic/atlas/forest_trail_atlas.png`
- `outputs/terrain_dataset_v3_realistic/atlas/maps/<地图ID>.png`

图中深色大圆是建筑的保守占地代理，白点是普通障碍物，橙色小方块是可跨越特征，彩色线为前 5 条路线。二维图用于规划检查，真实画面质量应以 Isaac Sim RTX 视口为准。

## 自动质量验收

```bash
cd /home/abc/Documents/视觉跟踪/IsaacTrackLab
third_party/IsaacLab/.venv/bin/python scripts/validate_realistic_terrain_dataset.py
```

通过标准包括：60 张地图、1200 条路线、路线长度 280–360 m、路线唯一且目标可达、不得离开语义道路/步道、普通障碍物净空合格、禁入水域不被穿越、碰撞数量与清单一致、摩擦材质存在、湖水与岸线至少保留 4 cm 净空，以及三类场景各自的结构和地形复杂度要求。

当前结果：`PASS maps=60 routes=1200 failed=0`。

## 建议的人工检查顺序

1. 先看三张 atlas，检查道路/步道连通性和路线覆盖。
2. 每类至少抽查 3 张三维地图，分别使用俯视、斜视、贴地相机。
3. 公园重点检查水面是否水平、岸线是否露出、路线是否绕湖。
4. 森林打开碰撞代理，检查低木头/小石块有碰撞，同时确认路线仍会经过。
5. 城市检查车辆方向、路口净空、路缘缺口和建筑是否侵入道路。
6. 最后在 RTX 视口用目标训练分辨率检查 PBR 材质、纹理尺度、阴影和视觉噪声。

## 重新生成

```bash
PYTHONPATH=source python scripts/generate_realistic_terrain_dataset_catalog.py
third_party/IsaacLab/.venv/bin/python scripts/build_outdoor_map_batch.py \
  --config configs/terrain_dataset_v3_realistic.json
third_party/IsaacLab/.venv/bin/python scripts/generate_outdoor_routes.py \
  --config configs/terrain_dataset_v3_realistic.json --count 20
third_party/IsaacLab/.venv/bin/python scripts/validate_realistic_terrain_dataset.py
```

目录生成是确定性的。若修改规划规则，应依次重建 USD、重算路线、重新验收，不应只改 JSON 后继续使用旧场景。
