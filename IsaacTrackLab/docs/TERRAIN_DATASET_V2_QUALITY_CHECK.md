# 城市、公园、森林泥土山径数据集：生成与质量检查

这套数据集与冻结的 305 m 城市闭环基线相互独立。它包含城市、公园、森林泥土山径各 20 张地图，共 60 张；每张地图有 20 条确定性随机路线，共 1200 条。当前仍没有动态遮挡事件或干扰人物。

## 生成结果

- 场景：`assets/scenes/terrain_dataset_v2/<地图ID>/<地图ID>.usdc`
- 场景清单：每个地图目录的 `scene_manifest.json`
- 路线数据：每个地图目录的 `routes/generated_routes.json`
- 路线叠加层：每个地图目录的 `routes/generated_routes.usda`
- 全量报告：`outputs/terrain_dataset_v2/quality/quality_report.json`
- 表格报告：`outputs/terrain_dataset_v2/quality/quality_report.csv`
- 2D 图集：`outputs/terrain_dataset_v2/atlas/*_atlas.png`

每张地图采用带种子的多频高度场、局部山丘/洼地、脊线和沟槽。规划器和 USD 构建器调用同一个解析高度函数，避免“路线认为能走、场景几何却不同”的漂移。人物目标的坡度上限独立且低于机器人的上限；路线同时检查连续线段的障碍净空和禁入区相交，不只检查离散端点。

## 一键自动验收

从 Isaac Lab 目录运行：

```bash
OMNI_KIT_ACCEPT_EULA=yes \
PYTHONPATH=/home/abc/Documents/视觉跟踪/IsaacTrackLab/source \
uv run --extra isaacsim python \
  /home/abc/Documents/视觉跟踪/IsaacTrackLab/scripts/validate_terrain_dataset_v2.py \
  --config /home/abc/Documents/视觉跟踪/IsaacTrackLab/configs/terrain_dataset_v2_catalog.json
```

只有输出 `TERRAIN_DATASET_QUALITY PASS maps=60 routes=1200 failed=0` 才算通过。脚本逐图检查：

1. USDC 能打开、单位为米、地面和所有障碍有碰撞 API；地面绑定物理材质。
2. 每图恰好 20 条路线且几何签名不重复。
3. 每个路线点的高度与实际生成 USD 所用的地形函数一致。
4. 路线长度、人物坡度上限、点间连续性、障碍连续净空和不可通行区均合格。
5. 城市、公园、森林分别满足递增的高差与 95 分位坡度门槛，防止地图退化成平面。

`quality_report.csv` 适合排序查看最难地图。重点列为 `height_range_m`、`slope_p95_deg`、`route_slope_max_deg`、`route_clearance_min_m`、`route_coverage_10m_cells`、`unique_route_count`。

## 先看 2D 图集

城市、公园和森林图集分别显示 20 张地图；底图明暗代表高程，白点代表静态障碍，黑色大圆代表城市建筑的保守碰撞占地，橙色矩形代表有代价的可通行复杂地面，红色半透明矩形代表禁入区域。为了可读性，每格只画前 5 条路线，完整 20 条仍在 JSON 和 USDA 中。

重新生成图集：

```bash
OMNI_KIT_ACCEPT_EULA=yes \
PYTHONPATH=/home/abc/Documents/视觉跟踪/IsaacTrackLab/source \
uv run --extra isaacsim python \
  /home/abc/Documents/视觉跟踪/IsaacTrackLab/scripts/render_terrain_dataset_atlas.py
```

## Isaac Sim 中抽样检查

推荐使用交互式浏览器，一次启动后即可查看全部 60 张地图和每图 20 条路线：

```bash
cd /home/abc/Documents/视觉跟踪/IsaacTrackLab
./scripts/open_terrain_browser.sh --map city_01 --route 1
```

浏览器左上角有操作面板。左右方向键切换地图，上下方向键切换路线；`A` 在当前路线和全部 20 条路线之间切换；`C` 显示或隐藏碰撞代理；`F` 恢复斜视相机。面板也提供城市、公园、森林快速跳转以及俯视、斜视、近地相机按钮。绿色球为路线起点，红色球为终点。

每一类至少抽查高差最小、坡度最大和障碍最多的 3 张图：

1. 在 Isaac Sim 中打开地图的 `.usdc`，确认默认 Prim 是 `/OutdoorMap`，材质清晰且没有粉色丢失贴图。
2. 将同目录的 `routes/generated_routes.usda` 作为 Reference 加入 Stage；彩色曲线应在地表上方约 0.3 m，不能穿过建筑、车辆、树干、巨石或池塘禁入区。
3. 打开 Physics Debug/Colliders，检查地形、城市建筑、车辆、树干、岩石等都显示碰撞轮廓；碰撞代理默认在普通渲染中不可见。
4. 选中 `/OutdoorMap/Terrain/Heightfield`，确认 `CollisionAPI` 存在，并且 `material:binding:physics` 指向 `/OutdoorMap/Looks/TerrainPhysics`。挑战地面子集应绑定各自的摩擦材质。
5. 任选一条路线，按 JSON 中相邻 `points_m` 驱动目标人物；目标脚底应始终贴地、无瞬移，速度可继续沿用项目的 1–1.5 m/s 范围。若目标在某点无法稳定行走，应记录地图 ID、路线序号和点序号，而不是直接修改路线文件。

## 重建命令

先生成目录，再构建 USD 和路线：

```bash
cd /home/abc/Documents/视觉跟踪/IsaacTrackLab
PYTHONPATH=source python scripts/generate_terrain_dataset_catalog.py

cd third_party/IsaacLab
OMNI_KIT_ACCEPT_EULA=yes PYTHONPATH=/home/abc/Documents/视觉跟踪/IsaacTrackLab/source \
uv run --extra isaacsim python /home/abc/Documents/视觉跟踪/IsaacTrackLab/scripts/build_outdoor_map_batch.py \
  --config /home/abc/Documents/视觉跟踪/IsaacTrackLab/configs/terrain_dataset_v2_catalog.json

OMNI_KIT_ACCEPT_EULA=yes PYTHONPATH=/home/abc/Documents/视觉跟踪/IsaacTrackLab/source \
uv run --extra isaacsim python /home/abc/Documents/视觉跟踪/IsaacTrackLab/scripts/generate_outdoor_routes.py \
  --config /home/abc/Documents/视觉跟踪/IsaacTrackLab/configs/terrain_dataset_v2_catalog.json --count 20
```

若只修改某张图，在两个命令后都加 `--map city_01`（或对应地图 ID）。路线是可复现的；只有改变地图种子、`--seed-offset` 或目录生成参数时才会产生新路线。
