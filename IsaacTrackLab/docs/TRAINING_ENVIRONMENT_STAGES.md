# 端到端训练环境：阶段状态与检查

## 阶段1：冻结305 m基线并隔离控制模式（已验证）

正式视频和结果仍是 `showcase_hybrid`：MaST框提供距离/居中反馈，预定目标方位提供转向前馈，紫色背心检测可以重置跟踪器。它只用于场景、物理、录制和回归展示，不能当作纯端到端结果。

`pure_e2e` 命令路径只接受视觉控制输出，不读取预定路线方位，也不允许真值框重置跟踪器。检查冻结文件：

```bash
cd /home/abc/Documents/视觉跟踪/IsaacTrackLab
python scripts/validate_baseline_305m.py
```

完整校验会读取531 MB视频；快速校验可加 `--skip-large-media`。短时纯模式证据在 `outputs/baseline_305m/diagnostics/stage1_pure_e2e_smoke.json` 和同名视频中。

## 阶段2：标准训练环境契约（已验证）

`source/isaac_track_lab/envs/core.py` 定义：

- `ActorObservation`：RGB、模板、关节位置/速度、基座角速度、重力投影、足端接触和可选深度；
- `PrivilegedSignals`：距离、真值可见性、碰撞、机体高度/倾角和边界，只进奖励、终止和指标；
- `PlanarVelocityAction`：`vx_mps, vy_mps, yaw_rate_rps`；
- `StandardTrackingEnv.reset/step`：统一训练和评测入口。

同一种子必须得到相同的初始观测与扰动计划。运行锁定环境的测试：

```bash
cd /home/abc/Documents/视觉跟踪/IsaacTrackLab/third_party/IsaacLab
OMNI_KIT_ACCEPT_EULA=yes PYTHONPATH=/home/abc/Documents/视觉跟踪/IsaacTrackLab/source \
  uv run --extra isaacsim python -m unittest discover \
  -s /home/abc/Documents/视觉跟踪/IsaacTrackLab/tests -v
```

## 阶段3：无真值找回诊断（已验证，事件已从活动场景移除）

曾用一个带运动学碰撞代理的车辆做过一次诊断，用来确认视觉丢失后的停车扫描和重新捕获链路。该车辆和强制遮挡参数已从默认渲染脚本移除。当前活动的305 m场景不含任何新增遮挡或干扰事件，共使用91个城市碰撞体和1个目标人物碰撞代理。

`pure_e2e` 的丢失判定只使用MaST可见性/置信度、初始模板与当前预测框的外观相似度。连续丢失后机器人停止前进并左右扫描；连续三帧重新确认后恢复跟随。验证结果为2段视觉丢失、2次找回、最终状态 `tracking`、覆盖率95.83%、最终距离3.35 m，碰撞/跌倒/越界均为0。

```bash
cd /home/abc/Documents/视觉跟踪/IsaacTrackLab
python scripts/validate_recovery_gate.py
```

视频和JSON位于 `outputs/baseline_305m/recovery/physical_occlusion_recovery_gate_v4.*`，只作为历史诊断证据，不是当前默认场景的组成部分。

## 阶段4：模块化训练接口（已验证）

`configs/training_phases.json` 只定义三个模块的训练开关与串联数据契约，不决定网络结构、损失函数或优化器。已支持：

- 仅训练视觉编码器；
- 仅训练算法约束导航模块；
- 仅训练运控模块；
- 联合训练视觉与导航，冻结运控；
- 三段端到端联合训练。

`source/isaac_track_lab/training/freezing.py` 会校验三模块注册表，对参数设置 `requires_grad`，只向优化器返回当前可训练参数，并输出可写入 checkpoint 的冻结审计记录。

干扰、遮挡、资产替换、路线修改和场景扩展均不会自动执行；必须收到具体指示后，才会一次加入一个指定事件。

## 阶段5：现有场景十分钟验收（已验证）

在不扩展地图、不修改路线、不加入事件或干扰人物的前提下，完成了600 s `pure_e2e` 连续测试。目标以1.25 m/s走完整条305.176 m路线后停在终点，机器人继续视觉闭环保持到600 s结束。

正式结果：追踪覆盖率100%、可见率100%、目标丢失0、重捕获0、碰撞0、跌倒0、越界0；平均目标距离3.497 m、最终距离3.221 m。D455内参、安装外参、路面摩擦和92个碰撞体均保持冻结值。控制路径不读取路线方位、真值框、目标/机器人世界位姿或路线进度；仿真真值只用于终止和指标。

```bash
cd /home/abc/Documents/视觉跟踪/IsaacTrackLab
python scripts/validate_ten_minute_acceptance.py
```

正式视频和JSON位于 `outputs/baseline_305m/ten_minute/stage5_10min_no_event_v3.*`。

## 阶段6：扩展场景与一小时最终验收（最后执行）

用户已明确授权本阶段的第一部分：批量建立全地形地图。当前已生成森林、峡谷、雪坡、湿地、沙丘和海岸 6 张静态地图，以及每图 16 条带种子的碰撞/坡度约束路线；USD、局部摩擦、碰撞、2K PBR/HDRI 引用和 96 条路线已通过结构校验。详见 `docs/OUTDOOR_ALL_TERRAIN_BATCH.md`。

这不代表阶段6最终完成：尚未逐图完成 Go2 出生稳定、固定速度、转向、坡面和整条路线动力学回归，也没有执行一小时任务。没有加入遮挡事件或干扰人物。最终一小时阶段仍要求连续3600 s、不循环小路线、不在检查点重置机器人或跟踪记忆，并按冻结种子报告成功率、跟踪覆盖、找回、碰撞、跌倒、越界和控制平滑度。完成前不得报告“一小时抗干扰通过”。
