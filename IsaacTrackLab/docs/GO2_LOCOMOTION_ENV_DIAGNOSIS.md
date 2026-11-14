# Go2运控—环境解耦诊断

## 结论

同一份实机Go2 ONNX运控在空白PhysX平地上能够正确复现1.25 m/s：实际速度为1.241–1.247 m/s，指令跟随比为99.25%–99.73%，没有跌倒或力矩饱和。因此当前主要问题不是运控网络、20/0.5关节PD或摩擦系数过大。

城市道路USD的碰撞网格原先存在面绕序错误。`box_mesh()`写出的六个面的几何绕序与声明法向相反；渲染采用声明法向所以画面看起来正常，但PhysX按三角形绕序处理碰撞。修复前，Go2稳定高度从正常的约0.578 m降到0.362 m，足端穿入路面，1.25 m/s指令下实际速度约为0。现已反转错误面绕序，并在场景验证器中加入“几何法向与声明法向同向”的回归检查。修复后不加任何支撑平面，原生城市道路实测1.240 m/s、0.436 m步幅、基座高度0.578 m。

人物动画采用的USD Timeline刷新是一个独立风险。在已建立PhysX张量视图后手动推进Timeline，最小复现会报告张量视图失效；运行时反复写人物根节点USD位姿又会引起全场景重同步，使同一步态从1.21 m/s降到0.60 m/s。现在视觉人物根节点使用Fabric位姿写入，运动学胶囊使用PhysX张量写入，骨骼步行循环不再推进Timeline。修复后完整305.176 m回归的机器人平均平面速度为1.230 m/s。

## 实测表

| 环境 | 静/动摩擦 | 实际速度 | 指令跟随比 | 步幅 | 周期 | 关节跟踪RMSE |
|---|---:|---:|---:|---:|---:|---:|
| 空白平地 | 0.25/0.20 | 1.247 m/s | 99.73% | 0.422 m | 0.338 s | 0.232 rad |
| 空白平地 | 0.50/0.40 | 1.245 m/s | 99.59% | 0.437 m | 0.351 s | 0.230 rad |
| 空白平地 | 0.75/0.60 | 1.243 m/s | 99.41% | 0.436 m | 0.351 s | 0.230 rad |
| 空白平地 | 0.95/0.82 | 1.241 m/s | 99.25% | 0.435 m | 0.351 s | 0.230 rad |
| 双层同高平地 | 0.95/0.82 | 1.239 m/s | 99.13% | 0.435 m | 0.351 s | 0.230 rad |
| 城市原生道路（修复前） | 0.95/0.82 | ~0 m/s | ~0% | 无有效步态 | — | 0.503 rad |
| 城市+连续支撑面（旧诊断） | 0.95/0.82 | 1.240 m/s | 99.23% | 0.435 m | 0.351 s | 0.230 rad |
| 城市原生道路（绕序修复后） | 0.95/0.82 | 1.240 m/s | 99.17% | 0.436 m | 0.351 s | 0.230 rad |

完整JSON、50 Hz时序CSV和汇总表在 `outputs/locomotion_benchmark/`。

## 如何复测

从Isaac Lab目录执行。以下命令不创建人物、相机、跟踪器、路线或任务终止器：

```bash
cd /home/abc/Documents/视觉跟踪/IsaacTrackLab/third_party/IsaacLab
OMNI_KIT_ACCEPT_EULA=yes \
PYTHONPATH=/home/abc/Documents/视觉跟踪/IsaacTrackLab/source \
uv run --extra isaacsim python \
  /home/abc/Documents/视觉跟踪/IsaacTrackLab/scripts/benchmark_go2_locomotion.py \
  --device cuda:0 --command-vx-mps 1.25 --duration-s 8 \
  --static-friction 0.95 --dynamic-friction 0.82 \
  --output-stem flat_recheck
```

城市原生道路复测：

```bash
# 修复后原生道路：不带支撑平面，应约1.24 m/s
/home/abc/Documents/视觉跟踪/IsaacTrackLab/scripts/benchmark_go2_locomotion.py \
  --device cuda:0 --environment city \
  --command-vx-mps 1.25 --output-stem city_native_recheck
```

增加 `--gui` 可以在Isaac Sim窗口中观察。JSON中重点检查：

- `settled_base_height_m`：城市道路顶面为0.20 m，正常值应约0.578 m；
- `mean_forward_speed_mps`和`speed_command_ratio`：1.25 m/s指令应接近1.24 m/s和99%；
- `joint_tracking_rmse_rad`：当前健康基线约0.23 rad；
- `joint_peak_to_peak_rad`：用于比较每个关节目标/实际摆幅；
- `torque_saturation_fraction`：本次为0，执行器力矩上限不是瓶颈；
- `fallen`：必须为false。

`foot_contact_proxy`只是按足端高度估算的诊断量，不等同于力传感器，不能单独用于认定打滑。

## 与实机逐项对齐

当前仿真已经对齐了策略的关键接口：50 Hz策略周期、FL/FR/RL/RR策略关节顺序、实机SDK到策略的`joint_ids_map`、动作缩放0.25、默认关节角、Kp=20和Kd=0.5。Isaac侧仍使用NVIDIA Go2 USD与DCMotor饱和模型，并不包含实机电机驱动延迟、足垫柔顺性和机身载荷偏差。

要判断“同一组关节动作是否产生不同步幅”，实机应在水平防滑地面记录一次8秒固定1.25 m/s试验，保存50 Hz的12路目标角、实际角、关节速度、IMU、机身里程计和足端力。与本报告比较以下冻结指标：

- 速度指令比：建议误差不超过5%；
- 步幅：仿真基线约0.435 m，建议误差不超过10%；
- 步态周期：仿真基线约0.351 s，建议误差不超过5%；
- 每关节目标/实际峰峰值与RMSE；
- 基座高度、俯仰/横滚和横向漂移。

如果实机速度和步幅也接近这些数值，就无需重新训练运控，应该修道路碰撞网格并解耦动画时间轴。如果实机差异超过阈值，再校准Go2 USD的质量/惯量、足端接触材料、执行器力矩—转速曲线、控制延迟和机身载荷。
