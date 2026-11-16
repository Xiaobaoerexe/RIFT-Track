# AS2 深度感知运控训练

面向复现实验的同事：本分支在 Isaac Sim / PhysX 中训练 AS2，使用 SuperMimic 式世界模型想象优化和 CTS 教师／学生策略。`main` 保留 AS2 盲狗基线；请在本 depth 分支工作。

默认复现 **137 版本**：3 个稠密 Actor 专家、4 个任务 Critic、学生本体 MoE + 深度 CNN + GRU。预测深度 latent 在进入学生策略时停止梯度，WM 自身的 latent 递推仍保留梯度。教师不接收任务 ID。

## 1. 自动识别单卡／多卡4090

2026-11-16：同步175已验收的加速实现，137仍只在策略入口截断预测latent梯度。
默认推荐 `configs/workstation_4090_24g.json`：4096环境、8192融合想象、2048物理监督窗口、2GiB CPU图像缓存、4个CPU线程；48GB显存及大内存机器也可用 `configs/host137.json`（26GiB CPU缓存、8线程）。

`gpu_mode=auto` 按CUDA可见设备的UUID核对NVIDIA清单，跳过其他计算进程占用的卡，主卡需至少21GiB空闲。额外空闲4090需至少4GiB空闲，自动分担冻结的深度加噪网络推理。每张卡处理完整图像块，保留逐图噪声键与返回顺序，输出回到主卡。几何、PhysX、Replay、WM/Actor/Critic/CTS参数和优化器仍在主卡；这不是DDP，也不是多份独立训练，更不能将两张24GB显存当成一张48GB。

该自动分配不修改全局批量、损失分母、优化器步数或课程。`CUDA_VISIBLE_DEVICES` 可限制可用卡；只暴露一张时自动沿用单卡计算。资源选择是启动快照，不是集群预约系统；团队应避免同时向同一批卡启动多个作业。

**多卡分支按要求仅做静态审查，没有执行GPU测试，不能承诺多卡加速倍数。** 4090通常缺少CUDA P2P，跨卡拷贝会有额外成本；加噪只是训练的一部分，Actor BPTT尚不跨卡。单卡加速证据见 `SPEED_ACCEPTANCE.json`。不建议用 `torchrun` 启动本入口。

## 2. 获取代码

```bash
git clone --branch as2-depth-instinct-wm50-vis10-20261115 \
  --single-branch https://github.com/Xiaobaoerexe/RIFT-Track.git as2-depth
cd as2-depth
git rev-parse HEAD
```

不要向 `main` 合并或推送本实验。本次加速参照版本为 `d2d270b1c319f1f79786d6b3baa5c46a6f592a72`；本次同步已验收的加速配置、清除旧逻辑，并增加自动多卡冻结加噪推理；监督采样变化和验证边界见第8节。

## 3. 配置匹配的 Isaac Lab 环境

已使用的关键版本：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12，参考环境 3.12.16 |
| Isaac Sim | 6.2.0.0 |
| PyTorch | 2.11.0+cu130 |
| Warp | 1.17.0 |
| NumPy | 2.5.1 |
| trimesh | 4.11.1 |
| Isaac Lab / PhysX 包标签 | 28.0.0 / 8.0.0，来自锁定的源码树 |

**不能用 `pip install isaaclab` 代替匹配源码。** 本项目调用该源码树的 `launch_simulation`、PhysX 工厂和控制器接口。另一台服务器虽显示 Isaac Lab 3.0.0，但经逐文件比较核心实现一致；版本标签本身不足以判断兼容。

团队可从 137 获取目前安装使用的 `physx_resolved` 源码树及 `uv.lock`。以下命令在同事自己的 Linux 机器执行，不修改服务器：

```bash
export AS2_LAB_SOURCE="$HOME/deps/as2-physx-resolved"
mkdir -p "$AS2_LAB_SOURCE"
rsync -a --exclude='__pycache__' --exclude='*.pyc' --exclude='.venv' \
  abc@192.168.8.137:/home/abc/Documents/go2_supermimic_runtime/physx_resolved/ \
  "$AS2_LAB_SOURCE/"

# 与 environment/isaaclab_source_lock.json 比较，不要重新生成 uv.lock。
sha256sum "$AS2_LAB_SOURCE/pyproject.toml" "$AS2_LAB_SOURCE/uv.lock"
```

该依赖源码树不打包进本训练仓库。无法访问 137 时，先向团队获取同一份依赖树归档，并核对哈希；不要静默换成上游最新版。

安装 `uv` 后，在复制的依赖树中按其现有锁文件建立环境：

```bash
cd "$AS2_LAB_SOURCE"
uv sync --frozen --extra isaacsim --python 3.12
export AS2_PYTHON="$AS2_LAB_SOURCE/.venv/bin/python"

cd /path/to/as2-depth
uv pip install --python "$AS2_PYTHON" --no-deps -e .
```

上述安装步骤依据服务器现有源码及锁文件整理，尚未在一台全新工作站从零安装验收。下载失败应核对锁文件中的软件源访问，不要自动升级 Torch、Warp 或 Isaac Sim。共享 Isaac Lab 环境可能安装 Newton/MuJoCo 依赖，但本训练入口固定选用 **`isaacsim_physx`**；安装依赖不代表切换物理后端。

启动前检查版本与 API：

```bash
"$AS2_PYTHON" - <<'PY'
import importlib.metadata as m
import torch
from isaaclab.app import launch_simulation, add_launcher_args
from isaaclab.app.sim_launcher import _make_physics_cfg
from go2_physx_cts.config import TrainCfg
import json
for name in ('isaacsim', 'isaaclab', 'isaaclab_physx', 'torch', 'numpy', 'warp-lang', 'trimesh'):
    print(name, m.version(name))
assert torch.cuda.is_available()
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, round(p.total_memory / 2**30, 2), 'GiB')
cfg = TrainCfg.from_record(json.load(open('configs/workstation_4090_24g.json')))
cfg.validate()
assert cfg.latent_actor_stop_gradient
assert cfg.depth_period == 1 and cfg.imagination_depth_period == 5
print('Configuration and required imports passed; PhysX startup is checked by the actual run.')
PY
```

## 4. 启动一个自动利用可用GPU的训练

在仓库根目录运行，权重放在仓库外。确认GPU未被其他作业占用，不要复用已有run目录。

```bash
git status --short
mkdir -p "$HOME/as2-depth-runs"
"$AS2_PYTHON" -u -m go2_physx_cts.train \
  --config configs/workstation_4090_24g.json \
  --output "$HOME/as2-depth-runs/instinct_s17_speed" \
  --usd assets/as2/as2.usd
```

不要将 `CUDA_VISIBLE_DEVICES` 限制为单卡后期待多卡工作。如需手动只用一张可设置 `CUDA_VISIBLE_DEVICES=0`。解析后的实际设备和分工写入 `gpu_resources.json`；没有可用主卡就明确报错，不抢占现有任务。

**137当前运行目录 `/home/abc/Documents/as2_supermimic_depth` 不属于此次修改范围。** 本分支代码副本为 `/home/abc/Documents/as2_supermimic_depth_instinct`，发布工作树为 `/home/abc/Documents/as2_depth_github_20261115`；此次没有启动、重启或测试137训练。

## 5. 复现时保持不变的设置

- 4096 环境，每轮每环境采集 1 步，Replay 每环境 1000 步。
- 每轮 8192×32 想象：6144 教师、2048 学生；Actor 目标为两分支各自均值之和。
- WM 2048×8 自回归监督；2500 轮前每轮更新 2 次，之后 4 次。
- Critic 总计 5 遍×4 minibatch；学生蒸馏每轮 2×160×64，burn-in 16。
- 真实深度与 WM latent 递推 50 Hz；想象策略在第 0、5、10…步刷新视觉 latent，即 10 Hz；本体观测和 GRU 仍 50 Hz。137 版本仅截断策略入口的预测 latent 梯度。
- AS2 高度目标 0.36 m；相机相对 `base_link` 为 `(0.101958, 0, 0.327)` m，水平朝前。
- 训练从初始化权重开始，总计 120000 轮。0～1000 轮主要训练 WM，1000 轮起 Critic 预热，1500 轮起 Actor／教师／学生训练；不能用前几百轮判断行走是否学会。
- 5000 轮后场景比例为普通速度 60%、跨越 24%、高速 8%、急变向 8%；无恢复或避障任务。

`configs/host175.json` 是保留视觉入口梯度的对照；`host125.json` 使用 MLP Actor；`host156.json` 使用单 Critic／单奖励。它们不是双卡启动配置，不要混用后仍称为 137 复现。

## 6. 观察进度、恢复与对照

每个 run 保存：

| 文件 | 用途 |
| --- | --- |
| `resolved_config.json`、`process.json` | 实际配置、提交和进程身份 |
| `physics_contract.json` | 实际加载的 PhysX 参数、相机频率和依赖 |
| `training.jsonl`、`progress.json` | 迭代、损失、性能和课程统计 |
| `model_*.pt` | 模型、优化器、课程与调度状态 |
| `native_*.json` | 教师／学生固定任务原生评测 |

对比相同任务条件下的速度误差、存活与成功率，同时看阶段耗时。不要只看训练总奖励或 Critic loss。

恢复同一架构 checkpoint 时，在原命令中加 `--resume /absolute/path/model_XXXXXXXX.pt`，建议使用新的输出目录。恢复保留模型／优化器／课程状态，但仿真 episode、Replay 和 GRU 历史会重新建立，不是逐位连续重放。

`VISUAL_CLOCK_ACCEPTANCE.json` 记录上一个版本的全规模执行验收：不同服务器平均约 6.1～8.3 秒／轮；它不是双卡工作站测速，也不是已学会全部任务的证明。官方视觉 MuJoCo 链路尚未完成，不要将原生评测当作 sim2sim 验收。

## 7. 文件与来源

训练入口 `go2_physx_cts/train.py`；学习器 `learner.py`；想象 `imagine.py`；WM `world_model.py`；策略 `networks.py`；奖励 `task.py`。机器人 USD/URDF/mesh、奖励标尺以及冻结噪声权重均随分支提供，无需旧策略 checkpoint。

算法来源及固定提交见 `UPSTREAM.json`，物理／实现契约见 `BUILD_CONTRACT.txt`，版本实测见 `RUNTIME_LOCK.json`。保留 `licenses/` 中各上游许可证，其中 InstinctMJ 来源包含 CC-BY-NC-4.0；请按相应条款使用和分发。

## 8. 2026-11-16 加速与验证边界

- WM物理监督及latent递推仍为2048×8；每次更新随机512条完整窗口生成8步视觉标签，初始latent仍编码2048张，重建仍512张，全标签验证不抽样。
- 缓存最终FP16带噪图像；episode参与槽位偏移；噪声CNN用BF16，几何/随机场/腐蚀运算FP32。
- CTS合并B×T的Actor调用，去掉GRU重复重算；Critic分组与BF16隐藏层保持20次优化步；使用fusedAdam，删除无用Critic拟合图。
- 175/125/156完整更新平均4.21/3.89/4.41秒。125同机从6.30降至3.89秒。这是已测单卡版本的证据，不是137消融版或多卡版实测。
- 短期512窗口对照：32步latent误差约−1.03%，综合物理误差约+2.27%，速度预测RMSE约+7.12%。不能据此声称长期策略无损。
- 每5000轮记录只读8/16/32步全标签预测诊断，不参与停机或学习率调节。价值越界停机逻辑已删除；保留非有限数值报错。
- 本次没有在137执行测试、导入训练模块或启动仿真。多卡加噪路径新增，仅静态审查；跨卡数值及吞吐仍待团队在实际多卡机器验收。
