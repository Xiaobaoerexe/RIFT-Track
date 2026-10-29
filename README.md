# Go2 / As2 · PhysX + SuperMimic + CTS

本文件是项目唯一维护的工程说明。配置 JSON、依赖锁、许可证、实验原始记录不是冗余文档，不作删除。参数发生修改后运行 `python tools/build_readme.py`，更新本文的完整源码附录及资产参数。不得把历史记录中的参数当成当前配置。

## 当前版本与证据边界

本次新增双机器人配置；137 上正在运行的 `physx_rebuild` 保持原版本，新增版本位于 `/home/abc/Documents/go2_supermimic_lab/physx_dual_robot_20261029`。没有启动新的正式训练，没有操作125或真机。运行中的 Go2 版本为 `fa3c776af150986776b5c513130c0e7a8b38b4c6`，不是本次新增版本。

Go2 已验证低速基线：`GO2_PHYSX_SMCTS_s17_20261029_014053_137/flat/model_00040000.pt`，SHA256 `eb12315e21e0037d81514af43184210276dee515c3722b9817b12cdc90d6fd11`。历史教师/学生均在原生和匹配控制契约的官方sim2sim通过10/10低速case；不代表高速、全地形或真机认证。As2 是新增 PhysX 接入，资产加载检查不等于学会运动，也没有已验证的本框架 As2 专家或 RoboGauge 迁移方案。

研究原则：源码事实、作者报告、本项目实测分别标记；保留失败证据；模拟或计划不能写成实测。官方权重的真机效果不能混称自训成功。只使用 Isaac Sim/PhysX，禁止恢复 Newton 或已退役的自写 MuJoCo 评测器。checkpoint 的 PD、默认姿态和动作尺度不可静默替换。原生学习能力与迁移效果分开判断。学生 MoE 没有直接回报梯度不能单独作为失败原因。

## 固定参考与环境

|用途|仓库及提交|
|---|---|
|世界模型训练机制|Zhengtianjin/go2-supermimic `d9eee9b3f248e5300e636c433255db2c1c3e086a`|
|CTS、速度/地形课程|wty-yy/go2_rl_lab `6c95d9feef4814c25eb6536e61bb38e1d1d718d4`|
|Go2官方迁移|wty-yy/RoboGauge `add2c7ee0185ea8fd93d242f86e1c8cac6e9913b`|
|As2几何|Unitree unitree_ros/as2_description，按资产manifest逐文件SHA256固定；既有证据未固定仓库提交，不能假称已固定|
|As2电机参数|Unitree unitree_rl_mjlab 的 as2_constants.py 原始副本；SHA256见资产manifest；同样未补造提交|

训练解释器 `/home/abc/Documents/go2_supermimic_runtime/physx/bin/python`，Isaac Lab3 + Isaac Sim6.2/PhysX；官方迁移解释器 `/home/abc/Documents/go2_supermimic_runtime/robogauge/bin/python`。后者 Python3.11.18、MuJoCo3.2.3、Torch2.4.1+cu124、NumPy1.26.4、dm-control1.0.23。必须满足官方MuJoCo<3.3，不能使用训练环境的3.12。完整迁移依赖见下文嵌入的runtime清单及`robogauge.uv.lock`。

## 两种机器人分别启动

从137的新版本目录执行，两个脚本共用训练锁，已有训练运行时会拒绝重叠启动。这里给的是启动命令，本次没有执行正式训练。

```bash
cd /home/abc/Documents/go2_supermimic_lab/physx_dual_robot_20261029
bash train_go2.sh --stage flat --output /home/abc/data2/go2_supermimic_lab/runs/GO2_NEW_FLAT
bash train_as2.sh --stage flat --output /home/abc/data2/go2_supermimic_lab/runs/AS2_NEW_FLAT
```

预览参数不启动：追加 `--print-config`。`--output`必须是新目录；`--usd`可显式指定资产；`--resume`只接受同机器人且控制常量完全匹配的PhysX checkpoint；`--iterations`默认60000，没有6/8小时墙钟上限。原生训练器正式启动要求干净的Git版本。`TRAIN_PYTHON`可以覆盖解释器路径。

Go2混合地形另加 `--stage terrain --driver OFFICIAL_GO2_TORCHSCRIPT --flat-expert VALIDATED_GO2_FLAT_CHECKPOINT --resume VALIDATED_GO2_FLAT_CHECKPOINT --evaluation-python /home/abc/Documents/go2_supermimic_runtime/robogauge/bin/python`。不要使用占位路径。As2尚无适配且验证的专家，因此禁止套用该Go2地形采集配方；当前As2入口明确支持从平地开始，不承诺完整多地形专家训练或sim2sim。

## 网络、观测和梯度

|模块|输入→结构→输出|更新来源|
|---|---|---|
|教师编码器|167→512→256→32，ELU，末端L2归一化|想象回报BPTT|
|学生MoE|450维；8专家，骨干450→512→256→2048；分组1×1 Conv产生8×32；门控450→512→256→256→8+Softmax；混合后L2归一化|只由CTS蒸馏更新参数|
|共享Actor|32潜变量+45单帧=77→512→256→128→12，ELU|教师和学生两分支回报梯度|
|Critic|167特权+2分支标签=169→512→256→128→1，ELU|想象TD(lambda)目标的MSE|
|世界模型|176→1024→1024→1024→73，SiLU|真实Replay的8步自回归监督|

学生单帧45维依次为角速度3、重力3、命令3、关节位置偏差12、关节速度12、上一动作12。尺度依次为.25、1、[.5,1,1]、1、.05、1；均匀观测噪声幅度.05/.05/0/.01/.075/0。历史10帧共450维，按观测项分块、每块从旧到新；不是把10帧整帧直接串接。reset清零历史。教师/critic/WM可用77点地形扫描；学生没有地形扫描。

70维物理状态是高度1、重力3、去yaw速度3、身体角速度3、关节位置偏差12、速度12、髋/膝/足相对位置各12。世界模型输入包含状态70、当前/上一动作各12、环境上下文5、扫描77；输出状态增量70和二维位置/yaw增量3。上下文当前为固定摩擦入口及未启用随机项、episode负重；具体顺序和实际赋值以源码附录为准。坐标采用XYZW四元数、FL/FR/RL/RR腿顺序，每腿hip/thigh/calf；命令为身体坐标vx、vy和wz。

RunningNorm独立维护历史、单帧、教师、Critic及WM输入/输出统计；初始count=1e-4，标准差分母加1e-6，输入裁剪±10。WM监督损失按输出标准差归一化，重置之后窗口无效，8步预测不detach。Actor更新冻结WM、Critic、学生编码器参数，但保留它们对输入的导数。Critic输入不含Actor持有的潜变量，防止潜变量bootstrap捷径。

想象8192条×32步，教师6144、学生2048。损失精确为 `teacher_loss.mean() + student_loss.mean()`，两分支权重1:1；它是0.5/0.5表达式的2倍，不能声称数值/梯度完全相同。全批一次计算，累积完才裁剪。学生编码器只做潜变量MSE蒸馏，教师标签detach，门控均衡项权重.01。每轮20次CTS，每次1024个真实学生来源历史。

## 优化、Replay与训练节奏

4096环境；每轮采集1个50Hz控制步；每环境Replay1000步=20秒；所有来源共享Replay，用于WM监督和想象起点。无专家动作模仿。平地默认无外部策略；Go2地形为10%官方混合地形driver+10%本项目低速平地专家，其余教师/学生3:1。实际4096分配2458/820/409/409，并沿地形列分散。平地专家不等同于跳跃或躲避专家。

WM每次2048窗口×8步；平地每轮2次，地形4次。窗口25%稀有优先、25%长期archive（空时回流普通池），其余普通；优先候选为4倍批量。archive容量100000，事件保留概率.02；事件含离地、竖直速度或终止。接触代理阈值.035m，不是实际接触传感器。完整定义见buffer附录。

优化器均Adam，无权重衰减；Actor betas=(.9,.95)，其他默认(.9,.999)，eps默认1e-8。WM/Actor/Critic基础lr=2e-4；30000轮余弦下降至3%，之后维持；地形Actor再乘.3。Actor渐入1000轮。CTS固定lr=.001。全局梯度norm裁剪1，Actor另外逐元素裁剪±.1；非有限更新跳过并计数。Critic每轮5epochs×4minibatches。gamma=.99、lambda=.95。

Actor优化32步可微奖励加末端Critic bootstrap，软生存概率按倾角sigmoid；时间上限截断仍bootstrap，并屏蔽之后Critic样本。真实跌倒70°阈值；soft sigmoid中心70°、宽度5°。平地冷启动WM预热300轮；地形1000轮。新地形阶段恢复权重且重置Critic时另预热Critic500轮，因此Actor在1500后才更新，随后1000轮渐入。恢复会新建Replay和模拟episode，并非全状态逐位恢复。

动作无tanh，无直接动作裁剪；关节目标=默认姿态+原始动作×scale，电机执行器负责力矩限制。25%环境使用OU时间相关探索噪声，theta=2、sigma由.3×sqrt(2theta)计算、dt=.02；其余高斯std=.1；噪声加入全部provider选出的动作后，Replay保存真实执行动作。episode结束清零OU。

训练恶化回退：超过预热后，全provider平均每个向量step跌倒数>3，连续两次日志窗口触发。恢复最近good策略和Critic，lr_scale减半、重新渐入；保留新WM/Replay/archive，重建Actor/Critic/CTS优化器。不是由sim2sim一次失败自动回退。日志每50轮，保存每1000轮；每5000轮进行独立原生/模型/官方迁移诊断（需要evaluation-python）。无评测解释器时不自动评测。60000是初始轮次审查点，续训须依据证据，不是能力承诺。

## 物理参数与机器人差异

|参数|Go2|As2|
|---|---|---|
|资产|随项目assets/go2，原有已使用USD|assets/as2，官方URDF完整网格转换USD|
|名义质量|训练USD历史16.087kg|URDF/USD17.64kg，包含固定足端|
|初始高度|.29m|.4m|
|左/右髋姿态|+.1/-.1rad|-.1/+.1rad|
|大腿/小腿姿态|.9/-1.8rad|.9/-1.8rad|
|armature|.000111842×减速比²，hip/thigh6.33、calf9.495|.026/.026/.038kg·m²|
|Kp/Kd|armature×(2π10)² / 4×armature×(2π10)|40/40/60与2/2/3|
|执行力矩限制|23.7/23.7/45.43Nm|50/50/75Nm|
|速度限制|30.1/30.1/20.07rad/s|24/24/16rad/s|
|动作scale|.25×effort/Kp|.3125rad（本项目归一化方案）|
|电机|DCMotor饱和力矩为执行限制2倍|IdealPD恒定力矩限制，不沿用Go2速度曲线|
|奖励高度目标|.38m|.30m（既有工程目标，不是厂家标定）|

两者控制50Hz、物理200Hz、每动作4substeps；episode20s。关节soft limit为硬限位90%。显式执行器摩擦0，机身线/角阻尼0，最大去穿透速度1m/s，自碰撞开启，位置solver4次/速度0次。地面静/动摩擦1、恢复系数0，地形材质组合average。控制延迟每episode随机0/1/2个物理substep，属于项目假设，不是As2实测。负重在既有base COM处均匀0–2kg，仅reset修改质量、不修改COM；episode内固定。不代表扩展物体的惯量模型。

As2 URDF硬限力矩60/60/90Nm和训练执行限制50/50/75Nm分开。完整每个link的质量、COM、惯量、碰撞以及joint轴/限位在下文资产附录。As2仅将base_link命名映射为base，保留固定脚，禁止使用Go2的.213m足端回退几何。当前接入继承本任务奖励/课程，不能宣称复现Unitree原生As2训练方案。

## 奖励与课程

保留已验证13项速度任务并加2项官方定义/课程；不是完整照搬官方PPO15项奖励。每步总和乘.02。权重：track_xy2、track_yaw1、upright1、tilt_rate_l2−.05、hard_landing−.5、crouch−30、joint_deviation_l2−.1、base_height_low−10、terminated平地−200/地形−600、action_rate_l2−.05、dof_pos_limits−1、joint_torques_l2−2e−5、joint_acc_l2−2.5e−7。xy/yaw跟踪exp误差分母均.25，没有动态宽度。直立exp分母.16；crouch阈值.23、base_height_low阈值.18。完整公式见task源码。

新增竖直速度平方权重−2到0（1500轮），地形crop相对高度平方权重−1到−10（5000轮）。真实奖励用实际碰撞mesh射线；想象用mesh导出的.05m高度图，不能声称误差为零。

速度课程5000/15000/25000/35000/45000轮，最终平地vx[-4.5,4.2]、vy[-1,1]、wz[-3,3]；斜坡/波浪vx不超过1.5，楼梯/障碍不超过1，wheels[-2,1.5]。各完整阶段和地形边界、采样概率在commands源码附录。当前1轮=1真实采集步，官方PPO1轮=24步，不能混称同等物理经验。无额外Rapid网格课程。

混合地形为固定源码direct mesh，10×20块、每块9×9m，最大初始等级5；列比例按官方归一化。地形进阶按episode运动距离：超过有效半块长度升级，低于命令积分目标的一半且未升级则降级。与速度轮次课程独立。完整几何参数和采样在下文代码，不自创采集百分比。

## 验证、评测与记录

`python -m unittest discover -s tests -v`验证梯度/Replay/机器人保护等。需固定上游checkout的源码对照测试另带source-dir，不把skip当pass。`tools/check_robot_load.py`仅4环境、50控制步，检查实际加载、名义质量、0/0.5/1/2kg负重、状态有限和episode质量稳定，无模型更新，不证明运动能力。As2结果见下文runtime证据。正式训练吞吐不能由4环境检查推断。

Go2官方链路：训练Python运行 `python -m go2_physx_cts.robogauge_export --checkpoint CKPT --output EXPORT --iteration N`；隔离官方Python运行 `python -m go2_physx_cts.robogauge_bridge --exports EXPORT --output REPORT --iterations N --video`。必须使用checkpoint控制契约；As2导出到该桥接会明确拒绝。默认固定命令诊断关闭官方可选噪声/延迟，`--official-noise-delay`恢复；保留官方startup settling。训练USD与迁移XML质量/惯量/solver等仍有差异，不声称完全物理一致。

原始记录在项目父目录research/runs，状态JSON和上游SHA清单继续保留。历史失败方案不再作为活动入口。此次删除仅针对重复的活动说明；许可证、上游原始文档、论文材料、实验报告作为证据保留。删除前记录哈希、大小并压缩保存，清单位于research/document_cleanup.json。

## 所有训练配置字段（代码生成）
|字段|平地默认|
|---|---|
|`robot`|`'go2'`|
|`stage`|`'flat'`|
|`num_envs`|`4096`|
|`buffer_len`|`1000`|
|`collect_steps`|`1`|
|`iterations`|`60000`|
|`warmup_iterations`|`300`|
|`wm_hidden`|`(1024, 1024, 1024)`|
|`wm_window`|`8`|
|`wm_batch`|`2048`|
|`wm_updates`|`2`|
|`wm_prioritized_frac`|`0.25`|
|`wm_archive_frac`|`0.25`|
|`archive_size`|`100000`|
|`archive_keep_prob`|`0.02`|
|`imag_batch`|`8192`|
|`teacher_fraction`|`0.75`|
|`horizon`|`32`|
|`actor_microbatch`|`8192`|
|`actor_hidden`|`(512, 256, 128)`|
|`critic_hidden`|`(512, 256, 128)`|
|`cts_updates`|`20`|
|`cts_batch`|`1024`|
|`cts_lr`|`0.001`|
|`cts_balance`|`0.01`|
|`critic_epochs`|`5`|
|`critic_minibatches`|`4`|
|`lr`|`0.0002`|
|`actor_lr_scale`|`1.0`|
|`lr_final_frac`|`0.03`|
|`lr_decay_iterations`|`30000`|
|`actor_ramp_iterations`|`1000`|
|`critic_warmup_iterations`|`500`|
|`grad_clip`|`1.0`|
|`grad_clip_elem`|`0.1`|
|`gamma`|`0.99`|
|`lam`|`0.95`|
|`action_noise`|`0.1`|
|`explore_frac`|`0.25`|
|`explore_noise`|`0.3`|
|`explore_theta`|`2.0`|
|`driver_policy`|`''`|
|`driver_frac`|`0.0`|
|`expert_policy`|`''`|
|`expert_frac`|`0.0`|
|`allow_missing_flat_expert`|`False`|
|`collapse_falls`|`3.0`|
|`log_interval`|`50`|
|`save_interval`|`1000`|
|`seed`|`17`|
|`dt`|`0.02`|
|`physics_dt`|`0.005`|
|`episode_seconds`|`20.0`|
|`payload_max_kg`|`2.0`|
|`reset_critic`|`False`|
|`evaluation_interval`|`5000`|
|`evaluation_python`|`''`|
|`compile`|`True`|
|`device`|`'cuda:0'`|

地形覆盖值与参数检查逻辑见config完整源码。所有CLI参数见launch/train/supervise源码；未使用参数不宣称生效。

## Go2完整资产物理参数

```json
[
  {
    "link": "base",
    "inertial": "<inertial>\n      <origin xyz=\"0.021112 0 -0.005366\" rpy=\"0 0 0\" />\n      <mass value=\"6.921\" />\n      <inertia ixx=\"0.02448\" ixy=\"0.00012166\" ixz=\"0.0014849\" iyy=\"0.098077\" iyz=\"-3.12E-05\" izz=\"0.107\" />\n    </inertial>\n    ",
    "collisions": [
      "<collision>\n      <origin rpy=\"0 0 0\" xyz=\"0 0 0\" />\n      <geometry>\n        <box size=\"0.3762 0.0935 0.114\" />\n      </geometry>\n    </collision>\n  "
    ]
  },
  {
    "link": "Head_upper",
    "inertial": "<inertial>\n      <origin xyz=\"0 0 0\" rpy=\"0 0 0\" />\n      <mass value=\"0.001\" />\n      <inertia ixx=\"9.6e-06\" ixy=\"0\" ixz=\"0\" iyy=\"9.6e-06\" iyz=\"0\" izz=\"9.6e-06\" />\n    </inertial>\n    ",
    "collisions": [
      "<collision>\n      <origin xyz=\"0 0 0\" rpy=\"0 0 0\" />\n      <geometry>\n        <cylinder radius=\"0.05\" length=\"0.09\" />\n      </geometry>\n    </collision>\n  "
    ]
  },
  {
    "link": "Head_lower",
    "inertial": "<inertial>\n      <origin xyz=\"0 0 0\" rpy=\"0 0 0\" />\n      <mass value=\"0.001\" />\n      <inertia ixx=\"9.6e-06\" ixy=\"0\" ixz=\"0\" iyy=\"9.6e-06\" iyz=\"0\" izz=\"9.6e-06\" />\n    </inertial>\n    ",
    "collisions": [
      "<collision>\n      <origin xyz=\"0 0 0\" rpy=\"0 0 0\" />\n      <geometry>\n        <sphere radius=\"0.047\" />\n      </geometry>\n    </collision>\n  "
    ]
  },
  {
    "link": "FL_hip",
    "inertial": "<inertial>\n      <origin xyz=\"-0.0054 0.00194 -0.000105\" rpy=\"0 0 0\" />\n      <mass value=\"0.678\" />\n      <inertia ixx=\"0.00048\" ixy=\"-3.01E-06\" ixz=\"1.11E-06\" iyy=\"0.000884\" iyz=\"-1.42E-06\" izz=\"0.000596\" />\n    </inertial>\n    ",
    "collisions": [
      "<collision>\n      <origin rpy=\"1.5707963267948966 0 0\" xyz=\"0 0.08 0\" />\n      <geometry>\n        <cylinder length=\"0.04\" radius=\"0.046\" />\n      </geometry>\n    </collision>\n  "
    ]
  },
  {
    "link": "FL_hip_rotor",
    "inertial": "<inertial>\n      <origin rpy=\"0 0 0\" xyz=\"0.0 0.0 0.0\" />\n      <mass value=\"0.089\" />\n      <inertia ixx=\"0.000111842\" ixy=\"0.0\" ixz=\"0.0\" iyy=\"0.000059647\" iyz=\"0.0\" izz=\"0.000059647\" />\n    </inertial>\n  ",
    "collisions": []
  },
  {
    "link": "FL_thigh",
    "inertial": "<inertial>\n      <origin xyz=\"-0.00374 -0.0223 -0.0327\" rpy=\"0 0 0\" />\n      <mass value=\"1.152\" />\n      <inertia ixx=\"0.00584\" ixy=\"8.72E-05\" ixz=\"-0.000289\" iyy=\"0.0058\" iyz=\"0.000808\" izz=\"0.00103\" />\n    </inertial>\n    ",
    "collisions": [
      "<collision>\n      <origin rpy=\"0 1.5707963267948966 0\" xyz=\"0 0 -0.1065\" />\n      <geometry>\n        <box size=\"0.213 0.0245 0.034\" />\n      </geometry>\n    </collision>\n  "
    ]
  },
  {
    "link": "FL_thigh_rotor",
    "inertial": "<inertial>\n      <origin rpy=\"0 0 0\" xyz=\"0.0 0.0 0.0\" />\n      <mass value=\"0.089\" />\n      <inertia ixx=\"0.000059647\" ixy=\"0.0\" ixz=\"0.0\" iyy=\"0.000111842\" iyz=\"0.0\" izz=\"0.000059647\" />\n    </inertial>\n  ",
    "collisions": []
  },
  {
    "link": "FL_calf",
    "inertial": "<inertial>\n      <origin xyz=\"0.00548 -0.000975 -0.115\" rpy=\"0 0 0\" />\n      <mass value=\"0.154\" />\n      <inertia ixx=\"0.00108\" ixy=\"3.4E-07\" ixz=\"1.72E-05\" iyy=\"0.0011\" iyz=\"8.28E-06\" izz=\"3.29E-05\" />\n    </inertial>\n    ",
    "collisions": [
      "<collision>\n      <origin rpy=\"0 -0.21 0\" xyz=\"0.008 0 -0.06\" />\n      <geometry>\n        <cylinder length=\"0.12\" radius=\"0.012\" />\n      </geometry>\n    </collision>\n  "
    ]
  },
  {
    "link": "FL_calflower",
    "inertial": null,
    "collisions": [
      "<collision>\n      <origin rpy=\"0 0 0\" xyz=\"0 0 0\" />\n      <geometry>\n        <cylinder length=\"0.065\" radius=\"0.011\" />\n      </geometry>\n    </collision>\n  "
    ]
  },
  {
    "link": "FL_calflower1",
    "inertial": null,
    "collisions": [
      "<collision>\n      <origin rpy=\"0 0 0\" xyz=\"0 0 0\" />\n      <geometry>\n        <cylinder length=\"0.03\" radius=\"0.0155\" />\n      </geometry>\n    </collision>\n    ",
      "<collision>\n      <origin rpy=\"0 0 0\" xyz=\"-0.004 0 0.013\" />\n      <geometry>\n        <cylinder length=\"0.034\" radius=\"0.0115\" />\n      </geometry>\n    </collision>\n  "
    ]
  },
  {
    "link": "FL_calf_rotor",
    "inertial": "<inertial>\n      <origin rpy=\"0 0 0\" xyz=\"0.0 0.0 0.0\" />\n      <mass value=\"0.089\" />\n      <inertia ixx=\"0.000059647\" ixy=\"0.0\" ixz=\"0.0\" iyy=\"0.000111842\" iyz=\"0.0\" izz=\"0.000059647\" />\n    </inertial>\n  ",
    "collisions": []
  },
  {
    "link": "FL_foot",
    "inertial": "<inertial>\n      <origin xyz=\"0 0 0\" rpy=\"0 0 0\" />\n      <mass value=\"0.04\" />\n      <inertia ixx=\"9.6e-06\" ixy=\"0\" ixz=\"0\" iyy=\"9.6e-06\" iyz=\"0\" izz=\"9.6e-06\" />\n    </inertial>\n    ",
    "collisions": [
      "<collision>\n      <origin rpy=\"0 0 0\" xyz=\"-0.002 0 0\" />\n      <geometry>\n        <sphere radius=\"0.022\" />\n      </geometry>\n    </collision>\n  "
    ]
  },
  {
    "link": "FR_hip",
    "inertial": "<inertial>\n      <origin xyz=\"-0.0054 -0.00194 -0.000105\" rpy=\"0 0 0\" />\n      <mass value=\"0.678\" />\n      <inertia ixx=\"0.00048\" ixy=\"3.01E-06\" ixz=\"1.11E-06\" iyy=\"0.000884\" iyz=\"1.42E-06\" izz=\"0.000596\" />\n    </inertial>\n    ",
    "collisions": [
      "<collision>\n      <origin rpy=\"1.5707963267948966 0 0\" xyz=\"0 -0.08 0\" />\n      <geometry>\n        <cylinder length=\"0.04\" radius=\"0.046\" />\n      </geometry>\n    </collision>\n  "
    ]
  },
  {
    "link": "FR_hip_rotor",
    "inertial": "<inertial>\n      <origin rpy=\"0 0 0\" xyz=\"0.0 0.0 0.0\" />\n      <mass value=\"0.089\" />\n      <inertia ixx=\"0.000111842\" ixy=\"0.0\" ixz=\"0.0\" iyy=\"0.000059647\" iyz=\"0.0\" izz=\"0.000059647\" />\n    </inertial>\n  ",
    "collisions": []
  },
  {
    "link": "FR_thigh",
    "inertial": "<inertial>\n      <origin xyz=\"-0.00374 0.0223 -0.0327\" rpy=\"0 0 0\" />\n      <mass value=\"1.152\" />\n      <inertia ixx=\"0.00584\" ixy=\"-8.72E-05\" ixz=\"-0.000289\" iyy=\"0.0058\" iyz=\"-0.000808\" izz=\"0.00103\" />\n    </inertial>\n    ",
    "collisions": [
      "<collision>\n      <origin rpy=\"0 1.5707963267948966 0\" xyz=\"0 0 -0.1065\" />\n      <geometry>\n        <box size=\"0.213 0.0245 0.034\" />\n      </geometry>\n    </collision>\n  "
    ]
  },
  {
    "link": "FR_thigh_rotor",
    "inertial": "<inertial>\n      <origin rpy=\"0 0 0\" xyz=\"0.0 0.0 0.0\" />\n      <mass value=\"0.089\" />\n      <inertia ixx=\"0.000059647\" ixy=\"0.0\" ixz=\"0.0\" iyy=\"0.000111842\" iyz=\"0.0\" izz=\"0.000059647\" />\n    </inertial>\n  ",
    "collisions": []
  },
  {
    "link": "FR_calf",
    "inertial": "<inertial>\n      <origin xyz=\"0.00548 0.000975 -0.115\" rpy=\"0 0 0\" />\n      <mass value=\"0.154\" />\n      <inertia ixx=\"0.00108\" ixy=\"-3.4E-07\" ixz=\"1.72E-05\" iyy=\"0.0011\" iyz=\"-8.28E-06\" izz=\"3.29E-05\" />\n    </inertial>\n    ",
    "collisions": [
      "<collision>\n      <origin rpy=\"0 -0.2 0\" xyz=\"0.01 0 -0.06\" />\n      <geometry>\n        <cylinder length=\"0.12\" radius=\"0.013\" />\n      </geometry>\n    </collision>\n  "
    ]
  },
  {
    "link": "FR_calflower",
    "inertial": null,
    "collisions": [
      "<collision>\n      <origin rpy=\"0 0 0\" xyz=\"0 0 0\" />\n      <geometry>\n        <cylinder length=\"0.065\" radius=\"0.011\" />\n      </geometry>\n    </collision>\n  "
    ]
  },
  {
    "link": "FR_calflower1",
    "inertial": null,
    "collisions": [
      "<collision>\n      <origin rpy=\"0 0 0\" xyz=\"0 0 0\" />\n      <geometry>\n        <cylinder length=\"0.03\" radius=\"0.0155\" />\n      </geometry>\n    </collision>\n    ",
      "<collision>\n      <origin rpy=\"0 0 0\" xyz=\"-0.004 0 0.013\" />\n      <geometry>\n        <cylinder length=\"0.034\" radius=\"0.0115\" />\n      </geometry>\n    </collision>\n  "
    ]
  },
  {
    "link": "FR_calf_rotor",
    "inertial": "<inertial>\n      <origin rpy=\"0 0 0\" xyz=\"0.0 0.0 0.0\" />\n      <mass value=\"0.089\" />\n      <inertia ixx=\"0.000059647\" ixy=\"0.0\" ixz=\"0.0\" iyy=\"0.000111842\" iyz=\"0.0\" izz=\"0.000059647\" />\n    </inertial>\n  ",
    "collisions": []
  },
  {
    "link": "FR_foot",
    "inertial": "<inertial>\n      <origin xyz=\"0 0 0\" rpy=\"0 0 0\" />\n      <mass value=\"0.04\" />\n      <inertia ixx=\"9.6e-06\" ixy=\"0\" ixz=\"0\" iyy=\"9.6e-06\" iyz=\"0\" izz=\"9.6e-06\" />\n    </inertial>\n    ",
    "collisions": [
      "<collision>\n      <origin rpy=\"0 0 0\" xyz=\"-0.002 0 0\" />\n      <geometry>\n        <sphere radius=\"0.022\" />\n      </geometry>\n    </collision>\n  "
    ]
  },
  {
    "link": "RL_hip",
    "inertial": "<inertial>\n      <origin xyz=\"0.0054 0.00194 -0.000105\" rpy=\"0 0 0\" />\n      <mass value=\"0.678\" />\n      <inertia ixx=\"0.00048\" ixy=\"3.01E-06\" ixz=\"-1.11E-06\" iyy=\"0.000884\" iyz=\"-1.42E-06\" izz=\"0.000596\" />\n    </inertial>\n    ",
    "collisions": [
      "<collision>\n      <origin rpy=\"1.5707963267948966 0 0\" xyz=\"0 0.08 0\" />\n      <geometry>\n        <cylinder length=\"0.04\" radius=\"0.046\" />\n      </geometry>\n    </collision>\n  "
    ]
  },
  {
    "link": "RL_hip_rotor",
    "inertial": "<inertial>\n      <origin rpy=\"0 0 0\" xyz=\"0.0 0.0 0.0\" />\n      <mass value=\"0.089\" />\n      <inertia ixx=\"0.000111842\" ixy=\"0.0\" ixz=\"0.0\" iyy=\"0.000059647\" iyz=\"0.0\" izz=\"0.000059647\" />\n    </inertial>\n  ",
    "collisions": []
  },
  {
    "link": "RL_thigh",
    "inertial": "<inertial>\n      <origin xyz=\"-0.00374 -0.0223 -0.0327\" rpy=\"0 0 0\" />\n      <mass value=\"1.152\" />\n      <inertia ixx=\"0.00584\" ixy=\"8.72E-05\" ixz=\"-0.000289\" iyy=\"0.0058\" iyz=\"0.000808\" izz=\"0.00103\" />\n    </inertial>\n    ",
    "collisions": [
      "<collision>\n      <origin rpy=\"0 1.5707963267948966 0\" xyz=\"0 0 -0.1065\" />\n      <geometry>\n        <box size=\"0.213 0.0245 0.034\" />\n      </geometry>\n    </collision>\n  "
    ]
  },
  {
    "link": "RL_thigh_rotor",
    "inertial": "<inertial>\n      <origin rpy=\"0 0 0\" xyz=\"0.0 0.0 0.0\" />\n      <mass value=\"0.089\" />\n      <inertia ixx=\"0.000059647\" ixy=\"0.0\" ixz=\"0.0\" iyy=\"0.000111842\" iyz=\"0.0\" izz=\"0.000059647\" />\n    </inertial>\n  ",
    "collisions": []
  },
  {
    "link": "RL_calf",
    "inertial": "<inertial>\n      <origin xyz=\"0.00548 -0.000975 -0.115\" rpy=\"0 0 0\" />\n      <mass value=\"0.154\" />\n      <inertia ixx=\"0.00108\" ixy=\"3.4E-07\" ixz=\"1.72E-05\" iyy=\"0.0011\" iyz=\"8.28E-06\" izz=\"3.29E-05\" />\n    </inertial>\n    ",
    "collisions": [
      "<collision>\n      <origin rpy=\"0 -0.2 0\" xyz=\"0.01 0 -0.06\" />\n      <geometry>\n        <cylinder length=\"0.12\" radius=\"0.013\" />\n      </geometry>\n    </collision>\n  "
    ]
  },
  {
    "link": "RL_calflower",
    "inertial": null,
    "collisions": [
      "<collision>\n      <origin rpy=\"0 0 0\" xyz=\"0 0 0\" />\n      <geometry>\n        <cylinder length=\"0.065\" radius=\"0.011\" />\n      </geometry>\n    </collision>\n  "
    ]
  },
  {
    "link": "RL_calflower1",
    "inertial": null,
    "collisions": [
      "<collision>\n      <origin rpy=\"0 0 0\" xyz=\"0 0 0\" />\n      <geometry>\n        <cylinder length=\"0.03\" radius=\"0.0155\" />\n      </geometry>\n    </collision>\n    ",
      "<collision>\n      <origin rpy=\"0 0 0\" xyz=\"-0.004 0 0.013\" />\n      <geometry>\n        <cylinder length=\"0.034\" radius=\"0.0115\" />\n      </geometry>\n    </collision>\n  "
    ]
  },
  {
    "link": "RL_calf_rotor",
    "inertial": "<inertial>\n      <origin rpy=\"0 0 0\" xyz=\"0.0 0.0 0.0\" />\n      <mass value=\"0.089\" />\n      <inertia ixx=\"0.000059647\" ixy=\"0.0\" ixz=\"0.0\" iyy=\"0.000111842\" iyz=\"0.0\" izz=\"0.000059647\" />\n    </inertial>\n  ",
    "collisions": []
  },
  {
    "link": "RL_foot",
    "inertial": "<inertial>\n      <origin xyz=\"0 0 0\" rpy=\"0 0 0\" />\n      <mass value=\"0.04\" />\n      <inertia ixx=\"9.6e-06\" ixy=\"0\" ixz=\"0\" iyy=\"9.6e-06\" iyz=\"0\" izz=\"9.6e-06\" />\n    </inertial>\n    ",
    "collisions": [
      "<collision>\n      <origin rpy=\"0 0 0\" xyz=\"-0.002 0 0\" />\n      <geometry>\n        <sphere radius=\"0.022\" />\n      </geometry>\n    </collision>\n  "
    ]
  },
  {
    "link": "RR_hip",
    "inertial": "<inertial>\n      <origin xyz=\"0.0054 -0.00194 -0.000105\" rpy=\"0 0 0\" />\n      <mass value=\"0.678\" />\n      <inertia ixx=\"0.00048\" ixy=\"-3.01E-06\" ixz=\"-1.11E-06\" iyy=\"0.000884\" iyz=\"1.42E-06\" izz=\"0.000596\" />\n    </inertial>\n    ",
    "collisions": [
      "<collision>\n      <origin rpy=\"1.5707963267948966 0 0\" xyz=\"0 -0.08 0\" />\n      <geometry>\n        <cylinder length=\"0.04\" radius=\"0.046\" />\n      </geometry>\n    </collision>\n  "
    ]
  },
  {
    "link": "RR_hip_rotor",
    "inertial": "<inertial>\n      <origin rpy=\"0 0 0\" xyz=\"0.0 0.0 0.0\" />\n      <mass value=\"0.089\" />\n      <inertia ixx=\"0.000111842\" ixy=\"0.0\" ixz=\"0.0\" iyy=\"0.000059647\" iyz=\"0.0\" izz=\"0.000059647\" />\n    </inertial>\n  ",
    "collisions": []
  },
  {
    "link": "RR_thigh",
    "inertial": "<inertial>\n      <origin xyz=\"-0.00374 0.0223 -0.0327\" rpy=\"0 0 0\" />\n      <mass value=\"1.152\" />\n      <inertia ixx=\"0.00584\" ixy=\"-8.72E-05\" ixz=\"-0.000289\" iyy=\"0.0058\" iyz=\"-0.000808\" izz=\"0.00103\" />\n    </inertial>\n    ",
    "collisions": [
      "<collision>\n      <origin rpy=\"0 1.5707963267948966 0\" xyz=\"0 0 -0.1065\" />\n      <geometry>\n        <box size=\"0.213 0.0245 0.034\" />\n      </geometry>\n    </collision>\n  "
    ]
  },
  {
    "link": "RR_thigh_rotor",
    "inertial": "<inertial>\n      <origin rpy=\"0 0 0\" xyz=\"0.0 0.0 0.0\" />\n      <mass value=\"0.089\" />\n      <inertia ixx=\"0.000059647\" ixy=\"0.0\" ixz=\"0.0\" iyy=\"0.000111842\" iyz=\"0.0\" izz=\"0.000059647\" />\n    </inertial>\n  ",
    "collisions": []
  },
  {
    "link": "RR_calf",
    "inertial": "<inertial>\n      <origin xyz=\"0.00548 0.000975 -0.115\" rpy=\"0 0 0\" />\n      <mass value=\"0.154\" />\n      <inertia ixx=\"0.00108\" ixy=\"-3.4E-07\" ixz=\"1.72E-05\" iyy=\"0.0011\" iyz=\"-8.28E-06\" izz=\"3.29E-05\" />\n    </inertial>\n    ",
    "collisions": [
      "<collision>\n      <origin rpy=\"0 -0.2 0\" xyz=\"0.01 0 -0.06\" />\n      <geometry>\n        <cylinder length=\"0.12\" radius=\"0.013\" />\n      </geometry>\n    </collision>\n  "
    ]
  },
  {
    "link": "RR_calflower",
    "inertial": null,
    "collisions": [
      "<collision>\n      <origin rpy=\"0 0 0\" xyz=\"0 0 0\" />\n      <geometry>\n        <cylinder length=\"0.065\" radius=\"0.011\" />\n      </geometry>\n    </collision>\n  "
    ]
  },
  {
    "link": "RR_calflower1",
    "inertial": null,
    "collisions": [
      "<collision>\n      <origin rpy=\"0 0 0\" xyz=\"0 0 0\" />\n      <geometry>\n        <cylinder length=\"0.03\" radius=\"0.0155\" />\n      </geometry>\n    </collision>\n    ",
      "<collision>\n      <origin rpy=\"0 0 0\" xyz=\"-0.004 0 0.013\" />\n      <geometry>\n        <cylinder length=\"0.034\" radius=\"0.0115\" />\n      </geometry>\n    </collision>\n  "
    ]
  },
  {
    "link": "RR_calf_rotor",
    "inertial": "<inertial>\n      <origin rpy=\"0 0 0\" xyz=\"0.0 0.0 0.0\" />\n      <mass value=\"0.089\" />\n      <inertia ixx=\"0.000059647\" ixy=\"0.0\" ixz=\"0.0\" iyy=\"0.000111842\" iyz=\"0.0\" izz=\"0.000059647\" />\n    </inertial>\n  ",
    "collisions": []
  },
  {
    "link": "RR_foot",
    "inertial": "<inertial>\n      <origin xyz=\"0 0 0\" rpy=\"0 0 0\" />\n      <mass value=\"0.04\" />\n      <inertia ixx=\"9.6e-06\" ixy=\"0\" ixz=\"0\" iyy=\"9.6e-06\" iyz=\"0\" izz=\"9.6e-06\" />\n    </inertial>\n    ",
    "collisions": [
      "<collision>\n      <origin rpy=\"0 0 0\" xyz=\"-0.002 0 0\" />\n      <geometry>\n        <sphere radius=\"0.022\" />\n      </geometry>\n    </collision>\n  "
    ]
  },
  {
    "link": "imu",
    "inertial": "<inertial>\n      <origin xyz=\"0 0 0\" rpy=\"0 0 0\" />\n      <mass value=\"0\" />\n      <inertia ixx=\"0\" ixy=\"0\" ixz=\"0\" iyy=\"0\" iyz=\"0\" izz=\"0\" />\n    </inertial>\n  ",
    "collisions": []
  },
  {
    "link": "radar",
    "inertial": "<inertial>\n      <origin xyz=\"0 0 0\" rpy=\"0 0 0\" />\n      <mass value=\"0\" />\n      <inertia ixx=\"0\" ixy=\"0\" ixz=\"0\" iyy=\"0\" iyz=\"0\" izz=\"0\" />\n    </inertial>\n  ",
    "collisions": []
  },
  {
    "link": "front_camera",
    "inertial": null,
    "collisions": []
  },
  {
    "joint": {
      "name": "Head_upper_joint",
      "type": "fixed",
      "dont_collapse": "true"
    },
    "definition": "<joint name=\"Head_upper_joint\" type=\"fixed\" dont_collapse=\"true\">\n    <origin xyz=\"0.285 0 0.01\" rpy=\"0 0 0\" />\n    <parent link=\"base\" />\n    <child link=\"Head_upper\" />\n    <axis xyz=\"0 0 0\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "Head_lower_joint",
      "type": "fixed",
      "dont_collapse": "true"
    },
    "definition": "<joint name=\"Head_lower_joint\" type=\"fixed\" dont_collapse=\"true\">\n    <origin xyz=\"0.008 0 -0.07\" rpy=\"0 0 0\" />\n    <parent link=\"Head_upper\" />\n    <child link=\"Head_lower\" />\n    <axis xyz=\"0 0 0\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "FL_hip_joint",
      "type": "revolute"
    },
    "definition": "<joint name=\"FL_hip_joint\" type=\"revolute\">\n    <origin xyz=\"0.1934 0.0465 0\" rpy=\"0 0 0\" />\n    <parent link=\"base\" />\n    <child link=\"FL_hip\" />\n    <axis xyz=\"1 0 0\" />\n    <limit lower=\"-1.0472\" upper=\"1.0472\" effort=\"23.7\" velocity=\"30.1\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "FL_hip_rotor_joint",
      "type": "fixed"
    },
    "definition": "<joint name=\"FL_hip_rotor_joint\" type=\"fixed\">\n    <origin rpy=\"0 0 0\" xyz=\"0.11215 0.04675 0\" />\n    <parent link=\"base\" />\n    <child link=\"FL_hip_rotor\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "FL_thigh_joint",
      "type": "revolute"
    },
    "definition": "<joint name=\"FL_thigh_joint\" type=\"revolute\">\n    <origin xyz=\"0 0.0955 0\" rpy=\"0 0 0\" />\n    <parent link=\"FL_hip\" />\n    <child link=\"FL_thigh\" />\n    <axis xyz=\"0 1 0\" />\n    <limit lower=\"-1.5708\" upper=\"3.4907\" effort=\"23.7\" velocity=\"30.1\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "FL_thigh_rotor_joint",
      "type": "fixed"
    },
    "definition": "<joint name=\"FL_thigh_rotor_joint\" type=\"fixed\">\n    <origin rpy=\"0 0 0\" xyz=\"0 -0.00015 0\" />\n    <parent link=\"FL_hip\" />\n    <child link=\"FL_thigh_rotor\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "FL_calf_joint",
      "type": "revolute"
    },
    "definition": "<joint name=\"FL_calf_joint\" type=\"revolute\">\n    <origin xyz=\"0 0 -0.213\" rpy=\"0 0 0\" />\n    <parent link=\"FL_thigh\" />\n    <child link=\"FL_calf\" />\n    <axis xyz=\"0 1 0\" />\n    <limit lower=\"-2.7227\" upper=\"-0.83776\" effort=\"45.43\" velocity=\"15.70\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "FL_calflower_joint",
      "type": "fixed"
    },
    "definition": "<joint name=\"FL_calflower_joint\" type=\"fixed\">\n    <origin xyz=\"0.020 0 -0.148\" rpy=\"0 0.05 0\" />\n    <parent link=\"FL_calf\" />\n    <child link=\"FL_calflower\" />\n    <axis xyz=\"0 0 0\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "FL_calflower1_joint",
      "type": "fixed"
    },
    "definition": "<joint name=\"FL_calflower1_joint\" type=\"fixed\">\n    <origin xyz=\"-0.01 0 -0.04\" rpy=\"0 0.48 0\" />\n    <parent link=\"FL_calflower\" />\n    <child link=\"FL_calflower1\" />\n    <axis xyz=\"0 0 0\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "FL_calf_rotor_joint",
      "type": "fixed"
    },
    "definition": "<joint name=\"FL_calf_rotor_joint\" type=\"fixed\">\n    <origin rpy=\"0 0 0\" xyz=\"0 -0.03235 0\" />\n    <parent link=\"FL_thigh\" />\n    <child link=\"FL_calf_rotor\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "FL_foot_joint",
      "type": "fixed",
      "dont_collapse": "true"
    },
    "definition": "<joint name=\"FL_foot_joint\" type=\"fixed\" dont_collapse=\"true\">\n    <origin xyz=\"0 0 -0.213\" rpy=\"0 0 0\" />\n    <parent link=\"FL_calf\" />\n    <child link=\"FL_foot\" />\n    <axis xyz=\"0 0 0\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "FR_hip_joint",
      "type": "revolute"
    },
    "definition": "<joint name=\"FR_hip_joint\" type=\"revolute\">\n    <origin xyz=\"0.1934 -0.0465 0\" rpy=\"0 0 0\" />\n    <parent link=\"base\" />\n    <child link=\"FR_hip\" />\n    <axis xyz=\"1 0 0\" />\n    <limit lower=\"-1.0472\" upper=\"1.0472\" effort=\"23.7\" velocity=\"30.1\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "FR_hip_rotor_joint",
      "type": "fixed"
    },
    "definition": "<joint name=\"FR_hip_rotor_joint\" type=\"fixed\">\n    <origin rpy=\"0 0 0\" xyz=\"0.11215 -0.04675 0\" />\n    <parent link=\"base\" />\n    <child link=\"FR_hip_rotor\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "FR_thigh_joint",
      "type": "revolute"
    },
    "definition": "<joint name=\"FR_thigh_joint\" type=\"revolute\">\n    <origin xyz=\"0 -0.0955 0\" rpy=\"0 0 0\" />\n    <parent link=\"FR_hip\" />\n    <child link=\"FR_thigh\" />\n    <axis xyz=\"0 1 0\" />\n    <limit lower=\"-1.5708\" upper=\"3.4907\" effort=\"23.7\" velocity=\"30.1\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "FR_thigh_rotor_joint",
      "type": "fixed"
    },
    "definition": "<joint name=\"FR_thigh_rotor_joint\" type=\"fixed\">\n    <origin rpy=\"0 0 0\" xyz=\"0 0.00015 0\" />\n    <parent link=\"FR_hip\" />\n    <child link=\"FR_thigh_rotor\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "FR_calf_joint",
      "type": "revolute"
    },
    "definition": "<joint name=\"FR_calf_joint\" type=\"revolute\">\n    <origin xyz=\"0 0 -0.213\" rpy=\"0 0 0\" />\n    <parent link=\"FR_thigh\" />\n    <child link=\"FR_calf\" />\n    <axis xyz=\"0 1 0\" />\n    <limit lower=\"-2.7227\" upper=\"-0.83776\" effort=\"45.43\" velocity=\"15.70\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "FR_calflower_joint",
      "type": "fixed"
    },
    "definition": "<joint name=\"FR_calflower_joint\" type=\"fixed\">\n    <origin xyz=\"0.020 0 -0.148\" rpy=\"0 0.05 0\" />\n    <parent link=\"FR_calf\" />\n    <child link=\"FR_calflower\" />\n    <axis xyz=\"0 0 0\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "FR_calflower1_joint",
      "type": "fixed"
    },
    "definition": "<joint name=\"FR_calflower1_joint\" type=\"fixed\">\n    <origin xyz=\"-0.01 0 -0.04\" rpy=\"0 0.48 0\" />\n    <parent link=\"FR_calflower\" />\n    <child link=\"FR_calflower1\" />\n    <axis xyz=\"0 0 0\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "FR_calf_rotor_joint",
      "type": "fixed"
    },
    "definition": "<joint name=\"FR_calf_rotor_joint\" type=\"fixed\">\n    <origin rpy=\"0 0 0\" xyz=\"0 0.03235 0\" />\n    <parent link=\"FR_thigh\" />\n    <child link=\"FR_calf_rotor\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "FR_foot_joint",
      "type": "fixed",
      "dont_collapse": "true"
    },
    "definition": "<joint name=\"FR_foot_joint\" type=\"fixed\" dont_collapse=\"true\">\n    <origin xyz=\"0 0 -0.213\" rpy=\"0 0 0\" />\n    <parent link=\"FR_calf\" />\n    <child link=\"FR_foot\" />\n    <axis xyz=\"0 0 0\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "RL_hip_joint",
      "type": "revolute"
    },
    "definition": "<joint name=\"RL_hip_joint\" type=\"revolute\">\n    <origin xyz=\"-0.1934 0.0465 0\" rpy=\"0 0 0\" />\n    <parent link=\"base\" />\n    <child link=\"RL_hip\" />\n    <axis xyz=\"1 0 0\" />\n    <limit lower=\"-1.0472\" upper=\"1.0472\" effort=\"23.7\" velocity=\"30.1\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "RL_hip_rotor_joint",
      "type": "fixed"
    },
    "definition": "<joint name=\"RL_hip_rotor_joint\" type=\"fixed\">\n    <origin rpy=\"0 0 0\" xyz=\"-0.11215 0.04675 0\" />\n    <parent link=\"base\" />\n    <child link=\"RL_hip_rotor\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "RL_thigh_joint",
      "type": "revolute"
    },
    "definition": "<joint name=\"RL_thigh_joint\" type=\"revolute\">\n    <origin xyz=\"0 0.0955 0\" rpy=\"0 0 0\" />\n    <parent link=\"RL_hip\" />\n    <child link=\"RL_thigh\" />\n    <axis xyz=\"0 1 0\" />\n    <limit lower=\"-0.5236\" upper=\"4.5379\" effort=\"23.7\" velocity=\"30.1\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "RL_thigh_rotor_joint",
      "type": "fixed"
    },
    "definition": "<joint name=\"RL_thigh_rotor_joint\" type=\"fixed\">\n    <origin rpy=\"0 0 0\" xyz=\"0 -0.00015 0\" />\n    <parent link=\"RL_hip\" />\n    <child link=\"RL_thigh_rotor\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "RL_calf_joint",
      "type": "revolute"
    },
    "definition": "<joint name=\"RL_calf_joint\" type=\"revolute\">\n    <origin xyz=\"0 0 -0.213\" rpy=\"0 0 0\" />\n    <parent link=\"RL_thigh\" />\n    <child link=\"RL_calf\" />\n    <axis xyz=\"0 1 0\" />\n    <limit lower=\"-2.7227\" upper=\"-0.83776\" effort=\"45.43\" velocity=\"15.70\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "RL_calflower_joint",
      "type": "fixed"
    },
    "definition": "<joint name=\"RL_calflower_joint\" type=\"fixed\">\n    <origin xyz=\"0.020 0 -0.148\" rpy=\"0 0.05 0\" />\n    <parent link=\"RL_calf\" />\n    <child link=\"RL_calflower\" />\n    <axis xyz=\"0 0 0\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "RL_calflower1_joint",
      "type": "fixed"
    },
    "definition": "<joint name=\"RL_calflower1_joint\" type=\"fixed\">\n    <origin xyz=\"-0.01 0 -0.04\" rpy=\"0 0.48 0\" />\n    <parent link=\"RL_calflower\" />\n    <child link=\"RL_calflower1\" />\n    <axis xyz=\"0 0 0\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "RL_calf_rotor_joint",
      "type": "fixed"
    },
    "definition": "<joint name=\"RL_calf_rotor_joint\" type=\"fixed\">\n    <origin rpy=\"0 0 0\" xyz=\"0 -0.03235 0\" />\n    <parent link=\"RL_thigh\" />\n    <child link=\"RL_calf_rotor\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "RL_foot_joint",
      "type": "fixed",
      "dont_collapse": "true"
    },
    "definition": "<joint name=\"RL_foot_joint\" type=\"fixed\" dont_collapse=\"true\">\n    <origin xyz=\"0 0 -0.213\" rpy=\"0 0 0\" />\n    <parent link=\"RL_calf\" />\n    <child link=\"RL_foot\" />\n    <axis xyz=\"0 0 0\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "RR_hip_joint",
      "type": "revolute"
    },
    "definition": "<joint name=\"RR_hip_joint\" type=\"revolute\">\n    <origin xyz=\"-0.1934 -0.0465 0\" rpy=\"0 0 0\" />\n    <parent link=\"base\" />\n    <child link=\"RR_hip\" />\n    <axis xyz=\"1 0 0\" />\n    <limit lower=\"-1.0472\" upper=\"1.0472\" effort=\"23.7\" velocity=\"30.1\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "RR_hip_rotor_joint",
      "type": "fixed"
    },
    "definition": "<joint name=\"RR_hip_rotor_joint\" type=\"fixed\">\n    <origin rpy=\"0 0 0\" xyz=\"-0.11215 -0.04675 0\" />\n    <parent link=\"base\" />\n    <child link=\"RR_hip_rotor\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "RR_thigh_joint",
      "type": "revolute"
    },
    "definition": "<joint name=\"RR_thigh_joint\" type=\"revolute\">\n    <origin xyz=\"0 -0.0955 0\" rpy=\"0 0 0\" />\n    <parent link=\"RR_hip\" />\n    <child link=\"RR_thigh\" />\n    <axis xyz=\"0 1 0\" />\n    <limit lower=\"-0.5236\" upper=\"4.5379\" effort=\"23.7\" velocity=\"30.1\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "RR_thigh_rotor_joint",
      "type": "fixed"
    },
    "definition": "<joint name=\"RR_thigh_rotor_joint\" type=\"fixed\">\n    <origin rpy=\"0 0 0\" xyz=\"0 0.00015 0\" />\n    <parent link=\"RR_hip\" />\n    <child link=\"RR_thigh_rotor\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "RR_calf_joint",
      "type": "revolute"
    },
    "definition": "<joint name=\"RR_calf_joint\" type=\"revolute\">\n    <origin xyz=\"0 0 -0.213\" rpy=\"0 0 0\" />\n    <parent link=\"RR_thigh\" />\n    <child link=\"RR_calf\" />\n    <axis xyz=\"0 1 0\" />\n    <limit lower=\"-2.7227\" upper=\"-0.83776\" effort=\"45.43\" velocity=\"15.70\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "RR_calflower_joint",
      "type": "fixed"
    },
    "definition": "<joint name=\"RR_calflower_joint\" type=\"fixed\">\n    <origin xyz=\"0.020 0 -0.148\" rpy=\"0 0.05 0\" />\n    <parent link=\"RR_calf\" />\n    <child link=\"RR_calflower\" />\n    <axis xyz=\"0 0 0\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "RR_calflower1_joint",
      "type": "fixed"
    },
    "definition": "<joint name=\"RR_calflower1_joint\" type=\"fixed\">\n    <origin xyz=\"-0.01 0 -0.04\" rpy=\"0 0.48 0\" />\n    <parent link=\"RR_calflower\" />\n    <child link=\"RR_calflower1\" />\n    <axis xyz=\"0 0 0\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "RR_calf_rotor_joint",
      "type": "fixed"
    },
    "definition": "<joint name=\"RR_calf_rotor_joint\" type=\"fixed\">\n    <origin rpy=\"0 0 0\" xyz=\"0 0.03235 0\" />\n    <parent link=\"RR_thigh\" />\n    <child link=\"RR_calf_rotor\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "RR_foot_joint",
      "type": "fixed",
      "dont_collapse": "true"
    },
    "definition": "<joint name=\"RR_foot_joint\" type=\"fixed\" dont_collapse=\"true\">\n    <origin xyz=\"0 0 -0.213\" rpy=\"0 0 0\" />\n    <parent link=\"RR_calf\" />\n    <child link=\"RR_foot\" />\n    <axis xyz=\"0 0 0\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "imu_joint",
      "type": "fixed"
    },
    "definition": "<joint name=\"imu_joint\" type=\"fixed\">\n    <origin xyz=\"-0.02557 0 0.04232\" rpy=\"0 0 0\" />\n    <parent link=\"base\" />\n    <child link=\"imu\" />\n    <axis xyz=\"0 0 0\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "radar_joint",
      "type": "fixed"
    },
    "definition": "<joint name=\"radar_joint\" type=\"fixed\">\n    <origin xyz=\"0.28945 0 -0.046825\" rpy=\"0 2.8782 0\" />\n    <parent link=\"base\" />\n    <child link=\"radar\" />\n    <axis xyz=\"0 0 0\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "front_camera_joint",
      "type": "fixed"
    },
    "definition": "<joint name=\"front_camera_joint\" type=\"fixed\">\n    <origin rpy=\"0 0 0\" xyz=\"0.32715 -0.00003 0.04297\" />\n      <parent link=\"base\" />\n      <child link=\"front_camera\" />\n  </joint>\n  "
  }
]
```

## As2完整资产物理参数

```json
[
  {
    "link": "world",
    "inertial": null,
    "collisions": []
  },
  {
    "link": "base_link",
    "inertial": "<inertial>\n      <origin xyz=\"0.045604 -0.0009 0.020368\" rpy=\"0 0 0\" />\n      <mass value=\"8.8\" />\n      <inertia ixx=\"0.044044\" ixy=\"1.959E-05\" ixz=\"-0.026181\" iyy=\"0.13739\" iyz=\"0.00030824\" izz=\"0.1462\" />\n    </inertial>\n    ",
    "collisions": [
      "<collision>\n      <origin xyz=\"-0.0 -0.0 0.006\" rpy=\"0 0 0\" />\n      <geometry>\n        <box size=\"0.2 0.224 0.14\" />\n      </geometry>\n    </collision>\n    ",
      "<collision>\n      <origin xyz=\"0.283 -0.0 0.078\" rpy=\"0 0 0\" />\n      <geometry>\n        <sphere radius=\"0.043\" />\n      </geometry>\n    </collision>\n    ",
      "<collision>\n      <origin xyz=\"0.23 -0.0 0.085\" rpy=\"0 1.68 0\" />\n      <geometry>\n        <cylinder radius=\"0.045\" length=\"0.06\" />\n      </geometry>\n    </collision>\n  "
    ]
  },
  {
    "link": "FL_hip",
    "inertial": "<inertial>\n      <origin xyz=\"-0.0046547 -0.0088505 3.4235E-05\" rpy=\"0 0 0\" />\n      <mass value=\"0.695\" />\n      <inertia ixx=\"0.00045574\" ixy=\"3.269E-05\" ixz=\"-1.7E-07\" iyy=\"0.00057398\" iyz=\"8.3E-07\" izz=\"0.0005503\" />\n    </inertial>\n    ",
    "collisions": []
  },
  {
    "link": "FL_thigh",
    "inertial": "<inertial>\n      <origin xyz=\"-0.0060186 -0.024079 -0.0374\" rpy=\"0 0 0\" />\n      <mass value=\"1.25\" />\n      <inertia ixx=\"0.0060449\" ixy=\"0.00020856\" ixz=\"-0.00063523\" iyy=\"0.0054351\" iyz=\"0.0010644\" izz=\"0.0015788\" />\n    </inertial>\n    ",
    "collisions": [
      "<collision>\n      <origin xyz=\"0.0 -0.03 0\" rpy=\"1.571 0 0\" />\n      <geometry>\n        <cylinder radius=\"0.04\" length=\"0.1\" />\n      </geometry>\n    </collision>\n    ",
      "<collision>\n      <origin xyz=\"-0.02 -0.00 -0.09\" rpy=\"0 0.1 0\" />\n      <geometry>\n        <box size=\"0.045 0.03 0.14\" />\n      </geometry>\n    </collision>\n    ",
      "<collision>\n      <origin xyz=\"-0.02 0.00 -0.21\" rpy=\"0 -0.77 0\" />\n      <geometry>\n        <box size=\"0.01 0.035 0.04\" />\n      </geometry>\n    </collision>\n  "
    ]
  },
  {
    "link": "FL_calf",
    "inertial": "<inertial>\n      <origin xyz=\"0.0078616 -7.7635E-08 -0.085034\" rpy=\"0 0 0\" />\n      <mass value=\"0.22\" />\n      <inertia ixx=\"0.0010957\" ixy=\"0\" ixz=\"0.00012446\" iyy=\"0.001123\" iyz=\"0\" izz=\"4.952E-05\" />\n    </inertial>\n    ",
    "collisions": [
      "<collision>\n      <origin xyz=\"0.011 -0.00 -0.08\" rpy=\"0 -0.285 0\" />\n      <geometry>\n        <cylinder radius=\"0.013\" length=\"0.05\" />\n      </geometry>\n    </collision>\n    ",
      "<collision>\n      <origin xyz=\"0.0215 -0.00 -0.13\" rpy=\"0 -0.1 0\" />\n      <geometry>\n        <cylinder radius=\"0.012\" length=\"0.06\" />\n      </geometry>\n    </collision>\n    ",
      "<collision>\n      <origin xyz=\"0.02 -0.00 -0.185\" rpy=\"0 0.15 0\" />\n      <geometry>\n        <cylinder radius=\"0.012\" length=\"0.055\" />\n      </geometry>\n    </collision>\n  "
    ]
  },
  {
    "link": "FL_foot",
    "inertial": "<inertial>\n      <origin xyz=\"0.0 0.0 0.0\" rpy=\"0 0 0\" />\n      <mass value=\"0.045\" />\n      <inertia ixx=\"1.0E-05\" ixy=\"0\" ixz=\"0\" iyy=\"1.0E-05\" iyz=\"0\" izz=\"1.0E-05\" />\n    </inertial>\n    ",
    "collisions": [
      "<collision>\n      <origin xyz=\"0.0 -0.00 -0.0\" rpy=\"0 0 0\" />\n      <geometry>\n        <sphere radius=\"0.028\" />\n      </geometry>\n    </collision>\n  "
    ]
  },
  {
    "link": "FR_hip",
    "inertial": "<inertial>\n      <origin xyz=\"-0.0046547 0.0088505 3.4234E-05\" rpy=\"0 0 0\" />\n      <mass value=\"0.695\" />\n      <inertia ixx=\"0.00045574\" ixy=\"-3.269E-05\" ixz=\"-1.7E-07\" iyy=\"0.00057398\" iyz=\"-8.3E-07\" izz=\"0.0005503\" />\n    </inertial>\n    ",
    "collisions": []
  },
  {
    "link": "FR_thigh",
    "inertial": "<inertial>\n      <origin xyz=\"-0.0060187 0.02408 -0.037399\" rpy=\"0 0 0\" />\n      <mass value=\"1.25\" />\n      <inertia ixx=\"0.006045\" ixy=\"-0.00020856\" ixz=\"-0.00063524\" iyy=\"0.0054353\" iyz=\"-0.0010645\" izz=\"0.0015788\" />\n    </inertial>\n    ",
    "collisions": [
      "<collision>\n      <origin xyz=\"0.0 0.03 0\" rpy=\"1.571 0 0\" />\n      <geometry>\n        <cylinder radius=\"0.04\" length=\"0.1\" />\n      </geometry>\n    </collision>\n    ",
      "<collision>\n      <origin xyz=\"-0.02 -0.00 -0.09\" rpy=\"0 0.1 0\" />\n      <geometry>\n        <box size=\"0.045 0.03 0.14\" />\n      </geometry>\n    </collision>\n    ",
      "<collision>\n      <origin xyz=\"-0.02 0.00 -0.21\" rpy=\"0 -0.77 0\" />\n      <geometry>\n        <box size=\"0.01 0.035 0.04\" />\n      </geometry>\n    </collision>\n  "
    ]
  },
  {
    "link": "FR_calf",
    "inertial": "<inertial>\n      <origin xyz=\"0.0078616 -8.926E-08 -0.085034\" rpy=\"0 0 0\" />\n      <mass value=\"0.22\" />\n      <inertia ixx=\"0.0010957\" ixy=\"0\" ixz=\"0.00012446\" iyy=\"0.001123\" iyz=\"0\" izz=\"4.952E-05\" />\n    </inertial>\n    ",
    "collisions": [
      "<collision>\n      <origin xyz=\"0.011 -0.00 -0.08\" rpy=\"0 -0.285 0\" />\n      <geometry>\n        <cylinder radius=\"0.013\" length=\"0.05\" />\n      </geometry>\n    </collision>\n    ",
      "<collision>\n      <origin xyz=\"0.0215 -0.00 -0.13\" rpy=\"0 -0.1 0\" />\n      <geometry>\n        <cylinder radius=\"0.012\" length=\"0.06\" />\n      </geometry>\n    </collision>\n    ",
      "<collision>\n      <origin xyz=\"0.02 -0.00 -0.185\" rpy=\"0 0.15 0\" />\n      <geometry>\n        <cylinder radius=\"0.012\" length=\"0.055\" />\n      </geometry>\n    </collision>\n  "
    ]
  },
  {
    "link": "FR_foot",
    "inertial": "<inertial>\n      <origin xyz=\"0.0 0.0 0.0\" rpy=\"0 0 0\" />\n      <mass value=\"0.045\" />\n      <inertia ixx=\"1.0E-05\" ixy=\"0\" ixz=\"0\" iyy=\"1.0E-05\" iyz=\"0\" izz=\"1.0E-05\" />\n    </inertial>\n    ",
    "collisions": [
      "<collision>\n      <origin xyz=\"0.0 -0.00 -0.0\" rpy=\"0 0 0\" />\n      <geometry>\n        <sphere radius=\"0.028\" />\n      </geometry>\n    </collision>\n  "
    ]
  },
  {
    "link": "RL_hip",
    "inertial": "<inertial>\n      <origin xyz=\"0.0046637 -0.0088505 -3.8901E-05\" rpy=\"0 0 0\" />\n      <mass value=\"0.695\" />\n      <inertia ixx=\"0.00045574\" ixy=\"-3.289E-05\" ixz=\"-1.8E-07\" iyy=\"0.00057392\" iyz=\"-7.3E-07\" izz=\"0.00055024\" />\n    </inertial>\n    ",
    "collisions": []
  },
  {
    "link": "RL_thigh",
    "inertial": "<inertial>\n      <origin xyz=\"-0.0068766 -0.022228 -0.039019\" rpy=\"0 0 0\" />\n      <mass value=\"1.25\" />\n      <inertia ixx=\"0.0061346\" ixy=\"0.00017144\" ixz=\"-0.00062209\" iyy=\"0.0055602\" iyz=\"0.0010266\" izz=\"0.0015189\" />\n    </inertial>\n    ",
    "collisions": [
      "<collision>\n      <origin xyz=\"0.0 -0.03 0\" rpy=\"1.571 0 0\" />\n      <geometry>\n        <cylinder radius=\"0.04\" length=\"0.1\" />\n      </geometry>\n    </collision>\n    ",
      "<collision>\n      <origin xyz=\"-0.02 -0.00 -0.09\" rpy=\"0 0.1 0\" />\n      <geometry>\n        <box size=\"0.045 0.03 0.14\" />\n      </geometry>\n    </collision>\n    ",
      "<collision>\n      <origin xyz=\"-0.02 0.00 -0.21\" rpy=\"0 -0.77 0\" />\n      <geometry>\n        <box size=\"0.01 0.035 0.04\" />\n      </geometry>\n    </collision>\n  "
    ]
  },
  {
    "link": "RL_calf",
    "inertial": "<inertial>\n      <origin xyz=\"0.0078616 -7.7491E-08 -0.085034\" rpy=\"0 0 0\" />\n      <mass value=\"0.22\" />\n      <inertia ixx=\"0.0010957\" ixy=\"0\" ixz=\"0.00012446\" iyy=\"0.001123\" iyz=\"0\" izz=\"4.952E-05\" />\n    </inertial>\n    ",
    "collisions": [
      "<collision>\n      <origin xyz=\"0.011 -0.00 -0.08\" rpy=\"0 -0.285 0\" />\n      <geometry>\n        <cylinder radius=\"0.013\" length=\"0.05\" />\n      </geometry>\n    </collision>\n    ",
      "<collision>\n      <origin xyz=\"0.0215 -0.00 -0.13\" rpy=\"0 -0.1 0\" />\n      <geometry>\n        <cylinder radius=\"0.012\" length=\"0.06\" />\n      </geometry>\n    </collision>\n    ",
      "<collision>\n      <origin xyz=\"0.02 -0.00 -0.185\" rpy=\"0 0.15 0\" />\n      <geometry>\n        <cylinder radius=\"0.012\" length=\"0.055\" />\n      </geometry>\n    </collision>\n  "
    ]
  },
  {
    "link": "RL_foot",
    "inertial": "<inertial>\n      <origin xyz=\"0.0 0.0 0.0\" rpy=\"0 0 0\" />\n      <mass value=\"0.045\" />\n      <inertia ixx=\"1.0E-05\" ixy=\"0\" ixz=\"0\" iyy=\"1.0E-05\" iyz=\"0\" izz=\"1.0E-05\" />\n    </inertial>\n    ",
    "collisions": [
      "<collision>\n      <origin xyz=\"0.0 -0.00 -0.0\" rpy=\"0 0 0\" />\n      <geometry>\n        <sphere radius=\"0.028\" />\n      </geometry>\n    </collision>\n  "
    ]
  },
  {
    "link": "RR_hip",
    "inertial": "<inertial>\n      <origin xyz=\"0.0046637 0.0088505 -3.89E-05\" rpy=\"0 0 0\" />\n      <mass value=\"0.695\" />\n      <inertia ixx=\"0.00045574\" ixy=\"3.289E-05\" ixz=\"-1.8E-07\" iyy=\"0.00057392\" iyz=\"7.3E-07\" izz=\"0.00055024\" />\n    </inertial>\n    ",
    "collisions": []
  },
  {
    "link": "RR_thigh",
    "inertial": "<inertial>\n      <origin xyz=\"-0.0068763 0.022228 -0.039019\" rpy=\"0 0 0\" />\n      <mass value=\"1.25\" />\n      <inertia ixx=\"0.0061347\" ixy=\"-0.00017144\" ixz=\"-0.00062209\" iyy=\"0.0055603\" iyz=\"-0.0010266\" izz=\"0.0015189\" />\n    </inertial>\n    ",
    "collisions": [
      "<collision>\n      <origin xyz=\"0.0 0.03 0\" rpy=\"1.571 0 0\" />\n      <geometry>\n        <cylinder radius=\"0.04\" length=\"0.1\" />\n      </geometry>\n    </collision>\n    ",
      "<collision>\n      <origin xyz=\"-0.02 -0.00 -0.09\" rpy=\"0 0.1 0\" />\n      <geometry>\n        <box size=\"0.045 0.03 0.14\" />\n      </geometry>\n    </collision>\n    ",
      "<collision>\n      <origin xyz=\"-0.02 0.00 -0.21\" rpy=\"0 -0.77 0\" />\n      <geometry>\n        <box size=\"0.01 0.035 0.04\" />\n      </geometry>\n    </collision>\n  "
    ]
  },
  {
    "link": "RR_calf",
    "inertial": "<inertial>\n      <origin xyz=\"0.0078616 -8.6653E-08 -0.085034\" rpy=\"0 0 0\" />\n      <mass value=\"0.22\" />\n      <inertia ixx=\"0.0010957\" ixy=\"0\" ixz=\"0.00012446\" iyy=\"0.001123\" iyz=\"0\" izz=\"4.952E-05\" />\n    </inertial>\n    ",
    "collisions": [
      "<collision>\n      <origin xyz=\"0.011 -0.00 -0.08\" rpy=\"0 -0.285 0\" />\n      <geometry>\n        <cylinder radius=\"0.013\" length=\"0.05\" />\n      </geometry>\n    </collision>\n    ",
      "<collision>\n      <origin xyz=\"0.0215 -0.00 -0.13\" rpy=\"0 -0.1 0\" />\n      <geometry>\n        <cylinder radius=\"0.012\" length=\"0.06\" />\n      </geometry>\n    </collision>\n    ",
      "<collision>\n      <origin xyz=\"0.02 -0.00 -0.185\" rpy=\"0 0.15 0\" />\n      <geometry>\n        <cylinder radius=\"0.012\" length=\"0.055\" />\n      </geometry>\n    </collision>\n  "
    ]
  },
  {
    "link": "RR_foot",
    "inertial": "<inertial>\n      <origin xyz=\"0.0 0.0 0.0\" rpy=\"0 0 0\" />\n      <mass value=\"0.045\" />\n      <inertia ixx=\"1.0E-05\" ixy=\"0\" ixz=\"0\" iyy=\"1.0E-05\" iyz=\"0\" izz=\"1.0E-05\" />\n    </inertial>\n    ",
    "collisions": [
      "<collision>\n      <origin xyz=\"0.0 -0.00 -0.0\" rpy=\"0 0 0\" />\n      <geometry>\n        <sphere radius=\"0.028\" />\n      </geometry>\n    </collision>\n  "
    ]
  },
  {
    "joint": {
      "name": "floating_base_joint",
      "type": "floating"
    },
    "definition": "<joint name=\"floating_base_joint\" type=\"floating\">\n    <parent link=\"world\" />\n    <child link=\"base_link\" />\n  </joint>\n\n  "
  },
  {
    "joint": {
      "name": "FL_hip_joint",
      "type": "revolute"
    },
    "definition": "<joint name=\"FL_hip_joint\" type=\"revolute\">\n    <origin xyz=\"0.2119 0.06 0\" rpy=\"0 0 0\" />\n    <parent link=\"base_link\" />\n    <child link=\"FL_hip\" />\n    <axis xyz=\"1 0 0\" />\n    <limit lower=\"-0.785\" upper=\"0.785\" effort=\"60.0\" velocity=\"24.0\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "FL_thigh_joint",
      "type": "revolute"
    },
    "definition": "<joint name=\"FL_thigh_joint\" type=\"revolute\">\n    <origin xyz=\"0 0.1054 0\" rpy=\"0 0 0\" />\n    <parent link=\"FL_hip\" />\n    <child link=\"FL_thigh\" />\n    <axis xyz=\"0 1 0\" />\n    <limit lower=\"-2\" upper=\"3.492\" effort=\"60.0\" velocity=\"24.0\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "FL_calf_joint",
      "type": "revolute"
    },
    "definition": "<joint name=\"FL_calf_joint\" type=\"revolute\">\n    <origin xyz=\"0 0 -0.212\" rpy=\"0 0 0\" />\n    <parent link=\"FL_thigh\" />\n    <child link=\"FL_calf\" />\n    <axis xyz=\"0 1 0\" />\n    <limit lower=\"-2.7677\" upper=\"-0.8133\" effort=\"90.0\" velocity=\"16.0\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "FL_foot_joint",
      "type": "fixed"
    },
    "definition": "<joint name=\"FL_foot_joint\" type=\"fixed\">\n    <origin xyz=\"0 0 -0.21344\" rpy=\"0 0 0\" />\n    <parent link=\"FL_calf\" />\n    <child link=\"FL_foot\" />\n    <axis xyz=\"0 0 0\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "FR_hip_joint",
      "type": "revolute"
    },
    "definition": "<joint name=\"FR_hip_joint\" type=\"revolute\">\n    <origin xyz=\"0.2119 -0.06 0\" rpy=\"0 0 0\" />\n    <parent link=\"base_link\" />\n    <child link=\"FR_hip\" />\n    <axis xyz=\"1 0 0\" />\n    <limit lower=\"-0.785\" upper=\"0.785\" effort=\"60.0\" velocity=\"24.0\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "FR_thigh_joint",
      "type": "revolute"
    },
    "definition": "<joint name=\"FR_thigh_joint\" type=\"revolute\">\n    <origin xyz=\"0 -0.1054 0\" rpy=\"0 0 0\" />\n    <parent link=\"FR_hip\" />\n    <child link=\"FR_thigh\" />\n    <axis xyz=\"0 1 0\" />\n    <limit lower=\"-2\" upper=\"3.492\" effort=\"60.0\" velocity=\"24.0\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "FR_calf_joint",
      "type": "revolute"
    },
    "definition": "<joint name=\"FR_calf_joint\" type=\"revolute\">\n    <origin xyz=\"0 0 -0.212\" rpy=\"0 0 0\" />\n    <parent link=\"FR_thigh\" />\n    <child link=\"FR_calf\" />\n    <axis xyz=\"0 1 0\" />\n    <limit lower=\"-2.7677\" upper=\"-0.8133\" effort=\"90.0\" velocity=\"16.0\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "FR_foot_joint",
      "type": "fixed"
    },
    "definition": "<joint name=\"FR_foot_joint\" type=\"fixed\">\n    <origin xyz=\"0 0 -0.21344\" rpy=\"0 0 0\" />\n    <parent link=\"FR_calf\" />\n    <child link=\"FR_foot\" />\n    <axis xyz=\"0 0 0\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "RL_hip_joint",
      "type": "revolute"
    },
    "definition": "<joint name=\"RL_hip_joint\" type=\"revolute\">\n    <origin xyz=\"-0.2119 0.06 0\" rpy=\"0 0 0\" />\n    <parent link=\"base_link\" />\n    <child link=\"RL_hip\" />\n    <axis xyz=\"1 0 0\" />\n    <limit lower=\"-0.785\" upper=\"0.785\" effort=\"60.0\" velocity=\"24.0\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "RL_thigh_joint",
      "type": "revolute"
    },
    "definition": "<joint name=\"RL_thigh_joint\" type=\"revolute\">\n    <origin xyz=\"0 0.1054 0\" rpy=\"0 0 0\" />\n    <parent link=\"RL_hip\" />\n    <child link=\"RL_thigh\" />\n    <axis xyz=\"0 1 0\" />\n    <limit lower=\"-1.3\" upper=\"4.19\" effort=\"60.0\" velocity=\"24.0\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "RL_calf_joint",
      "type": "revolute"
    },
    "definition": "<joint name=\"RL_calf_joint\" type=\"revolute\">\n    <origin xyz=\"0 0 -0.212\" rpy=\"0 0 0\" />\n    <parent link=\"RL_thigh\" />\n    <child link=\"RL_calf\" />\n    <axis xyz=\"0 1 0\" />\n    <limit lower=\"-2.7677\" upper=\"-0.8133\" effort=\"90.0\" velocity=\"16.0\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "RL_foot_joint",
      "type": "fixed"
    },
    "definition": "<joint name=\"RL_foot_joint\" type=\"fixed\">\n    <origin xyz=\"0 0 -0.21344\" rpy=\"0 0 0\" />\n    <parent link=\"RL_calf\" />\n    <child link=\"RL_foot\" />\n    <axis xyz=\"0 0 0\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "RR_hip_joint",
      "type": "revolute"
    },
    "definition": "<joint name=\"RR_hip_joint\" type=\"revolute\">\n    <origin xyz=\"-0.2119 -0.06 0\" rpy=\"0 0 0\" />\n    <parent link=\"base_link\" />\n    <child link=\"RR_hip\" />\n    <axis xyz=\"1 0 0\" />\n    <limit lower=\"-0.785\" upper=\"0.785\" effort=\"60.0\" velocity=\"24.0\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "RR_thigh_joint",
      "type": "revolute"
    },
    "definition": "<joint name=\"RR_thigh_joint\" type=\"revolute\">\n    <origin xyz=\"0 -0.1054 0\" rpy=\"0 0 0\" />\n    <parent link=\"RR_hip\" />\n    <child link=\"RR_thigh\" />\n    <axis xyz=\"0 1 0\" />\n    <limit lower=\"-1.3\" upper=\"4.19\" effort=\"60.0\" velocity=\"24.0\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "RR_calf_joint",
      "type": "revolute"
    },
    "definition": "<joint name=\"RR_calf_joint\" type=\"revolute\">\n    <origin xyz=\"0 0 -0.212\" rpy=\"0 0 0\" />\n    <parent link=\"RR_thigh\" />\n    <child link=\"RR_calf\" />\n    <axis xyz=\"0 1 0\" />\n    <limit lower=\"-2.7677\" upper=\"-0.8133\" effort=\"90.0\" velocity=\"16.0\" />\n  </joint>\n  "
  },
  {
    "joint": {
      "name": "RR_foot_joint",
      "type": "fixed"
    },
    "definition": "<joint name=\"RR_foot_joint\" type=\"fixed\">\n    <origin xyz=\"0 0 -0.21344\" rpy=\"0 0 0\" />\n    <parent link=\"RR_calf\" />\n    <child link=\"RR_foot\" />\n    <axis xyz=\"0 0 0\" />\n  </joint>\n"
  }
]
```

## 实测或依赖记录：assets/as2/asset_parameters.json

```json
{
  "nominal_mass_kg": 17.64,
  "inertials": [
    {
      "link": "base",
      "mass_kg": 8.8,
      "com_m": [
        0.045604,
        -0.0009,
        0.020368
      ],
      "inertia_kg_m2": [
        [
          0.044044,
          1.959e-05,
          -0.026181
        ],
        [
          1.959e-05,
          0.13739,
          0.00030824
        ],
        [
          -0.026181,
          0.00030824,
          0.1462
        ]
      ]
    },
    {
      "link": "FL_hip",
      "mass_kg": 0.695,
      "com_m": [
        -0.0046547,
        -0.0088505,
        3.4235e-05
      ],
      "inertia_kg_m2": [
        [
          0.00045574,
          3.269e-05,
          -1.7e-07
        ],
        [
          3.269e-05,
          0.00057398,
          8.3e-07
        ],
        [
          -1.7e-07,
          8.3e-07,
          0.0005503
        ]
      ]
    },
    {
      "link": "FL_thigh",
      "mass_kg": 1.25,
      "com_m": [
        -0.0060186,
        -0.024079,
        -0.0374
      ],
      "inertia_kg_m2": [
        [
          0.0060449,
          0.00020856,
          -0.00063523
        ],
        [
          0.00020856,
          0.0054351,
          0.0010644
        ],
        [
          -0.00063523,
          0.0010644,
          0.0015788
        ]
      ]
    },
    {
      "link": "FL_calf",
      "mass_kg": 0.22,
      "com_m": [
        0.0078616,
        -7.7635e-08,
        -0.085034
      ],
      "inertia_kg_m2": [
        [
          0.0010957,
          0.0,
          0.00012446
        ],
        [
          0.0,
          0.001123,
          0.0
        ],
        [
          0.00012446,
          0.0,
          4.952e-05
        ]
      ]
    },
    {
      "link": "FL_foot",
      "mass_kg": 0.045,
      "com_m": [
        0.0,
        0.0,
        0.0
      ],
      "inertia_kg_m2": [
        [
          1e-05,
          0.0,
          0.0
        ],
        [
          0.0,
          1e-05,
          0.0
        ],
        [
          0.0,
          0.0,
          1e-05
        ]
      ]
    },
    {
      "link": "FR_hip",
      "mass_kg": 0.695,
      "com_m": [
        -0.0046547,
        0.0088505,
        3.4234e-05
      ],
      "inertia_kg_m2": [
        [
          0.00045574,
          -3.269e-05,
          -1.7e-07
        ],
        [
          -3.269e-05,
          0.00057398,
          -8.3e-07
        ],
        [
          -1.7e-07,
          -8.3e-07,
          0.0005503
        ]
      ]
    },
    {
      "link": "FR_thigh",
      "mass_kg": 1.25,
      "com_m": [
        -0.0060187,
        0.02408,
        -0.037399
      ],
      "inertia_kg_m2": [
        [
          0.006045,
          -0.00020856,
          -0.00063524
        ],
        [
          -0.00020856,
          0.0054353,
          -0.0010645
        ],
        [
          -0.00063524,
          -0.0010645,
          0.0015788
        ]
      ]
    },
    {
      "link": "FR_calf",
      "mass_kg": 0.22,
      "com_m": [
        0.0078616,
        -8.926e-08,
        -0.085034
      ],
      "inertia_kg_m2": [
        [
          0.0010957,
          0.0,
          0.00012446
        ],
        [
          0.0,
          0.001123,
          0.0
        ],
        [
          0.00012446,
          0.0,
          4.952e-05
        ]
      ]
    },
    {
      "link": "FR_foot",
      "mass_kg": 0.045,
      "com_m": [
        0.0,
        0.0,
        0.0
      ],
      "inertia_kg_m2": [
        [
          1e-05,
          0.0,
          0.0
        ],
        [
          0.0,
          1e-05,
          0.0
        ],
        [
          0.0,
          0.0,
          1e-05
        ]
      ]
    },
    {
      "link": "RL_hip",
      "mass_kg": 0.695,
      "com_m": [
        0.0046637,
        -0.0088505,
        -3.8901e-05
      ],
      "inertia_kg_m2": [
        [
          0.00045574,
          -3.289e-05,
          -1.8e-07
        ],
        [
          -3.289e-05,
          0.00057392,
          -7.3e-07
        ],
        [
          -1.8e-07,
          -7.3e-07,
          0.00055024
        ]
      ]
    },
    {
      "link": "RL_thigh",
      "mass_kg": 1.25,
      "com_m": [
        -0.0068766,
        -0.022228,
        -0.039019
      ],
      "inertia_kg_m2": [
        [
          0.0061346,
          0.00017144,
          -0.00062209
        ],
        [
          0.00017144,
          0.0055602,
          0.0010266
        ],
        [
          -0.00062209,
          0.0010266,
          0.0015189
        ]
      ]
    },
    {
      "link": "RL_calf",
      "mass_kg": 0.22,
      "com_m": [
        0.0078616,
        -7.7491e-08,
        -0.085034
      ],
      "inertia_kg_m2": [
        [
          0.0010957,
          0.0,
          0.00012446
        ],
        [
          0.0,
          0.001123,
          0.0
        ],
        [
          0.00012446,
          0.0,
          4.952e-05
        ]
      ]
    },
    {
      "link": "RL_foot",
      "mass_kg": 0.045,
      "com_m": [
        0.0,
        0.0,
        0.0
      ],
      "inertia_kg_m2": [
        [
          1e-05,
          0.0,
          0.0
        ],
        [
          0.0,
          1e-05,
          0.0
        ],
        [
          0.0,
          0.0,
          1e-05
        ]
      ]
    },
    {
      "link": "RR_hip",
      "mass_kg": 0.695,
      "com_m": [
        0.0046637,
        0.0088505,
        -3.89e-05
      ],
      "inertia_kg_m2": [
        [
          0.00045574,
          3.289e-05,
          -1.8e-07
        ],
        [
          3.289e-05,
          0.00057392,
          7.3e-07
        ],
        [
          -1.8e-07,
          7.3e-07,
          0.00055024
        ]
      ]
    },
    {
      "link": "RR_thigh",
      "mass_kg": 1.25,
      "com_m": [
        -0.0068763,
        0.022228,
        -0.039019
      ],
      "inertia_kg_m2": [
        [
          0.0061347,
          -0.00017144,
          -0.00062209
        ],
        [
          -0.00017144,
          0.0055603,
          -0.0010266
        ],
        [
          -0.00062209,
          -0.0010266,
          0.0015189
        ]
      ]
    },
    {
      "link": "RR_calf",
      "mass_kg": 0.22,
      "com_m": [
        0.0078616,
        -8.6653e-08,
        -0.085034
      ],
      "inertia_kg_m2": [
        [
          0.0010957,
          0.0,
          0.00012446
        ],
        [
          0.0,
          0.001123,
          0.0
        ],
        [
          0.00012446,
          0.0,
          4.952e-05
        ]
      ]
    },
    {
      "link": "RR_foot",
      "mass_kg": 0.045,
      "com_m": [
        0.0,
        0.0,
        0.0
      ],
      "inertia_kg_m2": [
        [
          1e-05,
          0.0,
          0.0
        ],
        [
          0.0,
          1e-05,
          0.0
        ],
        [
          0.0,
          0.0,
          1e-05
        ]
      ]
    }
  ],
  "joint_parameters": [
    {
      "name": "FL_hip_joint",
      "lower": -0.785,
      "upper": 0.785,
      "effort": 60.0,
      "velocity": 24.0
    },
    {
      "name": "FL_thigh_joint",
      "lower": -2.0,
      "upper": 3.492,
      "effort": 60.0,
      "velocity": 24.0
    },
    {
      "name": "FL_calf_joint",
      "lower": -2.7677,
      "upper": -0.8133,
      "effort": 90.0,
      "velocity": 16.0
    },
    {
      "name": "FR_hip_joint",
      "lower": -0.785,
      "upper": 0.785,
      "effort": 60.0,
      "velocity": 24.0
    },
    {
      "name": "FR_thigh_joint",
      "lower": -2.0,
      "upper": 3.492,
      "effort": 60.0,
      "velocity": 24.0
    },
    {
      "name": "FR_calf_joint",
      "lower": -2.7677,
      "upper": -0.8133,
      "effort": 90.0,
      "velocity": 16.0
    },
    {
      "name": "RL_hip_joint",
      "lower": -0.785,
      "upper": 0.785,
      "effort": 60.0,
      "velocity": 24.0
    },
    {
      "name": "RL_thigh_joint",
      "lower": -1.3,
      "upper": 4.19,
      "effort": 60.0,
      "velocity": 24.0
    },
    {
      "name": "RL_calf_joint",
      "lower": -2.7677,
      "upper": -0.8133,
      "effort": 90.0,
      "velocity": 16.0
    },
    {
      "name": "RR_hip_joint",
      "lower": -0.785,
      "upper": 0.785,
      "effort": 60.0,
      "velocity": 24.0
    },
    {
      "name": "RR_thigh_joint",
      "lower": -1.3,
      "upper": 4.19,
      "effort": 60.0,
      "velocity": 24.0
    },
    {
      "name": "RR_calf_joint",
      "lower": -2.7677,
      "upper": -0.8133,
      "effort": 90.0,
      "velocity": 16.0
    }
  ],
  "collision_shapes": 31,
  "usd_sha256": "b0f287f97ff8b28cfb74884dbcbb9122e51b931541b2d6c2f0ec1138e73b3b91",
  "source_sha256": "76b500f7b8fd5ca619e8b426274064abe9bdc284601d5e17f22a79083188553d",
  "physics_runtime_verified": false
}
```

## 实测或依赖记录：assets/as2/runtime_check.json

```json
{
  "robot": "as2",
  "backend": "isaaclab_physx.assets.articulation.articulation",
  "nominal_mass_kg": [
    17.639999389648438,
    17.639999389648438,
    17.639999389648438,
    17.639999389648438
  ],
  "joint_names": [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint"
  ],
  "body_names": [
    "base",
    "FL_hip",
    "FR_hip",
    "RL_hip",
    "RR_hip",
    "FL_thigh",
    "FR_thigh",
    "RL_thigh",
    "RR_thigh",
    "FL_calf",
    "FR_calf",
    "RL_calf",
    "RR_calf",
    "FL_foot",
    "FR_foot",
    "RL_foot",
    "RR_foot"
  ],
  "consts": {
    "default_q": [
      -0.10000000149011612,
      0.8999999761581421,
      -1.7999999523162842,
      0.10000000149011612,
      0.8999999761581421,
      -1.7999999523162842,
      -0.10000000149011612,
      0.8999999761581421,
      -1.7999999523162842,
      0.10000000149011612,
      0.8999999761581421,
      -1.7999999523162842
    ],
    "action_scale": [
      0.3125,
      0.3125,
      0.3125,
      0.3125,
      0.3125,
      0.3125,
      0.3125,
      0.3125,
      0.3125,
      0.3125,
      0.3125,
      0.3125
    ],
    "kp": [
      40.0,
      40.0,
      60.0,
      40.0,
      40.0,
      60.0,
      40.0,
      40.0,
      60.0,
      40.0,
      40.0,
      60.0
    ],
    "kd": [
      2.0,
      2.0,
      3.0,
      2.0,
      2.0,
      3.0,
      2.0,
      2.0,
      3.0,
      2.0,
      2.0,
      3.0
    ],
    "effort": [
      50.0,
      50.0,
      75.0,
      50.0,
      50.0,
      75.0,
      50.0,
      50.0,
      75.0,
      50.0,
      50.0,
      75.0
    ],
    "soft_limits": [
      [
        -0.7064999938011169,
        0.7064999938011169
      ],
      [
        -1.7253998517990112,
        3.2173995971679688
      ],
      [
        -2.669980049133301,
        -0.9110199213027954
      ],
      [
        -0.7064999938011169,
        0.7064999938011169
      ],
      [
        -1.7253998517990112,
        3.2173995971679688
      ],
      [
        -2.669980049133301,
        -0.9110199213027954
      ],
      [
        -0.7064999938011169,
        0.7064999938011169
      ],
      [
        -1.025499701499939,
        3.915499687194824
      ],
      [
        -2.669980049133301,
        -0.9110199213027954
      ],
      [
        -0.7064999938011169,
        0.7064999938011169
      ],
      [
        -1.025499701499939,
        3.915499687194824
      ],
      [
        -2.669980049133301,
        -0.9110199213027954
      ]
    ]
  },
  "actual": {
    "body_mass": [
      [
        8.800000190734863,
        0.6949999928474426,
        0.6949999928474426,
        0.6949999928474426,
        0.6949999928474426,
        1.25,
        1.25,
        1.25,
        1.25,
        0.2199999988079071,
        0.2199999988079071,
        0.2199999988079071,
        0.2199999988079071,
        0.04500000178813934,
        0.04500000178813934,
        0.04500000178813934,
        0.04500000178813934
      ],
      [
        9.300000190734863,
        0.6949999928474426,
        0.6949999928474426,
        0.6949999928474426,
        0.6949999928474426,
        1.25,
        1.25,
        1.25,
        1.25,
        0.2199999988079071,
        0.2199999988079071,
        0.2199999988079071,
        0.2199999988079071,
        0.04500000178813934,
        0.04500000178813934,
        0.04500000178813934,
        0.04500000178813934
      ],
      [
        9.800000190734863,
        0.6949999928474426,
        0.6949999928474426,
        0.6949999928474426,
        0.6949999928474426,
        1.25,
        1.25,
        1.25,
        1.25,
        0.2199999988079071,
        0.2199999988079071,
        0.2199999988079071,
        0.2199999988079071,
        0.04500000178813934,
        0.04500000178813934,
        0.04500000178813934,
        0.04500000178813934
      ],
      [
        10.800000190734863,
        0.6949999928474426,
        0.6949999928474426,
        0.6949999928474426,
        0.6949999928474426,
        1.25,
        1.25,
        1.25,
        1.25,
        0.2199999988079071,
        0.2199999988079071,
        0.2199999988079071,
        0.2199999988079071,
        0.04500000178813934,
        0.04500000178813934,
        0.04500000178813934,
        0.04500000178813934
      ]
    ],
    "body_inertia": [
      [
        [
          0.044043999165296555,
          1.9589962903410196e-05,
          -0.02618100494146347,
          1.9589962903410196e-05,
          0.13739000260829926,
          0.0003082405310124159,
          -0.02618100866675377,
          0.0003082405310124159,
          0.146200031042099
        ],
        [
          0.00045573999523185194,
          3.269000444561243e-05,
          -1.7000752450258005e-07,
          3.269000444561243e-05,
          0.0005739799817092717,
          8.300012268591672e-07,
          -1.7000706975522917e-07,
          8.300012268591672e-07,
          0.0005502999993041158
        ],
        [
          0.00045573999523185194,
          -3.269000444561243e-05,
          -1.6999342733470257e-07,
          -3.269000444561243e-05,
          0.0005739799817092717,
          -8.300030458485708e-07,
          -1.699929725873517e-07,
          -8.300030458485708e-07,
          0.0005502999993041158
        ],
        [
          0.00045573999523185194,
          -3.289001324446872e-05,
          -1.8000673662754707e-07,
          -3.289001324446872e-05,
          0.0005739200860261917,
          -7.30001374904532e-07,
          -1.8000673662754707e-07,
          -7.300022843992338e-07,
          0.0005502400454133749
        ],
        [
          0.00045573999523185194,
          3.289001324446872e-05,
          -1.8001151147473138e-07,
          3.289001324446872e-05,
          0.0005739200860261917,
          7.30001374904532e-07,
          -1.8001151147473138e-07,
          7.30001374904532e-07,
          0.0005502400454133749
        ],
        [
          0.006044900510460138,
          0.0002085601445287466,
          -0.0006352303316816688,
          0.0002085601445287466,
          0.005435099825263023,
          0.001064400072209537,
          -0.0006352303316816688,
          0.0010643999557942152,
          0.0015788001473993063
        ],
        [
          0.0060450006276369095,
          -0.00020855996990576386,
          -0.0006352399359457195,
          -0.00020855991169810295,
          0.005435300059616566,
          -0.0010645001893863082,
          -0.0006352399359457195,
          -0.0010645000729709864,
          0.0015788000309839845
        ],
        [
          0.006134600844234228,
          0.00017143995501101017,
          -0.000622089602984488,
          0.00017143995501101017,
          0.0055601997300982475,
          0.0010266000172123313,
          -0.000622089602984488,
          0.0010266000172123313,
          0.0015188998077064753
        ],
        [
          0.006134700030088425,
          -0.000171440071426332,
          -0.0006220902432687581,
          -0.00017144010053016245,
          0.005560299847275019,
          -0.0010265997843816876,
          -0.0006220902432687581,
          -0.0010265999007970095,
          0.001518900040537119
        ],
        [
          0.0010957000777125359,
          -6.572833943785383e-11,
          0.00012446004257071763,
          -6.572833943785383e-11,
          0.0011230000527575612,
          5.922420887038982e-11,
          0.00012446005712263286,
          5.922420887038982e-11,
          4.952001108904369e-05
        ],
        [
          0.0010957000777125359,
          -6.572833943785383e-11,
          0.00012446004257071763,
          -6.572833943785383e-11,
          0.0011230000527575612,
          5.922420887038982e-11,
          0.00012446005712263286,
          5.922420887038982e-11,
          4.952001108904369e-05
        ],
        [
          0.0010957000777125359,
          -6.572833943785383e-11,
          0.00012446004257071763,
          -6.572833943785383e-11,
          0.0011230000527575612,
          5.922420887038982e-11,
          0.00012446005712263286,
          5.922420887038982e-11,
          4.952001108904369e-05
        ],
        [
          0.0010957000777125359,
          -6.572833943785383e-11,
          0.00012446004257071763,
          -6.572833943785383e-11,
          0.0011230000527575612,
          5.922420887038982e-11,
          0.00012446005712263286,
          5.922420887038982e-11,
          4.952001108904369e-05
        ],
        [
          9.999999747378752e-06,
          0.0,
          0.0,
          0.0,
          9.999999747378752e-06,
          0.0,
          0.0,
          0.0,
          9.999999747378752e-06
        ],
        [
          9.999999747378752e-06,
          0.0,
          0.0,
          0.0,
          9.999999747378752e-06,
          0.0,
          0.0,
          0.0,
          9.999999747378752e-06
        ],
        [
          9.999999747378752e-06,
          0.0,
          0.0,
          0.0,
          9.999999747378752e-06,
          0.0,
          0.0,
          0.0,
          9.999999747378752e-06
        ],
        [
          9.999999747378752e-06,
          0.0,
          0.0,
          0.0,
          9.999999747378752e-06,
          0.0,
          0.0,
          0.0,
          9.999999747378752e-06
        ]
      ],
      [
        [
          0.044043999165296555,
          1.9589962903410196e-05,
          -0.02618100494146347,
          1.9589962903410196e-05,
          0.13739000260829926,
          0.0003082405310124159,
          -0.02618100866675377,
          0.0003082405310124159,
          0.146200031042099
        ],
        [
          0.00045573999523185194,
          3.269000444561243e-05,
          -1.7000752450258005e-07,
          3.269000444561243e-05,
          0.0005739799817092717,
          8.300012268591672e-07,
          -1.7000706975522917e-07,
          8.300012268591672e-07,
          0.0005502999993041158
        ],
        [
          0.00045573999523185194,
          -3.269000444561243e-05,
          -1.6999342733470257e-07,
          -3.269000444561243e-05,
          0.0005739799817092717,
          -8.300030458485708e-07,
          -1.699929725873517e-07,
          -8.300030458485708e-07,
          0.0005502999993041158
        ],
        [
          0.00045573999523185194,
          -3.289001324446872e-05,
          -1.8000673662754707e-07,
          -3.289001324446872e-05,
          0.0005739200860261917,
          -7.30001374904532e-07,
          -1.8000673662754707e-07,
          -7.300022843992338e-07,
          0.0005502400454133749
        ],
        [
          0.00045573999523185194,
          3.289001324446872e-05,
          -1.8001151147473138e-07,
          3.289001324446872e-05,
          0.0005739200860261917,
          7.30001374904532e-07,
          -1.8001151147473138e-07,
          7.30001374904532e-07,
          0.0005502400454133749
        ],
        [
          0.006044900510460138,
          0.0002085601445287466,
          -0.0006352303316816688,
          0.0002085601445287466,
          0.005435099825263023,
          0.001064400072209537,
          -0.0006352303316816688,
          0.0010643999557942152,
          0.0015788001473993063
        ],
        [
          0.0060450006276369095,
          -0.00020855996990576386,
          -0.0006352399359457195,
          -0.00020855991169810295,
          0.005435300059616566,
          -0.0010645001893863082,
          -0.0006352399359457195,
          -0.0010645000729709864,
          0.0015788000309839845
        ],
        [
          0.006134600844234228,
          0.00017143995501101017,
          -0.000622089602984488,
          0.00017143995501101017,
          0.0055601997300982475,
          0.0010266000172123313,
          -0.000622089602984488,
          0.0010266000172123313,
          0.0015188998077064753
        ],
        [
          0.006134700030088425,
          -0.000171440071426332,
          -0.0006220902432687581,
          -0.00017144010053016245,
          0.005560299847275019,
          -0.0010265997843816876,
          -0.0006220902432687581,
          -0.0010265999007970095,
          0.001518900040537119
        ],
        [
          0.0010957000777125359,
          -6.572833943785383e-11,
          0.00012446004257071763,
          -6.572833943785383e-11,
          0.0011230000527575612,
          5.922420887038982e-11,
          0.00012446005712263286,
          5.922420887038982e-11,
          4.952001108904369e-05
        ],
        [
          0.0010957000777125359,
          -6.572833943785383e-11,
          0.00012446004257071763,
          -6.572833943785383e-11,
          0.0011230000527575612,
          5.922420887038982e-11,
          0.00012446005712263286,
          5.922420887038982e-11,
          4.952001108904369e-05
        ],
        [
          0.0010957000777125359,
          -6.572833943785383e-11,
          0.00012446004257071763,
          -6.572833943785383e-11,
          0.0011230000527575612,
          5.922420887038982e-11,
          0.00012446005712263286,
          5.922420887038982e-11,
          4.952001108904369e-05
        ],
        [
          0.0010957000777125359,
          -6.572833943785383e-11,
          0.00012446004257071763,
          -6.572833943785383e-11,
          0.0011230000527575612,
          5.922420887038982e-11,
          0.00012446005712263286,
          5.922420887038982e-11,
          4.952001108904369e-05
        ],
        [
          9.999999747378752e-06,
          0.0,
          0.0,
          0.0,
          9.999999747378752e-06,
          0.0,
          0.0,
          0.0,
          9.999999747378752e-06
        ],
        [
          9.999999747378752e-06,
          0.0,
          0.0,
          0.0,
          9.999999747378752e-06,
          0.0,
          0.0,
          0.0,
          9.999999747378752e-06
        ],
        [
          9.999999747378752e-06,
          0.0,
          0.0,
          0.0,
          9.999999747378752e-06,
          0.0,
          0.0,
          0.0,
          9.999999747378752e-06
        ],
        [
          9.999999747378752e-06,
          0.0,
          0.0,
          0.0,
          9.999999747378752e-06,
          0.0,
          0.0,
          0.0,
          9.999999747378752e-06
        ]
      ],
      [
        [
          0.044043999165296555,
          1.9589962903410196e-05,
          -0.02618100494146347,
          1.9589962903410196e-05,
          0.13739000260829926,
          0.0003082405310124159,
          -0.02618100866675377,
          0.0003082405310124159,
          0.146200031042099
        ],
        [
          0.00045573999523185194,
          3.269000444561243e-05,
          -1.7000752450258005e-07,
          3.269000444561243e-05,
          0.0005739799817092717,
          8.300012268591672e-07,
          -1.7000706975522917e-07,
          8.300012268591672e-07,
          0.0005502999993041158
        ],
        [
          0.00045573999523185194,
          -3.269000444561243e-05,
          -1.6999342733470257e-07,
          -3.269000444561243e-05,
          0.0005739799817092717,
          -8.300030458485708e-07,
          -1.699929725873517e-07,
          -8.300030458485708e-07,
          0.0005502999993041158
        ],
        [
          0.00045573999523185194,
          -3.289001324446872e-05,
          -1.8000673662754707e-07,
          -3.289001324446872e-05,
          0.0005739200860261917,
          -7.30001374904532e-07,
          -1.8000673662754707e-07,
          -7.300022843992338e-07,
          0.0005502400454133749
        ],
        [
          0.00045573999523185194,
          3.289001324446872e-05,
          -1.8001151147473138e-07,
          3.289001324446872e-05,
          0.0005739200860261917,
          7.30001374904532e-07,
          -1.8001151147473138e-07,
          7.30001374904532e-07,
          0.0005502400454133749
        ],
        [
          0.006044900510460138,
          0.0002085601445287466,
          -0.0006352303316816688,
          0.0002085601445287466,
          0.005435099825263023,
          0.001064400072209537,
          -0.0006352303316816688,
          0.0010643999557942152,
          0.0015788001473993063
        ],
        [
          0.0060450006276369095,
          -0.00020855996990576386,
          -0.0006352399359457195,
          -0.00020855991169810295,
          0.005435300059616566,
          -0.0010645001893863082,
          -0.0006352399359457195,
          -0.0010645000729709864,
          0.0015788000309839845
        ],
        [
          0.006134600844234228,
          0.00017143995501101017,
          -0.000622089602984488,
          0.00017143995501101017,
          0.0055601997300982475,
          0.0010266000172123313,
          -0.000622089602984488,
          0.0010266000172123313,
          0.0015188998077064753
        ],
        [
          0.006134700030088425,
          -0.000171440071426332,
          -0.0006220902432687581,
          -0.00017144010053016245,
          0.005560299847275019,
          -0.0010265997843816876,
          -0.0006220902432687581,
          -0.0010265999007970095,
          0.001518900040537119
        ],
        [
          0.0010957000777125359,
          -6.572833943785383e-11,
          0.00012446004257071763,
          -6.572833943785383e-11,
          0.0011230000527575612,
          5.922420887038982e-11,
          0.00012446005712263286,
          5.922420887038982e-11,
          4.952001108904369e-05
        ],
        [
          0.0010957000777125359,
          -6.572833943785383e-11,
          0.00012446004257071763,
          -6.572833943785383e-11,
          0.0011230000527575612,
          5.922420887038982e-11,
          0.00012446005712263286,
          5.922420887038982e-11,
          4.952001108904369e-05
        ],
        [
          0.0010957000777125359,
          -6.572833943785383e-11,
          0.00012446004257071763,
          -6.572833943785383e-11,
          0.0011230000527575612,
          5.922420887038982e-11,
          0.00012446005712263286,
          5.922420887038982e-11,
          4.952001108904369e-05
        ],
        [
          0.0010957000777125359,
          -6.572833943785383e-11,
          0.00012446004257071763,
          -6.572833943785383e-11,
          0.0011230000527575612,
          5.922420887038982e-11,
          0.00012446005712263286,
          5.922420887038982e-11,
          4.952001108904369e-05
        ],
        [
          9.999999747378752e-06,
          0.0,
          0.0,
          0.0,
          9.999999747378752e-06,
          0.0,
          0.0,
          0.0,
          9.999999747378752e-06
        ],
        [
          9.999999747378752e-06,
          0.0,
          0.0,
          0.0,
          9.999999747378752e-06,
          0.0,
          0.0,
          0.0,
          9.999999747378752e-06
        ],
        [
          9.999999747378752e-06,
          0.0,
          0.0,
          0.0,
          9.999999747378752e-06,
          0.0,
          0.0,
          0.0,
          9.999999747378752e-06
        ],
        [
          9.999999747378752e-06,
          0.0,
          0.0,
          0.0,
          9.999999747378752e-06,
          0.0,
          0.0,
          0.0,
          9.999999747378752e-06
        ]
      ],
      [
        [
          0.044043999165296555,
          1.9589962903410196e-05,
          -0.02618100494146347,
          1.9589962903410196e-05,
          0.13739000260829926,
          0.0003082405310124159,
          -0.02618100866675377,
          0.0003082405310124159,
          0.146200031042099
        ],
        [
          0.00045573999523185194,
          3.269000444561243e-05,
          -1.7000752450258005e-07,
          3.269000444561243e-05,
          0.0005739799817092717,
          8.300012268591672e-07,
          -1.7000706975522917e-07,
          8.300012268591672e-07,
          0.0005502999993041158
        ],
        [
          0.00045573999523185194,
          -3.269000444561243e-05,
          -1.6999342733470257e-07,
          -3.269000444561243e-05,
          0.0005739799817092717,
          -8.300030458485708e-07,
          -1.699929725873517e-07,
          -8.300030458485708e-07,
          0.0005502999993041158
        ],
        [
          0.00045573999523185194,
          -3.289001324446872e-05,
          -1.8000673662754707e-07,
          -3.289001324446872e-05,
          0.0005739200860261917,
          -7.30001374904532e-07,
          -1.8000673662754707e-07,
          -7.300022843992338e-07,
          0.0005502400454133749
        ],
        [
          0.00045573999523185194,
          3.289001324446872e-05,
          -1.8001151147473138e-07,
          3.289001324446872e-05,
          0.0005739200860261917,
          7.30001374904532e-07,
          -1.8001151147473138e-07,
          7.30001374904532e-07,
          0.0005502400454133749
        ],
        [
          0.006044900510460138,
          0.0002085601445287466,
          -0.0006352303316816688,
          0.0002085601445287466,
          0.005435099825263023,
          0.001064400072209537,
          -0.0006352303316816688,
          0.0010643999557942152,
          0.0015788001473993063
        ],
        [
          0.0060450006276369095,
          -0.00020855996990576386,
          -0.0006352399359457195,
          -0.00020855991169810295,
          0.005435300059616566,
          -0.0010645001893863082,
          -0.0006352399359457195,
          -0.0010645000729709864,
          0.0015788000309839845
        ],
        [
          0.006134600844234228,
          0.00017143995501101017,
          -0.000622089602984488,
          0.00017143995501101017,
          0.0055601997300982475,
          0.0010266000172123313,
          -0.000622089602984488,
          0.0010266000172123313,
          0.0015188998077064753
        ],
        [
          0.006134700030088425,
          -0.000171440071426332,
          -0.0006220902432687581,
          -0.00017144010053016245,
          0.005560299847275019,
          -0.0010265997843816876,
          -0.0006220902432687581,
          -0.0010265999007970095,
          0.001518900040537119
        ],
        [
          0.0010957000777125359,
          -6.572833943785383e-11,
          0.00012446004257071763,
          -6.572833943785383e-11,
          0.0011230000527575612,
          5.922420887038982e-11,
          0.00012446005712263286,
          5.922420887038982e-11,
          4.952001108904369e-05
        ],
        [
          0.0010957000777125359,
          -6.572833943785383e-11,
          0.00012446004257071763,
          -6.572833943785383e-11,
          0.0011230000527575612,
          5.922420887038982e-11,
          0.00012446005712263286,
          5.922420887038982e-11,
          4.952001108904369e-05
        ],
        [
          0.0010957000777125359,
          -6.572833943785383e-11,
          0.00012446004257071763,
          -6.572833943785383e-11,
          0.0011230000527575612,
          5.922420887038982e-11,
          0.00012446005712263286,
          5.922420887038982e-11,
          4.952001108904369e-05
        ],
        [
          0.0010957000777125359,
          -6.572833943785383e-11,
          0.00012446004257071763,
          -6.572833943785383e-11,
          0.0011230000527575612,
          5.922420887038982e-11,
          0.00012446005712263286,
          5.922420887038982e-11,
          4.952001108904369e-05
        ],
        [
          9.999999747378752e-06,
          0.0,
          0.0,
          0.0,
          9.999999747378752e-06,
          0.0,
          0.0,
          0.0,
          9.999999747378752e-06
        ],
        [
          9.999999747378752e-06,
          0.0,
          0.0,
          0.0,
          9.999999747378752e-06,
          0.0,
          0.0,
          0.0,
          9.999999747378752e-06
        ],
        [
          9.999999747378752e-06,
          0.0,
          0.0,
          0.0,
          9.999999747378752e-06,
          0.0,
          0.0,
          0.0,
          9.999999747378752e-06
        ],
        [
          9.999999747378752e-06,
          0.0,
          0.0,
          0.0,
          9.999999747378752e-06,
          0.0,
          0.0,
          0.0,
          9.999999747378752e-06
        ]
      ]
    ],
    "body_com_pose_b": [
      [
        [
          0.045604001730680466,
          -0.0008999999845400453,
          0.020368000492453575,
          0.9929504990577698,
          -0.0016064656665548682,
          0.11812762916088104,
          0.009623424150049686
        ],
        [
          -0.004654699936509132,
          -0.008850499987602234,
          3.4234999475302175e-05,
          -0.693281352519989,
          0.0867171660065651,
          -0.091359943151474,
          0.7095734477043152
        ],
        [
          -0.004654699936509132,
          0.008850499987602234,
          3.4233999031130224e-05,
          0.7095734477043152,
          0.091359943151474,
          -0.0867171660065651,
          -0.693281352519989
        ],
        [
          0.004663699772208929,
          -0.008850499987602234,
          -3.8900998333701864e-05,
          0.7084172368049622,
          0.09166879951953888,
          -0.08739705383777618,
          -0.6943368315696716
        ],
        [
          0.004663699772208929,
          0.008850499987602234,
          -3.890000152750872e-05,
          -0.6943368315696716,
          0.08739705383777618,
          -0.09166879951953888,
          0.7084172368049622
        ],
        [
          -0.006018599960952997,
          -0.02407900057733059,
          -0.03739999979734421,
          0.05066516622900963,
          0.7422234416007996,
          0.13225671648979187,
          0.6550157070159912
        ],
        [
          -0.006018700078129768,
          0.024080000817775726,
          -0.03739900141954422,
          -0.6550164818763733,
          0.13226424157619476,
          0.742221474647522,
          -0.05066408962011337
        ],
        [
          -0.006876599974930286,
          -0.022228000685572624,
          -0.03901899978518486,
          0.06042151898145676,
          0.7436626553535461,
          0.10779546201229095,
          0.6570351719856262
        ],
        [
          -0.00687630008906126,
          0.022228000685572624,
          -0.03901899978518486,
          -0.6570364236831665,
          0.10779619216918945,
          0.7436617612838745,
          -0.060417622327804565
        ],
        [
          0.007861600257456303,
          -7.763500065038897e-08,
          -0.08503399789333344,
          0.5283297300338745,
          0.4699656665325165,
          -0.4699656665325165,
          0.5283297300338745
        ],
        [
          0.007861600257456303,
          -8.925999850362132e-08,
          -0.08503399789333344,
          0.5283297300338745,
          0.4699656665325165,
          -0.4699656665325165,
          0.5283297300338745
        ],
        [
          0.007861600257456303,
          -7.749100205955983e-08,
          -0.08503399789333344,
          0.5283297300338745,
          0.4699656665325165,
          -0.4699656665325165,
          0.5283297300338745
        ],
        [
          0.007861600257456303,
          -8.66530029952628e-08,
          -0.08503399789333344,
          0.5283297300338745,
          0.4699656665325165,
          -0.4699656665325165,
          0.5283297300338745
        ],
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0
        ],
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0
        ],
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0
        ],
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0
        ]
      ],
      [
        [
          0.045604001730680466,
          -0.0008999999845400453,
          0.020368000492453575,
          0.9929504990577698,
          -0.0016064656665548682,
          0.11812762916088104,
          0.009623424150049686
        ],
        [
          -0.004654699936509132,
          -0.008850499987602234,
          3.4234999475302175e-05,
          -0.693281352519989,
          0.0867171660065651,
          -0.091359943151474,
          0.7095734477043152
        ],
        [
          -0.004654699936509132,
          0.008850499987602234,
          3.4233999031130224e-05,
          0.7095734477043152,
          0.091359943151474,
          -0.0867171660065651,
          -0.693281352519989
        ],
        [
          0.004663699772208929,
          -0.008850499987602234,
          -3.8900998333701864e-05,
          0.7084172368049622,
          0.09166879951953888,
          -0.08739705383777618,
          -0.6943368315696716
        ],
        [
          0.004663699772208929,
          0.008850499987602234,
          -3.890000152750872e-05,
          -0.6943368315696716,
          0.08739705383777618,
          -0.09166879951953888,
          0.7084172368049622
        ],
        [
          -0.006018599960952997,
          -0.02407900057733059,
          -0.03739999979734421,
          0.05066516622900963,
          0.7422234416007996,
          0.13225671648979187,
          0.6550157070159912
        ],
        [
          -0.006018700078129768,
          0.024080000817775726,
          -0.03739900141954422,
          -0.6550164818763733,
          0.13226424157619476,
          0.742221474647522,
          -0.05066408962011337
        ],
        [
          -0.006876599974930286,
          -0.022228000685572624,
          -0.03901899978518486,
          0.06042151898145676,
          0.7436626553535461,
          0.10779546201229095,
          0.6570351719856262
        ],
        [
          -0.00687630008906126,
          0.022228000685572624,
          -0.03901899978518486,
          -0.6570364236831665,
          0.10779619216918945,
          0.7436617612838745,
          -0.060417622327804565
        ],
        [
          0.007861600257456303,
          -7.763500065038897e-08,
          -0.08503399789333344,
          0.5283297300338745,
          0.4699656665325165,
          -0.4699656665325165,
          0.5283297300338745
        ],
        [
          0.007861600257456303,
          -8.925999850362132e-08,
          -0.08503399789333344,
          0.5283297300338745,
          0.4699656665325165,
          -0.4699656665325165,
          0.5283297300338745
        ],
        [
          0.007861600257456303,
          -7.749100205955983e-08,
          -0.08503399789333344,
          0.5283297300338745,
          0.4699656665325165,
          -0.4699656665325165,
          0.5283297300338745
        ],
        [
          0.007861600257456303,
          -8.66530029952628e-08,
          -0.08503399789333344,
          0.5283297300338745,
          0.4699656665325165,
          -0.4699656665325165,
          0.5283297300338745
        ],
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0
        ],
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0
        ],
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0
        ],
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0
        ]
      ],
      [
        [
          0.045604001730680466,
          -0.0008999999845400453,
          0.020368000492453575,
          0.9929504990577698,
          -0.0016064656665548682,
          0.11812762916088104,
          0.009623424150049686
        ],
        [
          -0.004654699936509132,
          -0.008850499987602234,
          3.4234999475302175e-05,
          -0.693281352519989,
          0.0867171660065651,
          -0.091359943151474,
          0.7095734477043152
        ],
        [
          -0.004654699936509132,
          0.008850499987602234,
          3.4233999031130224e-05,
          0.7095734477043152,
          0.091359943151474,
          -0.0867171660065651,
          -0.693281352519989
        ],
        [
          0.004663699772208929,
          -0.008850499987602234,
          -3.8900998333701864e-05,
          0.7084172368049622,
          0.09166879951953888,
          -0.08739705383777618,
          -0.6943368315696716
        ],
        [
          0.004663699772208929,
          0.008850499987602234,
          -3.890000152750872e-05,
          -0.6943368315696716,
          0.08739705383777618,
          -0.09166879951953888,
          0.7084172368049622
        ],
        [
          -0.006018599960952997,
          -0.02407900057733059,
          -0.03739999979734421,
          0.05066516622900963,
          0.7422234416007996,
          0.13225671648979187,
          0.6550157070159912
        ],
        [
          -0.006018700078129768,
          0.024080000817775726,
          -0.03739900141954422,
          -0.6550164818763733,
          0.13226424157619476,
          0.742221474647522,
          -0.05066408962011337
        ],
        [
          -0.006876599974930286,
          -0.022228000685572624,
          -0.03901899978518486,
          0.06042151898145676,
          0.7436626553535461,
          0.10779546201229095,
          0.6570351719856262
        ],
        [
          -0.00687630008906126,
          0.022228000685572624,
          -0.03901899978518486,
          -0.6570364236831665,
          0.10779619216918945,
          0.7436617612838745,
          -0.060417622327804565
        ],
        [
          0.007861600257456303,
          -7.763500065038897e-08,
          -0.08503399789333344,
          0.5283297300338745,
          0.4699656665325165,
          -0.4699656665325165,
          0.5283297300338745
        ],
        [
          0.007861600257456303,
          -8.925999850362132e-08,
          -0.08503399789333344,
          0.5283297300338745,
          0.4699656665325165,
          -0.4699656665325165,
          0.5283297300338745
        ],
        [
          0.007861600257456303,
          -7.749100205955983e-08,
          -0.08503399789333344,
          0.5283297300338745,
          0.4699656665325165,
          -0.4699656665325165,
          0.5283297300338745
        ],
        [
          0.007861600257456303,
          -8.66530029952628e-08,
          -0.08503399789333344,
          0.5283297300338745,
          0.4699656665325165,
          -0.4699656665325165,
          0.5283297300338745
        ],
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0
        ],
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0
        ],
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0
        ],
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0
        ]
      ],
      [
        [
          0.045604001730680466,
          -0.0008999999845400453,
          0.020368000492453575,
          0.9929504990577698,
          -0.0016064656665548682,
          0.11812762916088104,
          0.009623424150049686
        ],
        [
          -0.004654699936509132,
          -0.008850499987602234,
          3.4234999475302175e-05,
          -0.693281352519989,
          0.0867171660065651,
          -0.091359943151474,
          0.7095734477043152
        ],
        [
          -0.004654699936509132,
          0.008850499987602234,
          3.4233999031130224e-05,
          0.7095734477043152,
          0.091359943151474,
          -0.0867171660065651,
          -0.693281352519989
        ],
        [
          0.004663699772208929,
          -0.008850499987602234,
          -3.8900998333701864e-05,
          0.7084172368049622,
          0.09166879951953888,
          -0.08739705383777618,
          -0.6943368315696716
        ],
        [
          0.004663699772208929,
          0.008850499987602234,
          -3.890000152750872e-05,
          -0.6943368315696716,
          0.08739705383777618,
          -0.09166879951953888,
          0.7084172368049622
        ],
        [
          -0.006018599960952997,
          -0.02407900057733059,
          -0.03739999979734421,
          0.05066516622900963,
          0.7422234416007996,
          0.13225671648979187,
          0.6550157070159912
        ],
        [
          -0.006018700078129768,
          0.024080000817775726,
          -0.03739900141954422,
          -0.6550164818763733,
          0.13226424157619476,
          0.742221474647522,
          -0.05066408962011337
        ],
        [
          -0.006876599974930286,
          -0.022228000685572624,
          -0.03901899978518486,
          0.06042151898145676,
          0.7436626553535461,
          0.10779546201229095,
          0.6570351719856262
        ],
        [
          -0.00687630008906126,
          0.022228000685572624,
          -0.03901899978518486,
          -0.6570364236831665,
          0.10779619216918945,
          0.7436617612838745,
          -0.060417622327804565
        ],
        [
          0.007861600257456303,
          -7.763500065038897e-08,
          -0.08503399789333344,
          0.5283297300338745,
          0.4699656665325165,
          -0.4699656665325165,
          0.5283297300338745
        ],
        [
          0.007861600257456303,
          -8.925999850362132e-08,
          -0.08503399789333344,
          0.5283297300338745,
          0.4699656665325165,
          -0.4699656665325165,
          0.5283297300338745
        ],
        [
          0.007861600257456303,
          -7.749100205955983e-08,
          -0.08503399789333344,
          0.5283297300338745,
          0.4699656665325165,
          -0.4699656665325165,
          0.5283297300338745
        ],
        [
          0.007861600257456303,
          -8.66530029952628e-08,
          -0.08503399789333344,
          0.5283297300338745,
          0.4699656665325165,
          -0.4699656665325165,
          0.5283297300338745
        ],
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0
        ],
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0
        ],
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0
        ],
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0
        ]
      ]
    ],
    "joint_pos_limits": [
      [
        [
          -0.7850000262260437,
          0.7850000262260437
        ],
        [
          -2.0,
          3.491999864578247
        ],
        [
          -2.767699956893921,
          -0.8132999539375305
        ],
        [
          -0.7850000262260437,
          0.7850000262260437
        ],
        [
          -2.0,
          3.491999864578247
        ],
        [
          -2.767699956893921,
          -0.8132999539375305
        ],
        [
          -0.7850000262260437,
          0.7850000262260437
        ],
        [
          -1.2999999523162842,
          4.190000057220459
        ],
        [
          -2.767699956893921,
          -0.8132999539375305
        ],
        [
          -0.7850000262260437,
          0.7850000262260437
        ],
        [
          -1.2999999523162842,
          4.190000057220459
        ],
        [
          -2.767699956893921,
          -0.8132999539375305
        ]
      ],
      [
        [
          -0.7850000262260437,
          0.7850000262260437
        ],
        [
          -2.0,
          3.491999864578247
        ],
        [
          -2.767699956893921,
          -0.8132999539375305
        ],
        [
          -0.7850000262260437,
          0.7850000262260437
        ],
        [
          -2.0,
          3.491999864578247
        ],
        [
          -2.767699956893921,
          -0.8132999539375305
        ],
        [
          -0.7850000262260437,
          0.7850000262260437
        ],
        [
          -1.2999999523162842,
          4.190000057220459
        ],
        [
          -2.767699956893921,
          -0.8132999539375305
        ],
        [
          -0.7850000262260437,
          0.7850000262260437
        ],
        [
          -1.2999999523162842,
          4.190000057220459
        ],
        [
          -2.767699956893921,
          -0.8132999539375305
        ]
      ],
      [
        [
          -0.7850000262260437,
          0.7850000262260437
        ],
        [
          -2.0,
          3.491999864578247
        ],
        [
          -2.767699956893921,
          -0.8132999539375305
        ],
        [
          -0.7850000262260437,
          0.7850000262260437
        ],
        [
          -2.0,
          3.491999864578247
        ],
        [
          -2.767699956893921,
          -0.8132999539375305
        ],
        [
          -0.7850000262260437,
          0.7850000262260437
        ],
        [
          -1.2999999523162842,
          4.190000057220459
        ],
        [
          -2.767699956893921,
          -0.8132999539375305
        ],
        [
          -0.7850000262260437,
          0.7850000262260437
        ],
        [
          -1.2999999523162842,
          4.190000057220459
        ],
        [
          -2.767699956893921,
          -0.8132999539375305
        ]
      ],
      [
        [
          -0.7850000262260437,
          0.7850000262260437
        ],
        [
          -2.0,
          3.491999864578247
        ],
        [
          -2.767699956893921,
          -0.8132999539375305
        ],
        [
          -0.7850000262260437,
          0.7850000262260437
        ],
        [
          -2.0,
          3.491999864578247
        ],
        [
          -2.767699956893921,
          -0.8132999539375305
        ],
        [
          -0.7850000262260437,
          0.7850000262260437
        ],
        [
          -1.2999999523162842,
          4.190000057220459
        ],
        [
          -2.767699956893921,
          -0.8132999539375305
        ],
        [
          -0.7850000262260437,
          0.7850000262260437
        ],
        [
          -1.2999999523162842,
          4.190000057220459
        ],
        [
          -2.767699956893921,
          -0.8132999539375305
        ]
      ]
    ],
    "joint_vel_limits": [
      [
        24.0,
        24.0,
        16.0,
        24.0,
        24.0,
        16.0,
        24.0,
        24.0,
        16.0,
        24.0,
        24.0,
        16.0
      ],
      [
        24.0,
        24.0,
        16.0,
        24.0,
        24.0,
        16.0,
        24.0,
        24.0,
        16.0,
        24.0,
        24.0,
        16.0
      ],
      [
        24.0,
        24.0,
        16.0,
        24.0,
        24.0,
        16.0,
        24.0,
        24.0,
        16.0,
        24.0,
        24.0,
        16.0
      ],
      [
        24.0,
        24.0,
        16.0,
        24.0,
        24.0,
        16.0,
        24.0,
        24.0,
        16.0,
        24.0,
        24.0,
        16.0
      ]
    ],
    "joint_effort_limits": [
      [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
      ],
      [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
      ],
      [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
      ],
      [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
      ]
    ],
    "joint_armature": [
      [
        0.026000000536441803,
        0.026000000536441803,
        0.03799999877810478,
        0.026000000536441803,
        0.026000000536441803,
        0.03799999877810478,
        0.026000000536441803,
        0.026000000536441803,
        0.03799999877810478,
        0.026000000536441803,
        0.026000000536441803,
        0.03799999877810478
      ],
      [
        0.026000000536441803,
        0.026000000536441803,
        0.03799999877810478,
        0.026000000536441803,
        0.026000000536441803,
        0.03799999877810478,
        0.026000000536441803,
        0.026000000536441803,
        0.03799999877810478,
        0.026000000536441803,
        0.026000000536441803,
        0.03799999877810478
      ],
      [
        0.026000000536441803,
        0.026000000536441803,
        0.03799999877810478,
        0.026000000536441803,
        0.026000000536441803,
        0.03799999877810478,
        0.026000000536441803,
        0.026000000536441803,
        0.03799999877810478,
        0.026000000536441803,
        0.026000000536441803,
        0.03799999877810478
      ],
      [
        0.026000000536441803,
        0.026000000536441803,
        0.03799999877810478,
        0.026000000536441803,
        0.026000000536441803,
        0.03799999877810478,
        0.026000000536441803,
        0.026000000536441803,
        0.03799999877810478,
        0.026000000536441803,
        0.026000000536441803,
        0.03799999877810478
      ]
    ]
  },
  "control_steps": 50,
  "falls": 0,
  "policy_updates": 0,
  "capability_claim": false
}
```

## 实测或依赖记录：assets/go2/runtime_check.json

```json
{
  "robot": "go2",
  "backend": "isaaclab_physx.assets.articulation.articulation",
  "nominal_mass_kg": [
    16.086999893188477,
    16.086999893188477,
    16.086999893188477,
    16.086999893188477
  ],
  "joint_names": [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint"
  ],
  "body_names": [
    "base",
    "FL_hip",
    "FR_hip",
    "RL_hip",
    "RR_hip",
    "Head_upper",
    "FL_thigh",
    "FR_thigh",
    "RL_thigh",
    "RR_thigh",
    "Head_lower",
    "FL_calf",
    "FR_calf",
    "RL_calf",
    "RR_calf",
    "FL_foot",
    "FR_foot",
    "RL_foot",
    "RR_foot"
  ],
  "consts": {
    "default_q": [
      0.10000000149011612,
      0.8999999761581421,
      -1.7999999523162842,
      -0.10000000149011612,
      0.8999999761581421,
      -1.7999999523162842,
      0.10000000149011612,
      0.8999999761581421,
      -1.7999999523162842,
      -0.10000000149011612,
      0.8999999761581421,
      -1.7999999523162842
    ],
    "action_scale": [
      0.3349008560180664,
      0.3349008560180664,
      0.2853173315525055,
      0.3349008560180664,
      0.3349008560180664,
      0.2853173315525055,
      0.3349008560180664,
      0.3349008560180664,
      0.2853173315525055,
      0.3349008560180664,
      0.3349008560180664,
      0.2853173315525055
    ],
    "kp": [
      17.691802978515625,
      17.691802978515625,
      39.806556701660156,
      17.691802978515625,
      17.691802978515625,
      39.806556701660156,
      17.691802978515625,
      17.691802978515625,
      39.806556701660156,
      17.691802978515625,
      17.691802978515625,
      39.806556701660156
    ],
    "kd": [
      1.1262950897216797,
      1.1262950897216797,
      2.5341639518737793,
      1.1262950897216797,
      1.1262950897216797,
      2.5341639518737793,
      1.1262950897216797,
      1.1262950897216797,
      2.5341639518737793,
      1.1262950897216797,
      1.1262950897216797,
      2.5341639518737793
    ],
    "effort": [
      23.700000762939453,
      23.700000762939453,
      45.43000030517578,
      23.700000762939453,
      23.700000762939453,
      45.43000030517578,
      23.700000762939453,
      23.700000762939453,
      45.43000030517578,
      23.700000762939453,
      23.700000762939453,
      45.43000030517578
    ],
    "soft_limits": [
      [
        -0.9424799680709839,
        0.9424799680709839
      ],
      [
        -1.3177249431610107,
        3.2376248836517334
      ],
      [
        -2.628453016281128,
        -0.9320070147514343
      ],
      [
        -0.9424799680709839,
        0.9424799680709839
      ],
      [
        -1.3177249431610107,
        3.2376248836517334
      ],
      [
        -2.628453016281128,
        -0.9320070147514343
      ],
      [
        -0.9424799680709839,
        0.9424799680709839
      ],
      [
        -0.2705249786376953,
        4.284825325012207
      ],
      [
        -2.628453016281128,
        -0.9320070147514343
      ],
      [
        -0.9424799680709839,
        0.9424799680709839
      ],
      [
        -0.2705249786376953,
        4.284825325012207
      ],
      [
        -2.628453016281128,
        -0.9320070147514343
      ]
    ]
  },
  "actual": {
    "body_mass": [
      [
        7.2769999504089355,
        0.7670000195503235,
        0.7670000195503235,
        0.7670000195503235,
        0.7670000195503235,
        0.0010000000474974513,
        1.2410000562667847,
        1.2410000562667847,
        1.2410000562667847,
        1.2410000562667847,
        0.0010000000474974513,
        0.15399999916553497,
        0.15399999916553497,
        0.15399999916553497,
        0.15399999916553497,
        0.03999999910593033,
        0.03999999910593033,
        0.03999999910593033,
        0.03999999910593033
      ],
      [
        7.777000427246094,
        0.7670000195503235,
        0.7670000195503235,
        0.7670000195503235,
        0.7670000195503235,
        0.0010000000474974513,
        1.2410000562667847,
        1.2410000562667847,
        1.2410000562667847,
        1.2410000562667847,
        0.0010000000474974513,
        0.15399999916553497,
        0.15399999916553497,
        0.15399999916553497,
        0.15399999916553497,
        0.03999999910593033,
        0.03999999910593033,
        0.03999999910593033,
        0.03999999910593033
      ],
      [
        8.277000427246094,
        0.7670000195503235,
        0.7670000195503235,
        0.7670000195503235,
        0.7670000195503235,
        0.0010000000474974513,
        1.2410000562667847,
        1.2410000562667847,
        1.2410000562667847,
        1.2410000562667847,
        0.0010000000474974513,
        0.15399999916553497,
        0.15399999916553497,
        0.15399999916553497,
        0.15399999916553497,
        0.03999999910593033,
        0.03999999910593033,
        0.03999999910593033,
        0.03999999910593033
      ],
      [
        9.277000427246094,
        0.7670000195503235,
        0.7670000195503235,
        0.7670000195503235,
        0.7670000195503235,
        0.0010000000474974513,
        1.2410000562667847,
        1.2410000562667847,
        1.2410000562667847,
        1.2410000562667847,
        0.0010000000474974513,
        0.15399999916553497,
        0.15399999916553497,
        0.15399999916553497,
        0.15399999916553497,
        0.03999999910593033,
        0.03999999910593033,
        0.03999999910593033,
        0.03999999910593033
      ]
    ],
    "body_inertia": [
      [
        [
          0.0257151760160923,
          0.0001216600212501362,
          0.0015232570003718138,
          0.0001216600212501362,
          0.10295388847589493,
          -3.1199975637719035e-05,
          0.0015232570003718138,
          -3.1199975637719035e-05,
          0.11264519393444061
        ],
        [
          0.0005399914807640016,
          -2.1220994312898256e-06,
          1.0653922117853654e-06,
          -2.1220994312898256e-06,
          0.000998136936686933,
          -1.4027380075276596e-06,
          1.0653922117853654e-06,
          -1.4027380075276596e-06,
          0.0006582847563549876
        ],
        [
          0.0005399914807640016,
          2.12209920391615e-06,
          1.0653924391590408e-06,
          2.12209920391615e-06,
          0.000998136936686933,
          1.4027532415639143e-06,
          1.0653924391590408e-06,
          1.4027532415639143e-06,
          0.0006582847563549876
        ],
        [
          0.0005399914807640016,
          2.1220994312898256e-06,
          -1.0653922117853654e-06,
          2.1220994312898256e-06,
          0.000998136936686933,
          -1.4027380075276596e-06,
          -1.0653922117853654e-06,
          -1.4027380075276596e-06,
          0.0006582847563549876
        ],
        [
          0.0005399914807640016,
          -2.12209920391615e-06,
          -1.0653924391590408e-06,
          -2.12209920391615e-06,
          0.000998136936686933,
          1.4027532415639143e-06,
          -1.0653924391590408e-06,
          1.4027532415639143e-06,
          0.0006582847563549876
        ],
        [
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06
        ],
        [
          0.005996335297822952,
          9.03045292943716e-05,
          -0.00029910437297075987,
          9.030464570969343e-05,
          0.006001342087984085,
          0.0008351504220627248,
          -0.00029910437297075987,
          0.0008351504220627248,
          0.001099147368222475
        ],
        [
          0.005996335297822952,
          -9.030639193952084e-05,
          -0.00029910379089415073,
          -9.030650835484266e-05,
          0.0060013411566615105,
          -0.0008351521100848913,
          -0.00029910379089415073,
          -0.0008351521100848913,
          0.0010991478338837624
        ],
        [
          0.005996335297822952,
          9.03045292943716e-05,
          -0.00029910437297075987,
          9.030464570969343e-05,
          0.006001342087984085,
          0.0008351504220627248,
          -0.00029910437297075987,
          0.0008351504220627248,
          0.001099147368222475
        ],
        [
          0.005996335297822952,
          -9.030639193952084e-05,
          -0.00029910379089415073,
          -9.030650835484266e-05,
          0.0060013411566615105,
          -0.0008351521100848913,
          -0.00029910379089415073,
          -0.0008351521100848913,
          0.0010991478338837624
        ],
        [
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06
        ],
        [
          0.0010800004238262773,
          3.3975447877310216e-07,
          1.7200001821038313e-05,
          3.397526597836986e-07,
          0.0011000004597008228,
          8.279685062007047e-06,
          1.7200001821038313e-05,
          8.279684152512345e-06,
          3.290001041023061e-05
        ],
        [
          0.0010800004238262773,
          -3.4026925277430564e-07,
          1.7200014553964138e-05,
          -3.402674337849021e-07,
          0.0011000004597008228,
          -8.280216206912883e-06,
          1.7200014553964138e-05,
          -8.280216206912883e-06,
          3.290001768618822e-05
        ],
        [
          0.0010800004238262773,
          3.3975447877310216e-07,
          1.7200001821038313e-05,
          3.397526597836986e-07,
          0.0011000004597008228,
          8.279685062007047e-06,
          1.7200001821038313e-05,
          8.279684152512345e-06,
          3.290001041023061e-05
        ],
        [
          0.0010800004238262773,
          -3.4026925277430564e-07,
          1.7200014553964138e-05,
          -3.402674337849021e-07,
          0.0011000004597008228,
          -8.280216206912883e-06,
          1.7200014553964138e-05,
          -8.280216206912883e-06,
          3.290001768618822e-05
        ],
        [
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06
        ],
        [
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06
        ],
        [
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06
        ],
        [
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06
        ]
      ],
      [
        [
          0.0257151760160923,
          0.0001216600212501362,
          0.0015232570003718138,
          0.0001216600212501362,
          0.10295388847589493,
          -3.1199975637719035e-05,
          0.0015232570003718138,
          -3.1199975637719035e-05,
          0.11264519393444061
        ],
        [
          0.0005399914807640016,
          -2.1220994312898256e-06,
          1.0653922117853654e-06,
          -2.1220994312898256e-06,
          0.000998136936686933,
          -1.4027380075276596e-06,
          1.0653922117853654e-06,
          -1.4027380075276596e-06,
          0.0006582847563549876
        ],
        [
          0.0005399914807640016,
          2.12209920391615e-06,
          1.0653924391590408e-06,
          2.12209920391615e-06,
          0.000998136936686933,
          1.4027532415639143e-06,
          1.0653924391590408e-06,
          1.4027532415639143e-06,
          0.0006582847563549876
        ],
        [
          0.0005399914807640016,
          2.1220994312898256e-06,
          -1.0653922117853654e-06,
          2.1220994312898256e-06,
          0.000998136936686933,
          -1.4027380075276596e-06,
          -1.0653922117853654e-06,
          -1.4027380075276596e-06,
          0.0006582847563549876
        ],
        [
          0.0005399914807640016,
          -2.12209920391615e-06,
          -1.0653924391590408e-06,
          -2.12209920391615e-06,
          0.000998136936686933,
          1.4027532415639143e-06,
          -1.0653924391590408e-06,
          1.4027532415639143e-06,
          0.0006582847563549876
        ],
        [
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06
        ],
        [
          0.005996335297822952,
          9.03045292943716e-05,
          -0.00029910437297075987,
          9.030464570969343e-05,
          0.006001342087984085,
          0.0008351504220627248,
          -0.00029910437297075987,
          0.0008351504220627248,
          0.001099147368222475
        ],
        [
          0.005996335297822952,
          -9.030639193952084e-05,
          -0.00029910379089415073,
          -9.030650835484266e-05,
          0.0060013411566615105,
          -0.0008351521100848913,
          -0.00029910379089415073,
          -0.0008351521100848913,
          0.0010991478338837624
        ],
        [
          0.005996335297822952,
          9.03045292943716e-05,
          -0.00029910437297075987,
          9.030464570969343e-05,
          0.006001342087984085,
          0.0008351504220627248,
          -0.00029910437297075987,
          0.0008351504220627248,
          0.001099147368222475
        ],
        [
          0.005996335297822952,
          -9.030639193952084e-05,
          -0.00029910379089415073,
          -9.030650835484266e-05,
          0.0060013411566615105,
          -0.0008351521100848913,
          -0.00029910379089415073,
          -0.0008351521100848913,
          0.0010991478338837624
        ],
        [
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06
        ],
        [
          0.0010800004238262773,
          3.3975447877310216e-07,
          1.7200001821038313e-05,
          3.397526597836986e-07,
          0.0011000004597008228,
          8.279685062007047e-06,
          1.7200001821038313e-05,
          8.279684152512345e-06,
          3.290001041023061e-05
        ],
        [
          0.0010800004238262773,
          -3.4026925277430564e-07,
          1.7200014553964138e-05,
          -3.402674337849021e-07,
          0.0011000004597008228,
          -8.280216206912883e-06,
          1.7200014553964138e-05,
          -8.280216206912883e-06,
          3.290001768618822e-05
        ],
        [
          0.0010800004238262773,
          3.3975447877310216e-07,
          1.7200001821038313e-05,
          3.397526597836986e-07,
          0.0011000004597008228,
          8.279685062007047e-06,
          1.7200001821038313e-05,
          8.279684152512345e-06,
          3.290001041023061e-05
        ],
        [
          0.0010800004238262773,
          -3.4026925277430564e-07,
          1.7200014553964138e-05,
          -3.402674337849021e-07,
          0.0011000004597008228,
          -8.280216206912883e-06,
          1.7200014553964138e-05,
          -8.280216206912883e-06,
          3.290001768618822e-05
        ],
        [
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06
        ],
        [
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06
        ],
        [
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06
        ],
        [
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06
        ]
      ],
      [
        [
          0.0257151760160923,
          0.0001216600212501362,
          0.0015232570003718138,
          0.0001216600212501362,
          0.10295388847589493,
          -3.1199975637719035e-05,
          0.0015232570003718138,
          -3.1199975637719035e-05,
          0.11264519393444061
        ],
        [
          0.0005399914807640016,
          -2.1220994312898256e-06,
          1.0653922117853654e-06,
          -2.1220994312898256e-06,
          0.000998136936686933,
          -1.4027380075276596e-06,
          1.0653922117853654e-06,
          -1.4027380075276596e-06,
          0.0006582847563549876
        ],
        [
          0.0005399914807640016,
          2.12209920391615e-06,
          1.0653924391590408e-06,
          2.12209920391615e-06,
          0.000998136936686933,
          1.4027532415639143e-06,
          1.0653924391590408e-06,
          1.4027532415639143e-06,
          0.0006582847563549876
        ],
        [
          0.0005399914807640016,
          2.1220994312898256e-06,
          -1.0653922117853654e-06,
          2.1220994312898256e-06,
          0.000998136936686933,
          -1.4027380075276596e-06,
          -1.0653922117853654e-06,
          -1.4027380075276596e-06,
          0.0006582847563549876
        ],
        [
          0.0005399914807640016,
          -2.12209920391615e-06,
          -1.0653924391590408e-06,
          -2.12209920391615e-06,
          0.000998136936686933,
          1.4027532415639143e-06,
          -1.0653924391590408e-06,
          1.4027532415639143e-06,
          0.0006582847563549876
        ],
        [
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06
        ],
        [
          0.005996335297822952,
          9.03045292943716e-05,
          -0.00029910437297075987,
          9.030464570969343e-05,
          0.006001342087984085,
          0.0008351504220627248,
          -0.00029910437297075987,
          0.0008351504220627248,
          0.001099147368222475
        ],
        [
          0.005996335297822952,
          -9.030639193952084e-05,
          -0.00029910379089415073,
          -9.030650835484266e-05,
          0.0060013411566615105,
          -0.0008351521100848913,
          -0.00029910379089415073,
          -0.0008351521100848913,
          0.0010991478338837624
        ],
        [
          0.005996335297822952,
          9.03045292943716e-05,
          -0.00029910437297075987,
          9.030464570969343e-05,
          0.006001342087984085,
          0.0008351504220627248,
          -0.00029910437297075987,
          0.0008351504220627248,
          0.001099147368222475
        ],
        [
          0.005996335297822952,
          -9.030639193952084e-05,
          -0.00029910379089415073,
          -9.030650835484266e-05,
          0.0060013411566615105,
          -0.0008351521100848913,
          -0.00029910379089415073,
          -0.0008351521100848913,
          0.0010991478338837624
        ],
        [
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06
        ],
        [
          0.0010800004238262773,
          3.3975447877310216e-07,
          1.7200001821038313e-05,
          3.397526597836986e-07,
          0.0011000004597008228,
          8.279685062007047e-06,
          1.7200001821038313e-05,
          8.279684152512345e-06,
          3.290001041023061e-05
        ],
        [
          0.0010800004238262773,
          -3.4026925277430564e-07,
          1.7200014553964138e-05,
          -3.402674337849021e-07,
          0.0011000004597008228,
          -8.280216206912883e-06,
          1.7200014553964138e-05,
          -8.280216206912883e-06,
          3.290001768618822e-05
        ],
        [
          0.0010800004238262773,
          3.3975447877310216e-07,
          1.7200001821038313e-05,
          3.397526597836986e-07,
          0.0011000004597008228,
          8.279685062007047e-06,
          1.7200001821038313e-05,
          8.279684152512345e-06,
          3.290001041023061e-05
        ],
        [
          0.0010800004238262773,
          -3.4026925277430564e-07,
          1.7200014553964138e-05,
          -3.402674337849021e-07,
          0.0011000004597008228,
          -8.280216206912883e-06,
          1.7200014553964138e-05,
          -8.280216206912883e-06,
          3.290001768618822e-05
        ],
        [
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06
        ],
        [
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06
        ],
        [
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06
        ],
        [
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06
        ]
      ],
      [
        [
          0.0257151760160923,
          0.0001216600212501362,
          0.0015232570003718138,
          0.0001216600212501362,
          0.10295388847589493,
          -3.1199975637719035e-05,
          0.0015232570003718138,
          -3.1199975637719035e-05,
          0.11264519393444061
        ],
        [
          0.0005399914807640016,
          -2.1220994312898256e-06,
          1.0653922117853654e-06,
          -2.1220994312898256e-06,
          0.000998136936686933,
          -1.4027380075276596e-06,
          1.0653922117853654e-06,
          -1.4027380075276596e-06,
          0.0006582847563549876
        ],
        [
          0.0005399914807640016,
          2.12209920391615e-06,
          1.0653924391590408e-06,
          2.12209920391615e-06,
          0.000998136936686933,
          1.4027532415639143e-06,
          1.0653924391590408e-06,
          1.4027532415639143e-06,
          0.0006582847563549876
        ],
        [
          0.0005399914807640016,
          2.1220994312898256e-06,
          -1.0653922117853654e-06,
          2.1220994312898256e-06,
          0.000998136936686933,
          -1.4027380075276596e-06,
          -1.0653922117853654e-06,
          -1.4027380075276596e-06,
          0.0006582847563549876
        ],
        [
          0.0005399914807640016,
          -2.12209920391615e-06,
          -1.0653924391590408e-06,
          -2.12209920391615e-06,
          0.000998136936686933,
          1.4027532415639143e-06,
          -1.0653924391590408e-06,
          1.4027532415639143e-06,
          0.0006582847563549876
        ],
        [
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06
        ],
        [
          0.005996335297822952,
          9.03045292943716e-05,
          -0.00029910437297075987,
          9.030464570969343e-05,
          0.006001342087984085,
          0.0008351504220627248,
          -0.00029910437297075987,
          0.0008351504220627248,
          0.001099147368222475
        ],
        [
          0.005996335297822952,
          -9.030639193952084e-05,
          -0.00029910379089415073,
          -9.030650835484266e-05,
          0.0060013411566615105,
          -0.0008351521100848913,
          -0.00029910379089415073,
          -0.0008351521100848913,
          0.0010991478338837624
        ],
        [
          0.005996335297822952,
          9.03045292943716e-05,
          -0.00029910437297075987,
          9.030464570969343e-05,
          0.006001342087984085,
          0.0008351504220627248,
          -0.00029910437297075987,
          0.0008351504220627248,
          0.001099147368222475
        ],
        [
          0.005996335297822952,
          -9.030639193952084e-05,
          -0.00029910379089415073,
          -9.030650835484266e-05,
          0.0060013411566615105,
          -0.0008351521100848913,
          -0.00029910379089415073,
          -0.0008351521100848913,
          0.0010991478338837624
        ],
        [
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06
        ],
        [
          0.0010800004238262773,
          3.3975447877310216e-07,
          1.7200001821038313e-05,
          3.397526597836986e-07,
          0.0011000004597008228,
          8.279685062007047e-06,
          1.7200001821038313e-05,
          8.279684152512345e-06,
          3.290001041023061e-05
        ],
        [
          0.0010800004238262773,
          -3.4026925277430564e-07,
          1.7200014553964138e-05,
          -3.402674337849021e-07,
          0.0011000004597008228,
          -8.280216206912883e-06,
          1.7200014553964138e-05,
          -8.280216206912883e-06,
          3.290001768618822e-05
        ],
        [
          0.0010800004238262773,
          3.3975447877310216e-07,
          1.7200001821038313e-05,
          3.397526597836986e-07,
          0.0011000004597008228,
          8.279685062007047e-06,
          1.7200001821038313e-05,
          8.279684152512345e-06,
          3.290001041023061e-05
        ],
        [
          0.0010800004238262773,
          -3.4026925277430564e-07,
          1.7200014553964138e-05,
          -3.402674337849021e-07,
          0.0011000004597008228,
          -8.280216206912883e-06,
          1.7200014553964138e-05,
          -8.280216206912883e-06,
          3.290001768618822e-05
        ],
        [
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06
        ],
        [
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06
        ],
        [
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06
        ],
        [
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06,
          0.0,
          0.0,
          0.0,
          9.60000033956021e-06
        ]
      ]
    ],
    "body_com_pose_b": [
      [
        [
          0.020079175010323524,
          0.0,
          -0.005103488452732563,
          0.0014884165721014142,
          0.008756772615015507,
          -0.0008037648512981832,
          0.9999602437019348
        ],
        [
          -0.00477340305224061,
          0.0016974837053567171,
          -9.281616803491488e-05,
          0.7056246399879456,
          0.004795296583324671,
          -0.001526498468592763,
          0.7085679769515991
        ],
        [
          -0.00477340305224061,
          -0.0016974837053567171,
          -9.281616803491488e-05,
          -0.7085679769515991,
          -0.001526498468592763,
          0.004795296583324671,
          -0.7056246399879456
        ],
        [
          0.00477340305224061,
          0.0016974837053567171,
          -9.281616803491488e-05,
          0.7056246399879456,
          -0.004795296583324671,
          0.001526498468592763,
          0.7085679769515991
        ],
        [
          0.00477340305224061,
          -0.0016974837053567171,
          -9.281616803491488e-05,
          -0.7085679769515991,
          0.001526498468592763,
          -0.004795296583324671,
          -0.7056246399879456
        ],
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0
        ],
        [
          -0.003471780801191926,
          -0.023020749911665916,
          -0.030354874208569527,
          -0.5454739332199097,
          -0.3638213574886322,
          -0.5805513262748718,
          0.4827551543712616
        ],
        [
          -0.003471780801191926,
          0.023020749911665916,
          -0.030354874208569527,
          -0.4827551543712616,
          -0.5805513262748718,
          -0.3638213574886322,
          0.5454739332199097
        ],
        [
          -0.003471780801191926,
          -0.023020749911665916,
          -0.030354874208569527,
          -0.5454739332199097,
          -0.3638213574886322,
          -0.5805513262748718,
          0.4827551543712616
        ],
        [
          -0.003471780801191926,
          0.023020749911665916,
          -0.030354874208569527,
          -0.4827551543712616,
          -0.5805513262748718,
          -0.3638213574886322,
          0.5454739332199097
        ],
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0
        ],
        [
          0.00547999981790781,
          -0.0009749999735504389,
          -0.11500000208616257,
          -0.4998696744441986,
          -0.49604398012161255,
          -0.5120041966438293,
          0.49185627698898315
        ],
        [
          0.00547999981790781,
          0.0009749999735504389,
          -0.11500000208616257,
          -0.49185627698898315,
          -0.5120041966438293,
          -0.49604398012161255,
          0.4998696744441986
        ],
        [
          0.00547999981790781,
          -0.0009749999735504389,
          -0.11500000208616257,
          -0.4998696744441986,
          -0.49604398012161255,
          -0.5120041966438293,
          0.49185627698898315
        ],
        [
          0.00547999981790781,
          0.0009749999735504389,
          -0.11500000208616257,
          -0.49185627698898315,
          -0.5120041966438293,
          -0.49604398012161255,
          0.4998696744441986
        ],
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0
        ],
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0
        ],
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0
        ],
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0
        ]
      ],
      [
        [
          0.020079175010323524,
          0.0,
          -0.005103488452732563,
          0.0014884165721014142,
          0.008756772615015507,
          -0.0008037648512981832,
          0.9999602437019348
        ],
        [
          -0.00477340305224061,
          0.0016974837053567171,
          -9.281616803491488e-05,
          0.7056246399879456,
          0.004795296583324671,
          -0.001526498468592763,
          0.7085679769515991
        ],
        [
          -0.00477340305224061,
          -0.0016974837053567171,
          -9.281616803491488e-05,
          -0.7085679769515991,
          -0.001526498468592763,
          0.004795296583324671,
          -0.7056246399879456
        ],
        [
          0.00477340305224061,
          0.0016974837053567171,
          -9.281616803491488e-05,
          0.7056246399879456,
          -0.004795296583324671,
          0.001526498468592763,
          0.7085679769515991
        ],
        [
          0.00477340305224061,
          -0.0016974837053567171,
          -9.281616803491488e-05,
          -0.7085679769515991,
          0.001526498468592763,
          -0.004795296583324671,
          -0.7056246399879456
        ],
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0
        ],
        [
          -0.003471780801191926,
          -0.023020749911665916,
          -0.030354874208569527,
          -0.5454739332199097,
          -0.3638213574886322,
          -0.5805513262748718,
          0.4827551543712616
        ],
        [
          -0.003471780801191926,
          0.023020749911665916,
          -0.030354874208569527,
          -0.4827551543712616,
          -0.5805513262748718,
          -0.3638213574886322,
          0.5454739332199097
        ],
        [
          -0.003471780801191926,
          -0.023020749911665916,
          -0.030354874208569527,
          -0.5454739332199097,
          -0.3638213574886322,
          -0.5805513262748718,
          0.4827551543712616
        ],
        [
          -0.003471780801191926,
          0.023020749911665916,
          -0.030354874208569527,
          -0.4827551543712616,
          -0.5805513262748718,
          -0.3638213574886322,
          0.5454739332199097
        ],
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0
        ],
        [
          0.00547999981790781,
          -0.0009749999735504389,
          -0.11500000208616257,
          -0.4998696744441986,
          -0.49604398012161255,
          -0.5120041966438293,
          0.49185627698898315
        ],
        [
          0.00547999981790781,
          0.0009749999735504389,
          -0.11500000208616257,
          -0.49185627698898315,
          -0.5120041966438293,
          -0.49604398012161255,
          0.4998696744441986
        ],
        [
          0.00547999981790781,
          -0.0009749999735504389,
          -0.11500000208616257,
          -0.4998696744441986,
          -0.49604398012161255,
          -0.5120041966438293,
          0.49185627698898315
        ],
        [
          0.00547999981790781,
          0.0009749999735504389,
          -0.11500000208616257,
          -0.49185627698898315,
          -0.5120041966438293,
          -0.49604398012161255,
          0.4998696744441986
        ],
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0
        ],
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0
        ],
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0
        ],
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0
        ]
      ],
      [
        [
          0.020079175010323524,
          0.0,
          -0.005103488452732563,
          0.0014884165721014142,
          0.008756772615015507,
          -0.0008037648512981832,
          0.9999602437019348
        ],
        [
          -0.00477340305224061,
          0.0016974837053567171,
          -9.281616803491488e-05,
          0.7056246399879456,
          0.004795296583324671,
          -0.001526498468592763,
          0.7085679769515991
        ],
        [
          -0.00477340305224061,
          -0.0016974837053567171,
          -9.281616803491488e-05,
          -0.7085679769515991,
          -0.001526498468592763,
          0.004795296583324671,
          -0.7056246399879456
        ],
        [
          0.00477340305224061,
          0.0016974837053567171,
          -9.281616803491488e-05,
          0.7056246399879456,
          -0.004795296583324671,
          0.001526498468592763,
          0.7085679769515991
        ],
        [
          0.00477340305224061,
          -0.0016974837053567171,
          -9.281616803491488e-05,
          -0.7085679769515991,
          0.001526498468592763,
          -0.004795296583324671,
          -0.7056246399879456
        ],
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0
        ],
        [
          -0.003471780801191926,
          -0.023020749911665916,
          -0.030354874208569527,
          -0.5454739332199097,
          -0.3638213574886322,
          -0.5805513262748718,
          0.4827551543712616
        ],
        [
          -0.003471780801191926,
          0.023020749911665916,
          -0.030354874208569527,
          -0.4827551543712616,
          -0.5805513262748718,
          -0.3638213574886322,
          0.5454739332199097
        ],
        [
          -0.003471780801191926,
          -0.023020749911665916,
          -0.030354874208569527,
          -0.5454739332199097,
          -0.3638213574886322,
          -0.5805513262748718,
          0.4827551543712616
        ],
        [
          -0.003471780801191926,
          0.023020749911665916,
          -0.030354874208569527,
          -0.4827551543712616,
          -0.5805513262748718,
          -0.3638213574886322,
          0.5454739332199097
        ],
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0
        ],
        [
          0.00547999981790781,
          -0.0009749999735504389,
          -0.11500000208616257,
          -0.4998696744441986,
          -0.49604398012161255,
          -0.5120041966438293,
          0.49185627698898315
        ],
        [
          0.00547999981790781,
          0.0009749999735504389,
          -0.11500000208616257,
          -0.49185627698898315,
          -0.5120041966438293,
          -0.49604398012161255,
          0.4998696744441986
        ],
        [
          0.00547999981790781,
          -0.0009749999735504389,
          -0.11500000208616257,
          -0.4998696744441986,
          -0.49604398012161255,
          -0.5120041966438293,
          0.49185627698898315
        ],
        [
          0.00547999981790781,
          0.0009749999735504389,
          -0.11500000208616257,
          -0.49185627698898315,
          -0.5120041966438293,
          -0.49604398012161255,
          0.4998696744441986
        ],
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0
        ],
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0
        ],
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0
        ],
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0
        ]
      ],
      [
        [
          0.020079175010323524,
          0.0,
          -0.005103488452732563,
          0.0014884165721014142,
          0.008756772615015507,
          -0.0008037648512981832,
          0.9999602437019348
        ],
        [
          -0.00477340305224061,
          0.0016974837053567171,
          -9.281616803491488e-05,
          0.7056246399879456,
          0.004795296583324671,
          -0.001526498468592763,
          0.7085679769515991
        ],
        [
          -0.00477340305224061,
          -0.0016974837053567171,
          -9.281616803491488e-05,
          -0.7085679769515991,
          -0.001526498468592763,
          0.004795296583324671,
          -0.7056246399879456
        ],
        [
          0.00477340305224061,
          0.0016974837053567171,
          -9.281616803491488e-05,
          0.7056246399879456,
          -0.004795296583324671,
          0.001526498468592763,
          0.7085679769515991
        ],
        [
          0.00477340305224061,
          -0.0016974837053567171,
          -9.281616803491488e-05,
          -0.7085679769515991,
          0.001526498468592763,
          -0.004795296583324671,
          -0.7056246399879456
        ],
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0
        ],
        [
          -0.003471780801191926,
          -0.023020749911665916,
          -0.030354874208569527,
          -0.5454739332199097,
          -0.3638213574886322,
          -0.5805513262748718,
          0.4827551543712616
        ],
        [
          -0.003471780801191926,
          0.023020749911665916,
          -0.030354874208569527,
          -0.4827551543712616,
          -0.5805513262748718,
          -0.3638213574886322,
          0.5454739332199097
        ],
        [
          -0.003471780801191926,
          -0.023020749911665916,
          -0.030354874208569527,
          -0.5454739332199097,
          -0.3638213574886322,
          -0.5805513262748718,
          0.4827551543712616
        ],
        [
          -0.003471780801191926,
          0.023020749911665916,
          -0.030354874208569527,
          -0.4827551543712616,
          -0.5805513262748718,
          -0.3638213574886322,
          0.5454739332199097
        ],
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0
        ],
        [
          0.00547999981790781,
          -0.0009749999735504389,
          -0.11500000208616257,
          -0.4998696744441986,
          -0.49604398012161255,
          -0.5120041966438293,
          0.49185627698898315
        ],
        [
          0.00547999981790781,
          0.0009749999735504389,
          -0.11500000208616257,
          -0.49185627698898315,
          -0.5120041966438293,
          -0.49604398012161255,
          0.4998696744441986
        ],
        [
          0.00547999981790781,
          -0.0009749999735504389,
          -0.11500000208616257,
          -0.4998696744441986,
          -0.49604398012161255,
          -0.5120041966438293,
          0.49185627698898315
        ],
        [
          0.00547999981790781,
          0.0009749999735504389,
          -0.11500000208616257,
          -0.49185627698898315,
          -0.5120041966438293,
          -0.49604398012161255,
          0.4998696744441986
        ],
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0
        ],
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0
        ],
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0
        ],
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0
        ]
      ]
    ],
    "joint_pos_limits": [
      [
        [
          -1.0471999645233154,
          1.0471999645233154
        ],
        [
          -1.5708000659942627,
          3.4907000064849854
        ],
        [
          -2.7227001190185547,
          -0.8377599716186523
        ],
        [
          -1.0471999645233154,
          1.0471999645233154
        ],
        [
          -1.5708000659942627,
          3.4907000064849854
        ],
        [
          -2.7227001190185547,
          -0.8377599716186523
        ],
        [
          -1.0471999645233154,
          1.0471999645233154
        ],
        [
          -0.5235999822616577,
          4.537900447845459
        ],
        [
          -2.7227001190185547,
          -0.8377599716186523
        ],
        [
          -1.0471999645233154,
          1.0471999645233154
        ],
        [
          -0.5235999822616577,
          4.537900447845459
        ],
        [
          -2.7227001190185547,
          -0.8377599716186523
        ]
      ],
      [
        [
          -1.0471999645233154,
          1.0471999645233154
        ],
        [
          -1.5708000659942627,
          3.4907000064849854
        ],
        [
          -2.7227001190185547,
          -0.8377599716186523
        ],
        [
          -1.0471999645233154,
          1.0471999645233154
        ],
        [
          -1.5708000659942627,
          3.4907000064849854
        ],
        [
          -2.7227001190185547,
          -0.8377599716186523
        ],
        [
          -1.0471999645233154,
          1.0471999645233154
        ],
        [
          -0.5235999822616577,
          4.537900447845459
        ],
        [
          -2.7227001190185547,
          -0.8377599716186523
        ],
        [
          -1.0471999645233154,
          1.0471999645233154
        ],
        [
          -0.5235999822616577,
          4.537900447845459
        ],
        [
          -2.7227001190185547,
          -0.8377599716186523
        ]
      ],
      [
        [
          -1.0471999645233154,
          1.0471999645233154
        ],
        [
          -1.5708000659942627,
          3.4907000064849854
        ],
        [
          -2.7227001190185547,
          -0.8377599716186523
        ],
        [
          -1.0471999645233154,
          1.0471999645233154
        ],
        [
          -1.5708000659942627,
          3.4907000064849854
        ],
        [
          -2.7227001190185547,
          -0.8377599716186523
        ],
        [
          -1.0471999645233154,
          1.0471999645233154
        ],
        [
          -0.5235999822616577,
          4.537900447845459
        ],
        [
          -2.7227001190185547,
          -0.8377599716186523
        ],
        [
          -1.0471999645233154,
          1.0471999645233154
        ],
        [
          -0.5235999822616577,
          4.537900447845459
        ],
        [
          -2.7227001190185547,
          -0.8377599716186523
        ]
      ],
      [
        [
          -1.0471999645233154,
          1.0471999645233154
        ],
        [
          -1.5708000659942627,
          3.4907000064849854
        ],
        [
          -2.7227001190185547,
          -0.8377599716186523
        ],
        [
          -1.0471999645233154,
          1.0471999645233154
        ],
        [
          -1.5708000659942627,
          3.4907000064849854
        ],
        [
          -2.7227001190185547,
          -0.8377599716186523
        ],
        [
          -1.0471999645233154,
          1.0471999645233154
        ],
        [
          -0.5235999822616577,
          4.537900447845459
        ],
        [
          -2.7227001190185547,
          -0.8377599716186523
        ],
        [
          -1.0471999645233154,
          1.0471999645233154
        ],
        [
          -0.5235999822616577,
          4.537900447845459
        ],
        [
          -2.7227001190185547,
          -0.8377599716186523
        ]
      ]
    ],
    "joint_vel_limits": [
      [
        30.100000381469727,
        30.100000381469727,
        20.06999969482422,
        30.100000381469727,
        30.100000381469727,
        20.06999969482422,
        30.100000381469727,
        30.100000381469727,
        20.06999969482422,
        30.100000381469727,
        30.100000381469727,
        20.06999969482422
      ],
      [
        30.100000381469727,
        30.100000381469727,
        20.06999969482422,
        30.100000381469727,
        30.100000381469727,
        20.06999969482422,
        30.100000381469727,
        30.100000381469727,
        20.06999969482422,
        30.100000381469727,
        30.100000381469727,
        20.06999969482422
      ],
      [
        30.100000381469727,
        30.100000381469727,
        20.06999969482422,
        30.100000381469727,
        30.100000381469727,
        20.06999969482422,
        30.100000381469727,
        30.100000381469727,
        20.06999969482422,
        30.100000381469727,
        30.100000381469727,
        20.06999969482422
      ],
      [
        30.100000381469727,
        30.100000381469727,
        20.06999969482422,
        30.100000381469727,
        30.100000381469727,
        20.06999969482422,
        30.100000381469727,
        30.100000381469727,
        20.06999969482422,
        30.100000381469727,
        30.100000381469727,
        20.06999969482422
      ]
    ],
    "joint_effort_limits": [
      [
        23.700000762939453,
        23.700000762939453,
        45.43000030517578,
        23.700000762939453,
        23.700000762939453,
        45.43000030517578,
        23.700000762939453,
        23.700000762939453,
        45.43000030517578,
        23.700000762939453,
        23.700000762939453,
        45.43000030517578
      ],
      [
        23.700000762939453,
        23.700000762939453,
        45.43000030517578,
        23.700000762939453,
        23.700000762939453,
        45.43000030517578,
        23.700000762939453,
        23.700000762939453,
        45.43000030517578,
        23.700000762939453,
        23.700000762939453,
        45.43000030517578
      ],
      [
        23.700000762939453,
        23.700000762939453,
        45.43000030517578,
        23.700000762939453,
        23.700000762939453,
        45.43000030517578,
        23.700000762939453,
        23.700000762939453,
        45.43000030517578,
        23.700000762939453,
        23.700000762939453,
        45.43000030517578
      ],
      [
        23.700000762939453,
        23.700000762939453,
        45.43000030517578,
        23.700000762939453,
        23.700000762939453,
        45.43000030517578,
        23.700000762939453,
        23.700000762939453,
        45.43000030517578,
        23.700000762939453,
        23.700000762939453,
        45.43000030517578
      ]
    ],
    "joint_armature": [
      [
        0.004481385927647352,
        0.004481385927647352,
        0.010083118453621864,
        0.004481385927647352,
        0.004481385927647352,
        0.010083118453621864,
        0.004481385927647352,
        0.004481385927647352,
        0.010083118453621864,
        0.004481385927647352,
        0.004481385927647352,
        0.010083118453621864
      ],
      [
        0.004481385927647352,
        0.004481385927647352,
        0.010083118453621864,
        0.004481385927647352,
        0.004481385927647352,
        0.010083118453621864,
        0.004481385927647352,
        0.004481385927647352,
        0.010083118453621864,
        0.004481385927647352,
        0.004481385927647352,
        0.010083118453621864
      ],
      [
        0.004481385927647352,
        0.004481385927647352,
        0.010083118453621864,
        0.004481385927647352,
        0.004481385927647352,
        0.010083118453621864,
        0.004481385927647352,
        0.004481385927647352,
        0.010083118453621864,
        0.004481385927647352,
        0.004481385927647352,
        0.010083118453621864
      ],
      [
        0.004481385927647352,
        0.004481385927647352,
        0.010083118453621864,
        0.004481385927647352,
        0.004481385927647352,
        0.010083118453621864,
        0.004481385927647352,
        0.004481385927647352,
        0.010083118453621864,
        0.004481385927647352,
        0.004481385927647352,
        0.010083118453621864
      ]
    ]
  },
  "control_steps": 50,
  "falls": 0,
  "policy_updates": 0,
  "capability_claim": false
}
```

## 实测或依赖记录：assets/manifest.json

```json
[
  {
    "path": "assets/as2/as2.usd",
    "bytes": 17205644,
    "sha256": "b0f287f97ff8b28cfb74884dbcbb9122e51b931541b2d6c2f0ec1138e73b3b91"
  },
  {
    "path": "assets/as2/as2_asset_manifest.json",
    "bytes": 5029,
    "sha256": "5b138093a4e9cd2ce06d48203f87468223acfb1ad97e863053b3c8b0e13e4613"
  },
  {
    "path": "assets/as2/as2_constants.py",
    "bytes": 3024,
    "sha256": "4e0099a1b3557e049fe2f5679e22bf3f64c061ae0d26c329fb029ef3e78c87e7"
  },
  {
    "path": "assets/as2/as2_env_cfgs.py",
    "bytes": 5739,
    "sha256": "4b979f27070295f3f46885f156afd0f5f2e4e27aae5852649a3a343e58b43b6f"
  },
  {
    "path": "assets/as2/asset_parameters.json",
    "bytes": 8810,
    "sha256": "d5e5001b9729e6153e78d1e32f23cb52874d4ea6a59da430fa952590f4b50dc6"
  },
  {
    "path": "assets/as2/official/as2_description/meshes/base_link.STL",
    "bytes": 6311184,
    "sha256": "4ace66bb12a972a67e1dbd823b55162e6d3931d23ccf56c707370c932f297f92"
  },
  {
    "path": "assets/as2/official/as2_description/meshes/FL_calf.STL",
    "bytes": 721684,
    "sha256": "d6bba46f70ec606a6a6dd94365e594f7f449b31a2fdc064f11c4dc6594d9c23e"
  },
  {
    "path": "assets/as2/official/as2_description/meshes/FL_foot.STL",
    "bytes": 3465284,
    "sha256": "ddf21582becb4df4ff7bd784c2ec0cb4b569c5c6f8ac96ae53e58f2b52f4950f"
  },
  {
    "path": "assets/as2/official/as2_description/meshes/FL_hip.STL",
    "bytes": 1383684,
    "sha256": "7566753929d4b4a34dfe7aa57a2601f7edbc9ecd3b3a6786a2045d8e8d7e3849"
  },
  {
    "path": "assets/as2/official/as2_description/meshes/FL_thigh.STL",
    "bytes": 634284,
    "sha256": "ad006344e8cb2ee51791f0b50cf7cb125ad1ae9c53b169a28c2ecddf3cfb3480"
  },
  {
    "path": "assets/as2/official/as2_description/meshes/FR_calf.STL",
    "bytes": 714184,
    "sha256": "5abf4370f3a837b27407f156cc2fc1b46b3c450b845b1231bfe32256f9376b45"
  },
  {
    "path": "assets/as2/official/as2_description/meshes/FR_foot.STL",
    "bytes": 3468484,
    "sha256": "1f3ae7b8df8ec8a8233761d32fe9ad4c801cff7403f07f495492a2e4c995481f"
  },
  {
    "path": "assets/as2/official/as2_description/meshes/FR_hip.STL",
    "bytes": 1383384,
    "sha256": "ab51c738698ae72c95d4aff893a8f42c59ff1c395e12ff88d5cf94413ff9f1da"
  },
  {
    "path": "assets/as2/official/as2_description/meshes/FR_thigh.STL",
    "bytes": 640384,
    "sha256": "71d25281826c92fb207e32ea10b53af0cfc5a2583c9766a13ab2bf5159463612"
  },
  {
    "path": "assets/as2/official/as2_description/meshes/RL_calf.STL",
    "bytes": 722784,
    "sha256": "f7927f861191efc5cbb0e350bb320d05b7af1599456d301b9eefcf8950bf8f14"
  },
  {
    "path": "assets/as2/official/as2_description/meshes/RL_foot.STL",
    "bytes": 3465284,
    "sha256": "5838f87163721303000799d7df5a5fce1eec4447d1db964466a9c009b59cdd86"
  },
  {
    "path": "assets/as2/official/as2_description/meshes/RL_hip.STL",
    "bytes": 1380684,
    "sha256": "ae47b3c00b93782747d8d2d90a2e7c0fba508e521743e603c2eb3f261199f7a6"
  },
  {
    "path": "assets/as2/official/as2_description/meshes/RL_thigh.STL",
    "bytes": 479784,
    "sha256": "33355396397dc35723c70384f08f50eaabe58d6f04490d77fa688cd6a70aa627"
  },
  {
    "path": "assets/as2/official/as2_description/meshes/RR_calf.STL",
    "bytes": 715784,
    "sha256": "b085a9b0d5086ff39993ededdf2bcb39642b130ea5351532026ff34507f97aa2"
  },
  {
    "path": "assets/as2/official/as2_description/meshes/RR_foot.STL",
    "bytes": 3468484,
    "sha256": "30795035d326e22c9469efabb60d0cd764c6ba3f76fcfd452e11c1f797882caf"
  },
  {
    "path": "assets/as2/official/as2_description/meshes/RR_hip.STL",
    "bytes": 1382984,
    "sha256": "3f99ca33686c23e0c5905606a4fc0291c16935dc3ab6362344b6266b5e06e672"
  },
  {
    "path": "assets/as2/official/as2_description/meshes/RR_thigh.STL",
    "bytes": 479284,
    "sha256": "1bcc5881b8b981cfca73f8718c1db6c2398b66fe7bb2fa6fb978cea7c8c7ec7d"
  },
  {
    "path": "assets/as2/official/as2_description/urdf/as2.urdf",
    "bytes": 21881,
    "sha256": "76b500f7b8fd5ca619e8b426274064abe9bdc284601d5e17f22a79083188553d"
  },
  {
    "path": "assets/as2/runtime_check.json",
    "bytes": 46127,
    "sha256": "6dc37ac9417a8fb16b1d260542e535824855759280a5e8ef49ca933d5bfaf62b"
  },
  {
    "path": "assets/go2/runtime_check.json",
    "bytes": 51541,
    "sha256": "d32929f641492fbfd623d273f4118b69630dba93a65a30bbd59e41d25b899986"
  },
  {
    "path": "assets/go2/urdf/go2.urdf",
    "bytes": 28582,
    "sha256": "9a3404330f271b042eb6ebb3070c07a7a59e30bf5d04150a525f57935156bd58"
  },
  {
    "path": "assets/go2/urdf/meshes/base.dae",
    "bytes": 10727399,
    "sha256": "e52bebbb5c7ff1f6aedee620c9b5b349f6e965360ecc63a9bb743eeff2d0be64"
  },
  {
    "path": "assets/go2/urdf/meshes/calf.dae",
    "bytes": 1125517,
    "sha256": "29872fa1a435b92bc2b395fdfa7f0aee26101db3c494681896bd4ef24ee7883c"
  },
  {
    "path": "assets/go2/urdf/meshes/calf_mirror.dae",
    "bytes": 1125457,
    "sha256": "79c66be7ed06760bd5001c067a1fb0ac517f639d8b1a5f89be471903b3f359e1"
  },
  {
    "path": "assets/go2/urdf/meshes/foot.dae",
    "bytes": 506988,
    "sha256": "ffbd95fd641866ce9bd277aa249b92dafc9148d70131fa2c52709a35ce5b710b"
  },
  {
    "path": "assets/go2/urdf/meshes/hip.dae",
    "bytes": 4666438,
    "sha256": "8eddfb2935507bc2fbcb610524baa27c847b2d631188234ff30dd43dad136643"
  },
  {
    "path": "assets/go2/urdf/meshes/thigh.dae",
    "bytes": 3884121,
    "sha256": "cf5293814a90b474fa19092b72a8be47ce8e520efa9d68ce11b137f79c7bdec5"
  },
  {
    "path": "assets/go2/urdf/meshes/thigh_mirror.dae",
    "bytes": 3896560,
    "sha256": "51754f3ace706c5aa3a84d635f61b621e59fe0d0daebde1ec5d6c342fbf203b3"
  },
  {
    "path": "assets/go2/usd/go2.usd",
    "bytes": 5734083,
    "sha256": "c71daf27f23dbfe4358052588722566b6163d721cd290a706de04e7b3cfe3928"
  }
]
```

## 实测或依赖记录：TRAINING_RUNTIME.json

```json
{
  "python": "3.12.16 (main, Oct 26 2026, 14:08:45) [Clang 23.1.3 ]",
  "packages": {
    "GitPython": "3.1.60",
    "aiohttp": "3.14.3",
    "h5py": "3.16.0",
    "nvidia-nccl-cu13": "2.28.9",
    "referencing": "0.37.0",
    "nbformat": "5.10.4",
    "ipywidgets": "8.1.8",
    "blinker": "1.9.0",
    "watchdog": "4.0.0",
    "ipython": "9.10.1",
    "aiodns": "3.1.1",
    "decorator": "4.4.2",
    "imgui-bundle": "1.92.601",
    "omniverseclient": "2.74.0",
    "tensordict": "0.14.2",
    "msal": "1.38.0",
    "alphashape": "1.3.1",
    "mujoco-warp": "3.12.0",
    "matplotlib": "3.10.8",
    "cmeel-boost": "1.89.0",
    "threadpoolctl": "3.6.0",
    "httpx": "0.28.1",
    "pure_eval": "0.2.3",
    "websockets": "14.2",
    "rich": "14.3.4",
    "awscrt": "0.23.8",
    "tensorboard-data-server": "0.7.2",
    "anyio": "4.14.2",
    "pyperclip": "1.11.0",
    "pycares": "5.0.1",
    "filelock": "3.25.2",
    "executing": "2.2.1",
    "einops": "0.9.0.dev0",
    "asttokens": "3.0.2",
    "rpds-py": "2026.6.3",
    "grpcio": "1.83.0",
    "python-dateutil": "2.9.0.post0",
    "omegaconf": "2.4.0",
    "newton-usd-schemas": "0.5.0",
    "qdldl": "0.1.7.post5",
    "platformdirs": "4.11.0",
    "cryptography": "50.0.1",
    "frozenlist": "1.5.0",
    "torchaudio": "2.11.0",
    "Flask": "3.1.3",
    "annotated-types": "0.8.0",
    "python-utils": "4.0.0",
    "mujoco": "3.12.0",
    "eigenpy": "3.12.0",
    "isaaclab_newton": "9.1.4",
    "httpcore": "1.0.9",
    "isaacsim-storage": "6.2.0.0",
    "isaacsim-gui": "6.2.0.0",
    "coal": "3.0.2",
    "isaacsim-rl": "6.2.0.0",
    "pytz": "2024.1",
    "isaacsim-extscache-kit": "6.2.0.0",
    "ipython_pygments_lexers": "1.1.1",
    "nvidia-curand": "10.4.0.35",
    "rtree": "1.4.1",
    "nvidia-cublas": "13.1.0.3",
    "psutil": "5.9.8",
    "PyOpenGL": "4.0.0a1",
    "tqdm": "4.70.0",
    "cycler": "0.12.1",
    "cmeel-assimp": "6.0.5",
    "isaacsim-kernel": "6.2.0.0",
    "httptools": "0.6.1",
    "pydantic_core": "2.47.0",
    "nvidia-cuda-nvrtc": "13.0.88",
    "rsl-rl-lib": "5.5.1",
    "markdown-it-py": "4.2.0",
    "isaaclab": "28.0.0",
    "urdf-usd-converter": "0.3.2",
    "ImageIO": "2.37.2",
    "fsspec": "2026.6.0",
    "packaging": "26.0",
    "cbor2": "5.9.0",
    "warp-lang": "1.17.0",
    "etils": "1.13.0",
    "matplotlib-inline": "0.2.2",
    "pyparsing": "3.3.2",
    "typing_extensions": "4.16.0",
    "fast_simplification": "0.1.13",
    "click-log": "0.4.0",
    "PyOpenGL-accelerate": "3.1.10",
    "idna": "3.10",
    "libcoal": "3.0.2",
    "isaacsim": "6.2.0.0",
    "loop-rate-limiters": "1.2.0",
    "ovstage": "0.2.1.385922",
    "cmeel-qhull": "8.0.2.1",
    "nvidia-cusolver": "12.0.4.66",
    "asteval": "1.0.6",
    "jsonschema-specifications": "2025.9.1",
    "addict": "2.4.0",
    "Pygments": "2.20.0",
    "llvmlite": "0.46.0",
    "dash": "4.4.1",
    "uvicorn": "0.29.0",
    "regex": "2026.7.19",
    "stack-data": "0.6.3",
    "isaaclab_physx": "8.0.0",
    "prettytable": "3.18.0",
    "open3d": "0.19.0",
    "parso": "0.8.7",
    "nvidia-cufile": "1.15.1.6",
    "nvidia-nvjitlink": "13.0.88",
    "tokenizers": "0.23.0rc0",
    "hydra-core": "1.4.0.dev10",
    "jupyter_core": "5.9.1",
    "flatbuffers": "25.12.19",
    "pyquaternion": "0.9.9",
    "multidict": "6.1.0",
    "botocore": "1.41.5",
    "cuda-toolkit": "13.0.2",
    "lazy-loader": "0.5",
    "isaacsim-asset": "6.2.0.0",
    "idna-ssl": "1.1.0",
    "zipp": "4.1.0",
    "nvidia-cufft": "12.0.0.61",
    "isaacsim-sensor": "6.2.0.0",
    "janus": "2.0.0",
    "pycparser": "3.0",
    "protobuf": "7.36.0rc1",
    "smmap": "5.0.3",
    "networkx": "3.6.1",
    "transformers": "5.10.4",
    "jupyterlab_widgets": "3.0.16",
    "click": "8.3.3",
    "py_trees": "2.4.0",
    "gitdb": "4.0.12",
    "osqp": "1.0.5",
    "msal-extensions": "1.3.1",
    "torchvision": "0.26.0+cu130",
    "python-multipart": "0.0.32",
    "huggingface_hub": "1.16.1",
    "isaacsim-extscache-kit-sdk": "6.2.0.0",
    "isaacsim-app": "6.2.0.0",
    "absl-py": "2.4.0",
    "comm": "0.2.3",
    "scipy": "1.17.0",
    "isaaclab-visualizers": "2.1.0",
    "qrcode": "7.4.2",
    "onnx": "1.22.0",
    "usd-exchange": "3.0.1",
    "scikit-learn": "1.9.0",
    "prompt_toolkit": "3.0.53",
    "itsdangerous": "2.2.0",
    "pypng": "0.20220715.0",
    "daqp": "0.8.5",
    "nvidia-cudnn-cu13": "9.19.0.56",
    "cuda-bindings": "13.3.1",
    "pin-pink": "3.3.0",
    "aiohappyeyeballs": "2.6.1",
    "isaacsim-ros2": "6.2.0.0",
    "setuptools": "81.0.0",
    "retrying": "1.4.2",
    "h11": "0.16.0",
    "importlib_metadata": "9.0.0",
    "orjson": "3.11.9",
    "sentry-sdk": "2.42.1",
    "sympy": "1.14.0",
    "coacd": "1.0.7",
    "cuda-pathfinder": "1.6.0",
    "meshio": "5.3.5",
    "pyglet": "2.1.15",
    "ml_dtypes": "0.5.4",
    "azure-core": "1.38.0",
    "isaacsim-code-editor": "6.2.0.0",
    "wcwidth": "0.8.2",
    "hf-xet": "1.6.0.dev0",
    "numpy-stl": "4.0.0",
    "jsonschema": "4.26.0",
    "fonttools": "4.61.1",
    "charset-normalizer": "3.3.2",
    "mujoco-usd-converter": "0.6.1",
    "oauthlib": "3.2.2",
    "isaaclab_ov": "5.1.5",
    "isaacsim-robot-motion": "6.2.0.0",
    "nvidia-cusparse": "12.6.3.3",
    "portalocker": "2.7.0",
    "isaacsim-replicator": "6.2.0.0",
    "PyYAML": "6.0.3",
    "certifi": "2026.7.22",
    "cmeel-octomap": "1.10.0",
    "isaacsim-template": "6.2.0.0",
    "PyJWT": "2.15.1",
    "Werkzeug": "3.1.8",
    "numba": "0.64.0",
    "importlib_resources": "7.1.0",
    "trimesh": "4.11.1",
    "MarkupSafe": "3.0.3",
    "boto3": "1.41.5",
    "plotly": "6.9.0",
    "s3transfer": "0.15.0",
    "annotated-doc": "0.0.4",
    "tensorboard": "2.21.0",
    "azure-storage-blob": "12.17.0",
    "shellingham": "1.5.4",
    "isodate": "0.6.1",
    "transitions": "0.9.3",
    "pycollada": "0.9.3",
    "nvidia-cusparselt-cu13": "0.8.0",
    "isaaclab_tasks": "25.2.0",
    "onnx-ir": "0.2.1",
    "isaaclab_rl": "1.5.1",
    "pyvers": "0.2.3",
    "requests": "2.32.3",
    "urllib3": "2.8.0",
    "widgetsnbextension": "4.0.15",
    "hidapi": "0.15.0",
    "isaacsim-core": "6.2.0.0",
    "resolve-robotics-uri-py": "0.5.1",
    "pydantic": "2.14.0a1",
    "shapely": "2.1.2",
    "isaacsim-benchmark": "6.2.0.0",
    "joblib": "1.5.3",
    "newton": "1.6.1",
    "Markdown": "3.10.2",
    "nvidia-cuda-cupti": "13.0.85",
    "cffi": "2.1.0",
    "yarl": "1.18.3",
    "nvidia-cuda-runtime": "13.0.96",
    "cmeel-zlib": "1.3.2",
    "isaaclab_contrib": "3.2.0",
    "onnxscript": "0.7.2.dev20260725",
    "isaacsim-ros1": "6.2.0.0",
    "glfw": "2.10.0",
    "isaacsim-test": "6.2.0.0",
    "cmeel-tinyxml2": "10.0.0",
    "libpinocchio": "3.9.0",
    "jedi": "0.20.0",
    "tinyobjloader": "2.0.0rc13",
    "typing-inspection": "0.4.2",
    "pin": "3.9.0",
    "starlette": "1.6.0",
    "six": "1.17.0",
    "attrs": "25.1.0",
    "mdurl": "0.1.2",
    "wrapt": "1.16.0",
    "pillow": "12.3.0",
    "aiofiles": "23.2.1",
    "ptyprocess": "0.7.0",
    "mpmath": "1.3.0",
    "isaacsim-robot": "6.2.0.0",
    "ConfigArgParse": "1.7.5",
    "coverage": "7.15.2",
    "propcache": "0.2.1",
    "pexpect": "4.9.0",
    "opencv-python-headless-noffmpeg": "4.14.0.94rc3",
    "isaaclab_ppisp": "0.3.1",
    "cloudpickle": "3.1.2",
    "isaacsim-extscache-physics": "6.2.0.0",
    "pandas": "3.0.5",
    "isaaclab_tasks_experimental": "1.0.1",
    "gymnasium": "1.3.0",
    "kiwisolver": "1.4.9",
    "isaaclab_assets": "2.0.1",
    "cmeel-console-bridge": "1.0.2.3",
    "cmeel": "0.60.1",
    "Farama-Notifications": "0.0.6",
    "async-timeout": "5.0.1",
    "numpy": "2.5.1",
    "safetensors": "0.8.0",
    "requests-oauthlib": "1.3.1",
    "isaacsim-robot-setup": "6.2.0.0",
    "qpsolvers": "4.13.0",
    "aioitertools": "0.11.0",
    "contourpy": "1.3.3",
    "Jinja2": "3.1.6",
    "aiobotocore": "2.26.0",
    "onnxruntime": "1.26.0",
    "nvidia-nvshmem-cu13": "3.4.5",
    "aiosignal": "1.4.0",
    "torch": "2.11.0+cu130",
    "nvidia-nvtx": "13.0.85",
    "jmespath": "1.0.1",
    "fastapi": "0.139.2",
    "narwhals": "2.24.0",
    "traitlets": "5.15.1",
    "cmeel-urdfdom": "4.0.1",
    "isaacsim-utils": "6.2.0.0",
    "pydot": "4.0.1",
    "isaaclab_experimental": "0.2.6",
    "toml": "0.10.2",
    "azure-identity": "1.13.0",
    "typer": "0.27.2",
    "fastjsonschema": "2.22.1",
    "triton": "3.6.0",
    "nest-asyncio": "1.6.0",
    "isaacsim-example": "6.2.0.0"
  }
}
```

## 实测或依赖记录：ROBOGAUGE_RUNTIME.json

```json
{
  "python": "3.11.18 (main, Oct 26 2026, 14:05:07) [Clang 23.1.3 ]",
  "versions": {
    "nvidia-nvtx-cu12": "12.4.99",
    "setuptools": "84.2.1",
    "grpcio": "1.85.0",
    "charset-normalizer": "3.5.2",
    "nvidia-cuda-cupti-cu12": "12.4.99",
    "nvidia-nvjitlink-cu12": "12.4.99",
    "pyparsing": "3.3.3",
    "lxml": "6.1.3",
    "tensorboard-data-server": "0.7.2",
    "robogauge": "1.1.9",
    "nvidia-nccl-cu12": "2.20.5",
    "Markdown": "3.11.2",
    "certifi": "2026.10.23",
    "fonttools": "4.68.0",
    "triton": "3.0.0",
    "glfw": "2.11.0",
    "kiwisolver": "1.5.1",
    "starlette": "1.10.0",
    "annotated-types": "0.8.0",
    "pydantic": "2.14.1",
    "nvidia-cusparse-cu12": "12.3.0.142",
    "nvidia-cublas-cu12": "12.4.2.65",
    "nvidia-cuda-nvrtc-cu12": "12.4.99",
    "numpy": "1.26.4",
    "requests": "2.34.2",
    "cycler": "0.12.1",
    "dm-tree": "0.1.8",
    "pydantic_core": "2.50.1",
    "typing-inspection": "0.4.4",
    "typing_extensions": "4.16.0",
    "imageio-ffmpeg": "0.7.0",
    "opentelemetry-api": "1.45.1",
    "uvicorn": "0.54.0",
    "nvidia-curand-cu12": "10.3.5.119",
    "python-dateutil": "3.0.0",
    "packaging": "26.3",
    "networkx": "3.6.1",
    "nvidia-cuda-runtime-cu12": "12.4.99",
    "psutil": "7.2.2",
    "h11": "0.16.0",
    "sympy": "1.14.0",
    "fsspec": "2026.9.0",
    "matplotlib": "3.6.3",
    "scipy": "1.17.1",
    "zipp": "4.1.2",
    "protobuf": "7.36.2",
    "annotated-doc": "0.0.5",
    "PyYAML": "6.0.3",
    "anyio": "4.15.1",
    "tqdm": "4.70.1",
    "tensorboard": "2.21.0",
    "torch": "2.4.1+cu124",
    "pygame": "2.6.1",
    "urllib3": "2.8.0",
    "nvidia-cufft-cu12": "11.2.0.44",
    "Pillow": "9.5.0",
    "idna": "3.20",
    "dm-env": "1.6",
    "etils": "1.14.0",
    "dm_control": "1.0.23",
    "six": "1.17.0",
    "tzdata": "2026.6",
    "filelock": "4.2.4",
    "mpmath": "1.3.0",
    "mujoco": "3.2.3",
    "absl-py": "2.5.1",
    "ImageIO": "2.38.1",
    "nvidia-cusolver-cu12": "11.6.0.99",
    "fastapi": "0.162.0",
    "click": "8.5.0",
    "MarkupSafe": "3.0.4",
    "contourpy": "1.3.3",
    "Jinja2": "3.1.6",
    "Werkzeug": "3.1.9",
    "PyOpenGL": "3.1.10",
    "labmaze": "1.0.6",
    "nvidia-cudnn-cu12": "9.1.0.70"
  },
  "golden_checks": [
    {
      "model": "10000_student_golden",
      "max_abs_error": 7.152557373046875e-07,
      "reset_at": [
        0,
        25
      ]
    },
    {
      "model": "10000_teacher_golden",
      "max_abs_error": 7.748603820800781e-07,
      "reset_at": [
        0,
        25
      ]
    },
    {
      "model": "40000_student_golden",
      "max_abs_error": 1.1920928955078125e-06,
      "reset_at": [
        0,
        25
      ]
    },
    {
      "model": "40000_teacher_golden",
      "max_abs_error": 7.748603820800781e-07,
      "reset_at": [
        0,
        25
      ]
    },
    {
      "model": "5000_student_golden",
      "max_abs_error": 5.960464477539062e-07,
      "reset_at": [
        0,
        25
      ]
    },
    {
      "model": "5000_teacher_golden",
      "max_abs_error": 7.152557373046875e-07,
      "reset_at": [
        0,
        25
      ]
    }
  ],
  "training_updates": 0
}
```

## 实测或依赖记录：verification.json

```json
{
  "base_release": "fa3c776af150986776b5c513130c0e7a8b38b4c6",
  "scope": "dual robot asset/config/entrypoint + documentation consolidation",
  "cpu_test_summary": "...........s.........s.s..\n----------------------------------------------------------------------\nRan 25 tests in 0.061s\n\nOK (skipped=3)\n",
  "robot_load_checks": {
    "go2": {
      "backend": "isaaclab_physx.assets.articulation.articulation",
      "nominal_mass_kg": [
        16.086999893188477,
        16.086999893188477,
        16.086999893188477,
        16.086999893188477
      ],
      "control_steps": 50,
      "falls": 0,
      "policy_updates": 0,
      "capability_claim": false
    },
    "as2": {
      "backend": "isaaclab_physx.assets.articulation.articulation",
      "nominal_mass_kg": [
        17.639999389648438,
        17.639999389648438,
        17.639999389648438,
        17.639999389648438
      ],
      "control_steps": 50,
      "falls": 0,
      "policy_updates": 0,
      "capability_claim": false
    }
  },
  "as2_source_manifest_all_files_verified": true,
  "new_formal_training_started": false,
  "previous_go2_run_modified": false,
  "as2_experts_verified": false,
  "as2_official_sim2sim_verified": false,
  "as2_training_updates_tested": false
}
```

## 可执行机制与全部固定参数源码附录

以下直接取自当前代码，包含未暴露为CLI的阈值、常数和完整计算；SHA256绑定对应文件。正文是解释，附录用于逐项核对，不是另一套配置。

### config.py · SHA256 6d920962878ea4529540f8d77ca02f7f96ff00e89d4e59d6325e22fd8205895c

```python
"""Explicit source-aligned defaults; task and CTS adaptations are named."""
from dataclasses import dataclass

@dataclass
class TrainCfg:
    robot: str = "go2"
    stage: str = "flat"
    num_envs: int = 4096
    buffer_len: int = 1000
    collect_steps: int = 1
    iterations: int = 60000
    warmup_iterations: int = 300
    wm_hidden: tuple = (1024, 1024, 1024)
    wm_window: int = 8
    wm_batch: int = 2048
    wm_updates: int = 2
    wm_prioritized_frac: float = .25
    wm_archive_frac: float = .25
    archive_size: int = 100000
    archive_keep_prob: float = .02
    imag_batch: int = 8192
    teacher_fraction: float = .75
    horizon: int = 32
    actor_microbatch: int = 8192
    actor_hidden: tuple = (512, 256, 128)
    critic_hidden: tuple = (512, 256, 128)
    cts_updates: int = 20
    cts_batch: int = 1024
    cts_lr: float = .001
    cts_balance: float = .01
    critic_epochs: int = 5
    critic_minibatches: int = 4
    lr: float = 2e-4
    actor_lr_scale: float = 1.
    lr_final_frac: float = .03
    lr_decay_iterations: int = 30000
    actor_ramp_iterations: int = 1000
    critic_warmup_iterations: int = 500
    grad_clip: float = 1.
    grad_clip_elem: float = .1
    gamma: float = .99
    lam: float = .95
    action_noise: float = .1
    explore_frac: float = .25
    explore_noise: float = .3
    explore_theta: float = 2.
    driver_policy: str = ""
    driver_frac: float = 0.
    expert_policy: str = ""
    expert_frac: float = 0.
    allow_missing_flat_expert: bool = False
    collapse_falls: float = 3.
    log_interval: int = 50
    save_interval: int = 1000
    seed: int = 17
    dt: float = .02
    physics_dt: float = .005
    episode_seconds: float = 20.
    payload_max_kg: float = 2.
    reset_critic: bool = False
    evaluation_interval: int = 5000
    evaluation_python: str = ''
    compile: bool = True
    device: str = "cuda:0"

    @classmethod
    def from_record(cls, record):
        """Migrate historical configuration metadata without changing checkpoint bytes."""
        values=dict(record)
        for obsolete in ('evaluation_xml','evaluation_profile','adaptive_iterations',
                         'terrain_curriculum_iterations','command_max'):
            values.pop(obsolete,None)
        return cls(**values)

    @classmethod
    def terrain(cls, **kwargs):
        return cls(stage="terrain", wm_updates=4, actor_lr_scale=.3,
                   warmup_iterations=1000, driver_frac=.1, expert_frac=.1, **kwargs)

    def validate(self, production=True):
        from .robots import profile
        profile(self.robot)
        if self.robot == 'as2' and (self.driver_frac or self.expert_frac or self.evaluation_python):
            raise ValueError('As2 experts and official sim2sim are not verified; Go2 providers/evaluator are forbidden')
        if self.stage not in ("flat", "terrain"):
            raise ValueError("Unknown stage")
        if self.collect_steps != 1 or self.horizon != 32 or self.wm_window != 8:
            raise ValueError("Source-aligned collect/H/K contract changed")
        if production and (self.num_envs, self.buffer_len, self.imag_batch, self.wm_batch) != (4096,1000,8192,2048):
            raise ValueError("Production scale must be4096/1000/8192/2048")
        if production and self.actor_microbatch!=self.imag_batch:
            raise ValueError('Production uses one8192 imagination cohort so global command-stage activation is shared')
        if self.wm_updates != (2 if self.stage == "flat" else 4):
            raise ValueError("WM updates must match stage recipe")
        if not 0 < self.teacher_fraction < 1:
            raise ValueError("Both CTS branches required")
        if not 0 <= self.driver_frac + self.expert_frac < 1:
            raise ValueError("Invalid data source fractions")
        if self.driver_frac and not self.driver_policy:
            raise ValueError("Continuous mixed-terrain expert missing; no silent fallback")
        if self.expert_frac and not self.expert_policy:
            raise ValueError("Flat expert missing; set fraction0 and explicitly acknowledge degraded recipe")
        if self.stage == "terrain" and not self.expert_frac and not self.allow_missing_flat_expert:
            raise ValueError("Terrain recipe lacks its10% flat skill expert")
        if self.payload_max_kg < 0 or self.payload_max_kg > 2:
            raise ValueError("Only fixed nonnegative0..2kg payload is supported")

```

### robots.py · SHA256 c96d3650c9aeade90fc061f9165f0bd8b6c0eed01996f609455ca270ca4b314e

```python
"""Robot-specific contracts. As2 is a new PhysX port, not a certified policy."""
from dataclasses import dataclass

@dataclass(frozen=True)
class RobotProfile:
    name: str
    initial_height: float
    left_hip: float
    reward_height: float
    calf_length: float | None
    velocity_limits: tuple

PROFILES = {
    'go2': RobotProfile('go2', .29, .1, .38, .213, (30.1, 30.1, 20.07)),
    # .30 is the prior project's engineering stance target, not an official reward.
    # Fixed foot bodies are mandatory: no Go2 geometry fallback is permitted.
    'as2': RobotProfile('as2', .4, -.1, .30, None, (24., 24., 16.)),
}

def profile(name):
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f'Unknown robot: {name}') from None

def as2_motor_values(device):
    import torch
    def t(values): return torch.tensor(values * 4, device=device)
    # Unitree unitree_rl_mjlab As2 constants; fixed-torque PD, not Go2 DC curve.
    inertia, kp, kd, effort = t([.026,.026,.038]), t([40.,40.,60.]), t([2.,2.,3.]), t([50.,50.,75.])
    # Project action normalization, explicitly not a manufacturer calibration.
    return inertia, kp, kd, effort, .25 * effort / kp

```

### networks.py · SHA256 9dc476a6bb1cb31d3c4938d2f7d5b9703dfa6a5ae0c8b867b3197aa53ddbe630

```python
"""Source RunningNorm/MLP and retained Go2RLLab MoE-CTS parameter ownership."""
import torch
from torch import nn
from .moe import MLP, MoE, L2Norm
from . import state as st

class RunningNorm(nn.Module):
    def __init__(self,dim,eps=1e-4,clip=10.):
        super().__init__();self.clip=clip
        self.register_buffer('mean',torch.zeros(dim));self.register_buffer('var',torch.ones(dim))
        self.register_buffer('count',torch.tensor(eps))
    @torch.no_grad()
    def update(self,x):
        x=x.reshape(-1,x.shape[-1])
        if not x.shape[0]:return
        b_mean,b_var,b_n=x.mean(0),x.var(0,unbiased=False),x.shape[0]
        delta=b_mean-self.mean;tot=self.count+b_n
        self.mean+=delta*b_n/tot
        self.var=(self.var*self.count+b_var*b_n+delta**2*self.count*b_n/tot)/tot
        self.count=tot
    def forward(self,x):return ((x-self.mean)/(self.var.sqrt()+1e-6)).clamp(-self.clip,self.clip)
    def unnormalize(self,y):return y*(self.var.sqrt()+1e-6)+self.mean

def mlp(input_dim,hidden,out_dim,act=nn.SiLU):
    layers=[]
    for h in hidden:layers += [nn.Linear(input_dim,h),act()];input_dim=h
    return nn.Sequential(*layers,nn.Linear(input_dim,out_dim))

class Student(nn.Module):
    def __init__(self):
        super().__init__();self.moe=MoE(8,450,(512,256,256),32,'elu');self.norm_layer=L2Norm()
    def forward(self,x):
        z,g=self.moe(x);return self.norm_layer(z),g

class Policy(nn.Module):
    def __init__(self,hidden=(512,256,128)):
        super().__init__()
        self.history_norm=RunningNorm(450);self.single_norm=RunningNorm(45);self.teacher_norm=RunningNorm(st.PRIV_DIM)
        self.teacher_encoder=nn.Sequential(MLP(st.PRIV_DIM,32,(512,256),'elu'),L2Norm())
        self.student_encoder=Student();self.actor=MLP(77,12,hidden,'elu')
    def forward(self,hist,priv,branch):
        if isinstance(branch,tuple):
            split=branch[0]
            z=torch.cat((self.teacher_encoder(self.teacher_norm(priv[:split])),self.student_encoder(self.history_norm(hist[split:]))[0]),0)
        else:z=self.teacher_encoder(self.teacher_norm(priv)) if branch==0 else self.student_encoder(self.history_norm(hist))[0]
        return self.actor(torch.cat((z,self.single_norm(st.single_from_history(hist))),-1))
    @torch.no_grad()
    def update_stats(self,hist,priv):
        self.history_norm.update(hist);self.single_norm.update(st.single_from_history(hist));self.teacher_norm.update(priv)

class Critic(nn.Module):
    def __init__(self,hidden=(512,256,128)):
        super().__init__();self.norm=RunningNorm(st.PRIV_DIM+2);self.net=mlp(st.PRIV_DIM+2,hidden,1,nn.ELU)
    def forward(self,x):return self.net(self.norm(x)).squeeze(-1)
    @staticmethod
    def features(priv,branch):
        tag=priv.new_zeros((priv.shape[0],2))
        if isinstance(branch,tuple):tag[:branch[0],0]=1;tag[branch[0]:,1]=1
        else:tag[:,branch]=1
        return torch.cat((priv,tag),-1)

class StudentExport(nn.Module):
    """Stateless450 history input,12 raw actions; no privileged inputs."""
    def __init__(self,p):
        super().__init__();self.history_norm=p.history_norm;self.single_norm=p.single_norm
        self.student_encoder=p.student_encoder;self.actor=p.actor
    def forward(self,hist):
        z,_=self.student_encoder(self.history_norm(hist))
        return self.actor(torch.cat((z,self.single_norm(st.single_from_history(hist))),-1))

```

### moe.py · SHA256 c373ce1cd3312de4413bd315f2f063f67502aa62c7b936b356f93dd678e1879f

```python
# Copyright (c) 2026 Tianyang Wu
# SPDX-License-Identifier: MIT

"""Mixture-of-experts network components used by MoE-CTS."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as functional


class CatELU(nn.Module):
    """Apply ELU to a tensor and its negation, doubling the feature dimension."""

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.elu = nn.ELU(inplace=inplace)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.dim() < 2:
            raise ValueError(f"CatELU expects a feature dimension, got shape {tuple(value.shape)}.")
        return torch.cat((self.elu(value), self.elu(-value)), dim=-1)


class L2Norm(nn.Module):
    """Normalize the last dimension to unit L2 norm."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return functional.normalize(value, p=2.0, dim=-1)


class SimNorm(nn.Module):
    """Normalize fixed-size latent groups with a simplex softmax."""

    def __init__(self, group_dim: int = 8) -> None:
        super().__init__()
        self.group_dim = int(group_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] % self.group_dim != 0:
            raise ValueError(
                f"SimNorm input dimension {value.shape[-1]} must be divisible by group_dim {self.group_dim}."
            )
        shape = value.shape
        normalized = functional.softmax(value.reshape(*shape[:-1], -1, self.group_dim), dim=-1)
        return normalized.reshape(*shape)

    def __repr__(self) -> str:
        return f"SimNorm(group_dim={self.group_dim})"


def get_activation(name: str) -> nn.Module:
    """Resolve the activation names supported by the reference MoE implementation."""
    activations: dict[str, type[nn.Module]] = {
        "elu": nn.ELU,
        "selu": nn.SELU,
        "relu": nn.ReLU,
        "crelu": nn.ReLU,
        "lrelu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "cat_elu": CatELU,
    }
    try:
        return activations[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"Unsupported MoE activation {name!r}; expected one of {sorted(activations)}.") from exc


class MLP(nn.Module):
    """Reference-compatible MLP with an explicit ``network`` state-dict prefix."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int | tuple[int, ...] | list[int],
        hidden_dims: tuple[int, ...] | list[int],
        activation: str = "elu",
        last_activation: str | None = None,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("MoE MLP requires at least one hidden layer.")

        layers: list[nn.Module] = []
        last_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, int(hidden_dim)))
            layers.append(get_activation(activation))
            last_dim = int(hidden_dim) * (2 if activation == "cat_elu" else 1)

        if isinstance(output_dim, int):
            layers.append(nn.Linear(last_dim, output_dim))
        else:
            output_shape = tuple(int(dim) for dim in output_dim)
            layers.append(nn.Linear(last_dim, math.prod(output_shape)))
            layers.append(nn.Unflatten(dim=-1, unflattened_size=output_shape))

        if last_activation is not None:
            layers.append(get_activation(last_activation))
        self.network = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class Experts(nn.Module):
    """Generate all expert outputs from a shared backbone and grouped heads."""

    def __init__(
        self,
        expert_num: int,
        input_dim: int,
        backbone_hidden_dims: tuple[int, ...] | list[int],
        expert_hidden_dim: int,
        output_dim: int,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.expert_num = int(expert_num)
        self.output_dim = int(output_dim)
        self.backbone = MLP(
            input_dim,
            self.expert_num * int(expert_hidden_dim),
            backbone_hidden_dims,
            activation,
            last_activation=activation,
        )
        activation_multiplier = 2 if activation == "cat_elu" else 1
        self.experts = nn.Conv1d(
            in_channels=self.expert_num * int(expert_hidden_dim) * activation_multiplier,
            out_channels=self.expert_num * self.output_dim,
            kernel_size=1,
            groups=self.expert_num,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        shared_features = self.backbone(value).unsqueeze(-1)
        expert_outputs = self.experts(shared_features).squeeze(-1)
        return expert_outputs.reshape(-1, self.expert_num, self.output_dim)


class MoE(nn.Module):
    """Dense mixture of experts with a learned softmax gating network."""

    def __init__(
        self,
        expert_num: int,
        input_dim: int,
        hidden_dims: tuple[int, ...] | list[int],
        output_dim: int,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        if len(hidden_dims) < 2:
            raise ValueError("MoE hidden_dims must contain a backbone and an expert hidden dimension.")
        self.experts = Experts(
            expert_num=expert_num,
            input_dim=input_dim,
            backbone_hidden_dims=hidden_dims[:-1],
            expert_hidden_dim=hidden_dims[-1],
            output_dim=output_dim,
            activation=activation,
        )
        self.gating_network = nn.Sequential(
            MLP(input_dim, expert_num, hidden_dims, activation),
            nn.Softmax(dim=-1),
        )

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.gating_network(value)
        expert_outputs = self.experts(value)
        output = torch.sum(weights.unsqueeze(-1) * expert_outputs, dim=1)
        return output, weights


__all__ = ["CatELU", "Experts", "L2Norm", "MLP", "MoE", "SimNorm", "get_activation"]

```

### state.py · SHA256 dfb78d6ffa85c70cc88978274a039007dc8bd5b54ce03363a29806bd1a80971e

```python
"""Source70-D state and45x10 blind CTS interface; no simulator imports."""
import torch
STATE_DIM, POSE_DELTA_DIM, CONTEXT_DIM, NUM_JOINTS = 70, 3, 5, 12
Z, GRAV, V_YAW, W_B = slice(0,1), slice(1,4), slice(4,7), slice(7,10)
Q, QD, HIPS, KNEES, FEET = slice(10,22),slice(22,34),slice(34,46),slice(46,58),slice(58,70)
LEGS = ("FL", "FR", "RL", "RR")
JOINT_NAMES = tuple(f"{leg}_{joint}_joint" for leg in LEGS for joint in ("hip","thigh","calf"))
TERM_DIMS = (3,3,3,12,12,12)
HISTORY, OBS_DIM, SINGLE_DIM = 10, 450, 45
PRIV_DIM = STATE_DIM + 3 + 12 + CONTEXT_DIM + 77

def feet_world_z(s):
    return s[:,Z]+s[:,FEET].reshape(-1,4,3)[...,2]

def yaw_inverse(v, yaw):
    c,s=yaw.cos(),yaw.sin()
    return torch.stack((c*v[...,0]+s*v[...,1],-s*v[...,0]+c*v[...,1],v[...,2]),-1)

def velocity_body(s):
    """Undo pitch and roll after yaw removal; command axes remain body axes."""
    g=s[:,GRAV];roll=torch.atan2(-g[:,1],-g[:,2])
    pitch=torch.atan2(g[:,0],torch.sqrt(g[:,1].square()+g[:,2].square()).clamp_min(1e-6))
    vx,vy,vz=s[:,V_YAW].unbind(-1);cp,sp=pitch.cos(),pitch.sin();cr,sr=roll.cos(),roll.sin()
    xp=cp*vx-sp*vz;zp=sp*vx+cp*vz
    return torch.stack((xp,cr*vy+sr*zp,-sr*vy+cr*zp),-1)

def pose_delta(p0,p1):
    d=p1-p0;c,s=p0[:,2].cos(),p0[:,2].sin()
    return torch.stack((c*d[:,0]+s*d[:,1],-s*d[:,0]+c*d[:,1],torch.atan2(d[:,2].sin(),d[:,2].cos())),-1)

def apply_pose_delta(p,d):
    c,s=p[:,2].cos(),p[:,2].sin()
    return torch.stack((p[:,0]+c*d[:,0]-s*d[:,1],p[:,1]+s*d[:,0]+c*d[:,1],p[:,2]+d[:,2]),-1)

def frame(s,a,command,noise=False):
    # Preserve deployed CTS term order and units/scales; running norms follow.
    terms=[s[:,W_B]*.25,s[:,GRAV],command*command.new_tensor([.5,1,1]),s[:,Q],s[:,QD]*.05,a]
    if noise:
        amplitudes=(.05,.05,0.,.01,.075,0.)
        terms=[x+(2*torch.rand_like(x)-1)*amp if amp else x for x,amp in zip(terms,amplitudes)]
    return torch.cat(terms,-1)

def push_history(hist,one):
    blocks=[];i=j=0
    for d in TERM_DIMS:
        blocks.append(torch.cat((hist[:,i+d:i+d*HISTORY],one[:,j:j+d]),-1))
        i+=d*HISTORY;j+=d
    return torch.cat(blocks,-1)

def single_from_history(hist):
    out=[];i=0
    for d in TERM_DIMS:
        out.append(hist[:,i+d*(HISTORY-1):i+d*HISTORY]);i+=d*HISTORY
    return torch.cat(out,-1)

def privileged(s,a,command,xi,relief):
    return torch.cat((s,command,a,xi,s[:,Z]+relief),-1)

```

### world_model.py · SHA256 28696d9a0b8065694bd5d1f2dbdccf0561aa9c7a313af83a3f842a63e5938801

```python
"""Deterministic world model f(s, a, a_prev, xi, ground scan) -> (s', planar pose delta).

Fit on k-step autoregressive windows from the replay buffer (SuperMimic Eq. 1,
state term only; reward and termination are analytic in imagine.py). Inputs
and outputs are normalized with running statistics from collected data.
"""

from __future__ import annotations

import torch
from torch import nn

from . import state as st
from .terrain import SCAN_POINTS


from .networks import RunningNorm, mlp


class WorldModel(nn.Module):
  """s' = f(s, a, a_prev, xi, scan). The scan is the base height above each point
  of the local ground grid (terrain and obstacles, see target/terrain.py), so
  stepping onto a stair or pushing into a wall is predictable."""

  def __init__(self, hidden: tuple[int, ...] = (1024, 1024, 1024)):
    super().__init__()
    in_dim = st.STATE_DIM + 2 * st.NUM_JOINTS + st.CONTEXT_DIM + SCAN_POINTS
    out_dim = st.STATE_DIM + st.POSE_DELTA_DIM
    self.net = mlp(in_dim, hidden, out_dim)
    self.in_norm = RunningNorm(in_dim)
    # Normalizes the per-step change of s and the pose delta.
    self.out_norm = RunningNorm(out_dim)

  @staticmethod
  def _inputs(s, a, a_prev, xi, relief):
    return torch.cat([s, a, a_prev, xi, s[:, st.Z] + relief], dim=-1)

  def update_stats(self, s, a, a_prev, xi, relief, s_next, pose_delta) -> None:
    self.in_norm.update(self._inputs(s, a, a_prev, xi, relief))
    self.out_norm.update(torch.cat([s_next - s, pose_delta], dim=-1))

  def forward(self, s, a, a_prev, xi, relief):
    """relief (B, P): ground under the base minus the ground at each scan point.
    Returns (s_next, pose_delta)."""
    y = self.out_norm.unnormalize(self.net(self.in_norm(self._inputs(s, a, a_prev, xi, relief))))
    return s + y[:, : st.STATE_DIM], y[:, st.STATE_DIM :]

  def loss(self, s0, actions, actions_prev, xi, reliefs, s_targets, pose_targets, valid):
    """k-step autoregressive loss in normalized output units.

    s0 (B, S); actions, actions_prev (B, k, 12); xi (B, C); reliefs (B, k, P) at
    the recorded poses; s_targets (B, k, S); pose_targets (B, k, 3); valid (B, k)
    masks steps after a reset inside the window.
    """
    k = actions.shape[1]
    scale = self.out_norm.var.sqrt() + 1e-6
    s = s0
    total, denom = 0.0, valid.sum().clamp(min=1.0)
    for j in range(k):
      s_pred, pose_pred = self(s, actions[:, j], actions_prev[:, j], xi, reliefs[:, j])
      err = torch.cat([s_pred - s_targets[:, j], pose_pred - pose_targets[:, j]], dim=-1)
      total = total + ((err / scale) ** 2).mean(-1).mul(valid[:, j]).sum()
      s = s_pred
    return total / denom

```

### buffer.py · SHA256 4c41e864e2ecc9911ec195174e4e14376c31c4abcaedf34af31bac7e501b4549

```python
"""One shared online replay for all providers and both actor branches.

Window selection/archive recipe follows pinned upstream train.py. Episode-end
transitions are excluded as in upstream; simulator reset jumps never become labels.
"""
import torch
from . import state as st
from .commands import CMD_STATE_DIM

class Replay:
    def __init__(self,n,length,device):
        self.n,self.L,self.device=n,length,device;self.ptr=self.size=0
        self.data={}
        for name,dim,dtype in [('s',70,torch.float32),('pose',3,torch.float32),('a',12,torch.float32),
                              ('a_prev',12,torch.float32),('obs',450,torch.float16),('relief',77,torch.float16),
                              ('command',3,torch.float32),('timer',1,torch.float32),('cmd_state',CMD_STATE_DIM,torch.float32),('done',1,torch.bool),
                              ('source',1,torch.uint8),('episode',1,torch.int32),('xi',st.CONTEXT_DIM,torch.float32)]:
            self.data[name]=torch.zeros((n,length,dim),device=device,dtype=dtype)
    def add(self,**values):
        if set(values)!=set(self.data):raise ValueError('Replay schema mismatch')
        for key,dst in self.data.items():dst[:,self.ptr]=values[key].reshape(dst.shape[0],dst.shape[-1])
        self.ptr=(self.ptr+1)%self.L;self.size=min(self.size+1,self.L)
    def physical(self,t):return (self.ptr-self.size+t)%self.L
    def sample_states(self,batch,source=None):
        if source is None:
            env=torch.randint(self.n,(batch,),device=self.device)
            slot=self.physical(torch.randint(self.size,(batch,),device=self.device))
        else:
            # Providers are fixed per environment, avoiding a4M-row scan every CTS minibatch.
            candidates=(self.data['source'][:,self.physical(0),0]==source).nonzero().flatten()
            if not len(candidates):raise ValueError('Requested replay source has no samples')
            env=candidates[torch.randint(len(candidates),(batch,),device=self.device)]
            slot=self.physical(torch.randint(self.size,(batch,),device=self.device))
        return env,slot
    def seeds(self,batch,xi,source=None):
        e,t=self.sample_states(batch,source)
        out={k:v[e,t].float() for k,v in self.data.items()}
        out['env']=e
        return out
    def windows(self,batch,k,xi):
        if self.size<=k+1:raise ValueError('Not enough contiguous history')
        e=torch.randint(self.n,(batch,),device=self.device)
        t=torch.randint(self.size-k,(batch,),device=self.device)
        idx=self.physical(t[:,None]+torch.arange(k+1,device=self.device));ee=e[:,None]
        done=self.data['done'][ee,idx[:,:k],0].float()
        valid=(1-done).cumprod(1)
        # Explicit episode identity catches any accidental writer boundary mismatch.
        ep=self.data['episode'][ee,idx,0]
        valid*=((ep[:,1:]==ep[:,:1]).float()).cumprod(1)
        return {'s':self.data['s'][ee,idx],'pose':self.data['pose'][ee,idx],
                'a':self.data['a'][ee,idx[:,:k]],'a_prev':self.data['a_prev'][ee,idx[:,:k]],
                'relief':self.data['relief'][ee,idx[:,:k]],'xi':self.data['xi'][e,idx[:,0]],
                'command':self.data['command'][e,idx[:,0]],'valid':valid}
    @property
    def storage_bytes(self):return sum(x.numel()*x.element_size() for x in self.data.values())

class Archive:
    def __init__(self,capacity,k,device):
        self.cap,self.device,self.ptr,self.size=capacity,device,0,0
        shapes={'s':(k+1,70),'pose':(k+1,3),'a':(k,12),'a_prev':(k,12),'xi':(st.CONTEXT_DIM,), 'command':(3,), 'valid':(k,),'relief':(k,77)}
        self.data={key:torch.zeros((capacity,*shape),device=device,dtype=torch.float16 if key=='relief' else torch.float32) for key,shape in shapes.items()}
    def add(self,w):
        n=min(len(w['s']),self.cap)
        if not n:return
        idx=(self.ptr+torch.arange(n,device=self.device))%self.cap
        for k in self.data:self.data[k][idx]=w[k][:n]
        self.ptr=(self.ptr+n)%self.cap;self.size=min(self.size+n,self.cap)
    def sample(self,n):
        idx=torch.randint(self.size,(n,),device=self.device)
        return {k:v[idx] for k,v in self.data.items()}

def rarity(w):
    s=w['s'];contact=(st.feet_world_z(s.reshape(-1,70))<.035).reshape(s.shape[0],-1,4).float()
    return (contact[:,1:]-contact[:,:-1]).abs().sum((1,2))+(contact.sum(-1)==0).float().sum(-1)+(w['valid'][:,-1]<.5).float()*4

def archive_worthy(w):
    s=w['s'];feet=st.feet_world_z(s.reshape(-1,70)).reshape(s.shape[0],-1,4)
    return (((feet>=.035).all(-1).float().sum(-1))>=2)|(s[:,:,6].abs().amax(-1)>.6)|(w['valid'][:,-1]<.5)

def mixed_windows(buf,archive,cfg,xi):
    nprio=int(cfg.wm_batch*cfg.wm_prioritized_frac)
    narch=int(cfg.wm_batch*cfg.wm_archive_frac) if archive.size else 0
    parts=[buf.windows(cfg.wm_batch-nprio-narch,cfg.wm_window,xi)]
    cand=buf.windows(4*max(nprio,1),cfg.wm_window,xi);score=rarity(cand)
    keep=archive_worthy(cand)&(torch.rand_like(score)<cfg.archive_keep_prob)
    archive.add({k:v[keep] for k,v in cand.items()})
    if nprio:
        top=(score+1e-3*torch.rand_like(score)).topk(nprio).indices
        parts.append({k:v[top] for k,v in cand.items()})
    if narch:parts.append(archive.sample(narch))
    return {k:torch.cat([p[k] for p in parts]) for k in parts[0]}

```

### imagine.py · SHA256 922b85da2dede3ba4b89b3e09e77b280dbfcd7606a1790516349a41f8c01e651

```python
"""Source BPTT/soft-survival return with three-velocity command and CTS adapters."""
import torch
from . import state as st
from .terrain import relief
from .task import reward
from .commands import OfficialCommands,EPISODE_ELAPSED


def time_limit_return(rewards,soft_done,timeout,values,gamma,valid_start=None):
    """Differentiable short return: time limits bootstrap, falls end survival.

    rewards/soft_done/timeout are[B,H], values[B,H+1]. The terminal
    transition still earns its reward. Values retain state-input gradients.
    """
    continuation=(1-soft_done)*(1-timeout.to(rewards.dtype))
    after=continuation.cumprod(-1)
    before=torch.cat((torch.ones_like(after[:,:1]),after[:,:-1]),-1)
    if valid_start is not None:
        before=before*valid_start[:,None];after=after*valid_start[:,None]
    discount=gamma**torch.arange(rewards.shape[1],device=rewards.device,dtype=rewards.dtype)
    immediate=(discount*before*rewards).sum(-1)
    timeout_bootstrap=(gamma*discount*before*(1-soft_done)*timeout*values[:,1:]).sum(-1)
    horizon_bootstrap=gamma**rewards.shape[1]*after[:,-1]*values[:,-1]
    bootstrap=timeout_bootstrap+horizon_bootstrap
    initial=torch.ones_like(after[:,:1]) if valid_start is None else valid_start[:,None].to(after.dtype)
    return immediate+bootstrap,torch.cat((initial,after),-1),immediate,bootstrap


@torch.no_grad()
def time_limit_lambda_targets(rewards,soft_done,timeout,values,gamma,lam):
    """TD-lambda cannot propagate a return across an episode time limit."""
    nxt=values[:,-1];targets=[]
    for t in reversed(range(rewards.shape[1])):
        tail=torch.where(timeout[:,t],values[:,t+1],(1-lam)*values[:,t+1]+lam*nxt)
        nxt=rewards[:,t]+gamma*(1-soft_done[:,t])*tail
        targets.append(nxt)
    return torch.stack(targets[::-1],1)

class Imagination:
    def __init__(self,cfg,consts,field=None,commands=None):
        self.cfg,self.c,self.field=cfg,consts,field
        from .robots import profile
        self.reward_height=profile(cfg.robot).reward_height
        self.commands=commands or OfficialCommands(cfg.dt,cfg.episode_seconds)
        self._step_fn=torch.compile(self._step,dynamic=False) if cfg.compile else self._step
    def _step(self,wm,policy,critic,s,pose,a_prev,hist,cmd_state,xi,branch,iteration,active):
        cfg=self.cfg;command=self.commands.command(cmd_state)
        rlf=relief(pose,self.field);priv=st.privileged(s,a_prev,command,xi,rlf)
        feature=critic.features(priv,branch)
        a=policy(hist,priv,branch)
        ns,dp=wm(s,a,a_prev,xi,rlf);np=st.apply_pose_delta(pose,dp)
        r,_,done=reward(s,ns,a,a_prev,command,self.c,cfg.stage,soft=True,iteration=iteration,pose=np,field=self.field,base_height_target=self.reward_height)
        # An inactive row is still evaluated in the fixed-size compiled graph.
        # Give only that discarded command candidate a safe elapsed time; keep
        # its actual terminal memory frozen, avoiding negative remaining-time
        # denominators after an already completed episode.
        elapsed=torch.where(active,cmd_state[:,EPISODE_ELAPSED],torch.zeros_like(cmd_state[:,EPISODE_ELAPSED]))
        candidate_command=torch.cat((cmd_state[:,:EPISODE_ELAPSED],elapsed[:,None],cmd_state[:,EPISODE_ELAPSED+1:]),-1)
        candidate_command=self.commands.advance(candidate_command,np[:,:2],iteration)
        command=self.commands.command(candidate_command)
        nhist=st.push_history(hist,st.frame(ns,a,command,noise=True))
        def select(new,old):return torch.where(active[:,None],new,old)
        return (select(ns,s),select(np,pose),select(a,a_prev),select(nhist,hist),
                select(candidate_command,cmd_state),torch.where(active,r,torch.zeros_like(r)),
                torch.where(active,done,torch.zeros_like(done)),feature)
    def rollout(self,wm,policy,critic,seed,branch,iteration=0):
        cfg=self.cfg;s,pose,a_prev,hist=seed['s'],seed['pose'],seed['a_prev'],seed['obs']
        cmd_state,xi=seed['cmd_state'],seed['xi']
        iteration=torch.as_tensor(iteration,device=s.device)
        max_steps=round(cfg.episode_seconds/cfg.dt)
        active=torch.round(cmd_state[:,EPISODE_ELAPSED]/cfg.dt)<max_steps
        valid_start=active.to(s.dtype)
        rewards=[];soft_done=[];timeouts=[];valid=[];features=[]
        for _ in range(cfg.horizon):
            valid.append(active)
            timeout=active&(torch.round(cmd_state[:,EPISODE_ELAPSED]/cfg.dt)+1>=max_steps)
            s,pose,a_prev,hist,cmd_state,r,done,feature=self._step_fn(wm,policy,critic,s,pose,a_prev,hist,cmd_state,xi,branch,iteration,active)
            features.append(feature)
            rewards.append(r);soft_done.append(done);timeouts.append(timeout)
            active=active&~timeout
        last_priv=st.privileged(s,a_prev,self.commands.command(cmd_state),xi,relief(pose,self.field))
        features.append(critic.features(last_priv,branch));features=torch.stack(features,1)
        rewards=torch.stack(rewards,1);soft_done=torch.stack(soft_done,1);timeouts=torch.stack(timeouts,1)
        objective,surv,immediate,bootstrap=time_limit_return(rewards,soft_done,timeouts,critic(features),cfg.gamma,valid_start)
        return objective,{'rewards':rewards.detach(),'surv':surv.detach(),'soft_done':soft_done.detach(),
                         'timeout':timeouts,'critic_valid':torch.stack(valid,1),'critic_in':features.detach(),
                         'immediate':immediate.detach(),'bootstrap':bootstrap.detach()}

    @torch.no_grad()
    def targets(self,critic,data):
        cfg=self.cfg
        return time_limit_lambda_targets(data['rewards'],data['soft_done'],data['timeout'],
                                         critic(data['critic_in']),cfg.gamma,cfg.lam)

```

### learner.py · SHA256 a6eca17f8960664b1cbd83b593df27caaa7904482f34b95ea2aeda3a6b8f87f3

```python
"""Three-phase source loop plus explicit CTS update; no old SMCTS imports."""
import copy,json,math,os,time,subprocess,sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
import torch
from . import UPSTREAM_COMMIT,state as st
from .networks import Policy,Critic,StudentExport
from .world_model import WorldModel
from .buffer import Replay,Archive,mixed_windows
from .imagine import Imagination
from .terrain import relief

@contextmanager
def frozen(*modules):
    params=[p for m in modules for p in m.parameters()];flags=[p.requires_grad for p in params]
    for p in params:p.requires_grad_(False);p.grad=None
    try:yield
    finally:
        for p,flag in zip(params,flags):p.requires_grad_(flag)

def clip_grad(params,global_norm=1.,elem=float('inf')):
    params=[p for p in params if p.grad is not None]
    norm=torch.nn.utils.clip_grad_norm_(params,global_norm)
    for p in params:p.grad.clamp_(-elem,elem)
    return norm

def scalar_finite(x):return bool(torch.isfinite(x).all())

def provider_assignment(n,driver_frac,expert_frac,teacher_fraction,device):
    """Spread each provider across contiguous terrain columns, with exact counts."""
    source=torch.zeros(n,dtype=torch.uint8,device=device)
    nd,ne=int(n*driver_frac),int(n*expert_frac)
    if nd+ne:
        ids=torch.linspace(0,n-1,nd+ne,device=device).round().long()
        # Interleave the two external providers before choosing student indices.
        rank=torch.arange(nd+ne,device=device)
        driver=(torch.floor((rank+1)*nd/(nd+ne))-torch.floor(rank*nd/(nd+ne))).bool()
        source[ids]=torch.where(driver,2,3).to(torch.uint8)
    own=(source==0).nonzero().flatten();ns=len(own)-int(len(own)*teacher_fraction)
    source[own[torch.linspace(0,len(own)-1,ns,device=device).round().long()]]=1
    return source

def branch_loss(objectives,branch,sizes):
    """Exact sum of branch means; microbatches use the full branch denominator."""
    if isinstance(branch,tuple):
        return -objectives[:sizes[0]].sum()/sizes[0]-objectives[sizes[0]:].sum()/sizes[1]
    return -objectives.sum()/sizes[branch]

class Learner:
    def __init__(self,cfg,env,output,driver=None,expert=None):
        self.cfg,self.env,self.output=cfg,env,Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.device=cfg.device;torch.manual_seed(cfg.seed)
        self.policy=Policy(cfg.actor_hidden).to(cfg.device);self.critic=Critic(cfg.critic_hidden).to(cfg.device)
        self.wm=WorldModel(cfg.wm_hidden).to(cfg.device)
        self.actor_params=list(self.policy.teacher_encoder.parameters())+list(self.policy.actor.parameters())
        self.opt_wm=torch.optim.Adam(self.wm.parameters(),lr=cfg.lr)
        self.opt_actor=torch.optim.Adam(self.actor_params,lr=cfg.lr,betas=(.9,.95))
        self.opt_critic=torch.optim.Adam(self.critic.parameters(),lr=cfg.lr)
        self.opt_cts=torch.optim.Adam(self.policy.student_encoder.parameters(),lr=cfg.cts_lr)
        self.buf=Replay(cfg.num_envs,cfg.buffer_len,cfg.device)
        self.archive=Archive(cfg.archive_size,cfg.wm_window,cfg.device)
        self.imag=Imagination(cfg,env.consts,env.field,env.command_sampler)
        self.driver,self.expert=driver,expert
        if cfg.driver_frac and driver is None:raise ValueError('External driver not loaded')
        if cfg.expert_frac and expert is None:raise ValueError('Second expert not loaded')
        self.index=torch.arange(cfg.num_envs,device=cfg.device)
        self.source=provider_assignment(cfg.num_envs,cfg.driver_frac,cfg.expert_frac,cfg.teacher_fraction,cfg.device)
        self.ou=torch.zeros((cfg.num_envs,12),device=cfg.device)
        self.it=0;self.lr_scale=1.;self.ramp_start=cfg.warmup_iterations;self.actor_frozen_until=0
        self.last_good=None;self.collapse_strikes=0;self.nonfinite=0
        self.counts=dict(collect=0,transitions=0,wm=0,actor=0,critic=0,cts=0)
        self.metrics=[];self.start_time=time.monotonic()
        self.provider_counts=[int((self.source==i).sum()) for i in range(4)]
        self.env.own_mask=self.source<2
        (self.output/'resolved_config.json').write_text(json.dumps(asdict(cfg),indent=2))
        (self.output/'run_identity.json').write_text(json.dumps({'upstream_commit':UPSTREAM_COMMIT,'physics':'isaacsim_physx','providers':self.provider_counts,'imaginary_branch_samples':[int(cfg.imag_batch*cfg.teacher_fraction),cfg.imag_batch-int(cfg.imag_batch*cfg.teacher_fraction)],'branch_objective_weights':[1.,1.],'branch_loss':'teacher_loss.mean() + student_loss.mean()','replay_tensor_bytes':self.buf.storage_bytes,'expert_missing':not bool(expert),'init':'cold','curriculum_clock':'learner iteration; one real vector step per iteration; not PPO24-step-equivalent'},indent=2))

    def learning_rates(self):
        cfg=self.cfg;t=min(1.,self.it/cfg.lr_decay_iterations)
        lr=cfg.lr*(cfg.lr_final_frac+(1-cfg.lr_final_frac)*.5*(1+math.cos(math.pi*t)))*self.lr_scale
        ramp=min(1.,max(0.,self.it-max(self.ramp_start,self.actor_frozen_until,cfg.warmup_iterations))/max(1,cfg.actor_ramp_iterations))
        for opt,scale in [(self.opt_wm,1),(self.opt_actor,ramp*cfg.actor_lr_scale),(self.opt_critic,1)]:
            for g in opt.param_groups:g['lr']=lr*scale
        return {'wm':lr,'actor':lr*ramp*cfg.actor_lr_scale,'critic':lr,'cts':self.cfg.cts_lr,'actor_ramp':ramp}

    @torch.no_grad()
    def collect(self):
        cfg=self.cfg;before=self.env.snapshot()
        priv=st.privileged(before['s'],before['a_prev'],before['command'],self.env.xi,before['relief'])
        a=torch.empty((cfg.num_envs,12),device=self.device)
        for branch in (0,1):
            ids=self.source==branch
            if ids.any():a[ids]=self.policy(before['obs'][ids],priv[ids],branch)
        if self.driver is not None:
            external=self.driver.act(before['command']);a[self.source==2]=external[self.source==2]
        if self.expert is not None:
            external=self.expert(before['obs']);a[self.source==3]=external[self.source==3]
        sigma=cfg.explore_noise*math.sqrt(2*cfg.explore_theta)
        self.ou=self.ou*(1-cfg.explore_theta*cfg.dt)+sigma*math.sqrt(cfg.dt)*torch.randn_like(a)
        noise=torch.where((self.index<int(cfg.num_envs*cfg.explore_frac))[:,None],self.ou,cfg.action_noise*torch.randn_like(a))
        # Pinned train.py adds exploration after provider selection, to all
        # executed actions. Replay records those actual perturbed actions.
        a=a+noise
        if not scalar_finite(a):raise FloatingPointError('Nonfinite real action')
        xi=self.env.xi.clone()
        after,done,metrics=self.env.step(a)
        self.buf.add(s=before['s'],pose=before['pose'],a=a,a_prev=before['a_prev'],obs=before['obs'],
                     command=before['command'],timer=before['timer'],cmd_state=before['cmd_state'],done=done,relief=before['relief'],
                     source=self.source,episode=before['episode'],xi=xi)
        keep=~done
        if keep.any():self.wm.update_stats(before['s'][keep],a[keep],before['a_prev'][keep],xi[keep],before['relief'][keep],after['s'][keep],st.pose_delta(before['pose'][keep],after['pose'][keep]))
        self.policy.update_stats(before['obs'],priv)
        self.ou[done]=0
        if self.driver is not None:self.driver.reset(done)
        self.counts['collect']+=1;self.counts['transitions']+=cfg.num_envs
        metrics['source_counts']=self.provider_counts;self.metrics.append(metrics)

    def fit_world_model(self):
        losses=[]
        for _ in range(self.cfg.wm_updates):
            w=mixed_windows(self.buf,self.archive,self.cfg,self.env.xi)
            dp=torch.stack([st.pose_delta(w['pose'][:,j],w['pose'][:,j+1]) for j in range(self.cfg.wm_window)],1)
            loss=self.wm.loss(w['s'][:,0],w['a'],w['a_prev'],w['xi'],w['relief'].float(),w['s'][:,1:],dp,w['valid'])
            self.opt_wm.zero_grad(set_to_none=True);loss.backward();gn=clip_grad(self.wm.parameters(),self.cfg.grad_clip)
            if scalar_finite(gn) and scalar_finite(loss):self.opt_wm.step();self.counts['wm']+=1
            else:self.nonfinite+=1
            losses.append(float(loss.detach()))
        return sum(losses)/len(losses)

    def improve_policy(self):
        cfg=self.cfg;nt=int(cfg.imag_batch*cfg.teacher_fraction);sizes=(nt,cfg.imag_batch-nt)
        all_data=[];objective_means=[];source_counts=torch.zeros(4,device=self.device)
        self.opt_actor.zero_grad(set_to_none=True)
        with frozen(self.wm,self.critic,self.policy.student_encoder):
            groups=[((nt,cfg.imag_batch-nt),cfg.imag_batch)] if cfg.actor_microbatch>=cfg.imag_batch else list(enumerate(sizes))
            for branch,batch in groups:
                seed=self.buf.seeds(batch,self.env.xi)
                source_counts+=torch.bincount(seed['source'][:,0].long(),minlength=4)
                totals=[]
                for lo in range(0,batch,cfg.actor_microbatch):
                    sub={k:v[lo:lo+cfg.actor_microbatch] for k,v in seed.items()}
                    obj,data=self.imag.rollout(self.wm,self.policy,self.critic,sub,branch,iteration=self.it)
                    branch_loss(obj,branch,sizes).backward();all_data.append(data);totals.append(obj.detach().sum())
                if isinstance(branch,tuple):objective_means=[float(obj[:nt].mean().detach()),float(obj[nt:].mean().detach())]
                else:objective_means.append(float(torch.stack(totals).sum()/batch))
            gn=clip_grad(self.actor_params,cfg.grad_clip,cfg.grad_clip_elem)
            finite=scalar_finite(gn) and all(math.isfinite(v) for v in objective_means)
            if self.it>self.actor_frozen_until and finite:self.opt_actor.step();self.counts['actor']+=1
            elif not finite:self.nonfinite+=1
        data={k:torch.cat([d[k] for d in all_data]) for k in all_data[0]}
        target=self.imag.targets(self.critic,data)
        x=data['critic_in'][:,:-1].reshape(-1,st.PRIV_DIM+2);y=target.reshape(-1)
        if 'critic_valid' in data:
            valid=data['critic_valid'].reshape(-1).bool();x=x[valid];y=y[valid]
        if len(x)<cfg.critic_minibatches:raise RuntimeError('Insufficient valid pre-timeout critic samples')
        self.critic.norm.update(x)
        losses=[]
        for _ in range(cfg.critic_epochs):
            order=torch.randperm(len(x),device=self.device)
            for ids in torch.tensor_split(order,cfg.critic_minibatches):
                loss=(self.critic(x[ids])-y[ids]).square().mean()
                self.opt_critic.zero_grad(set_to_none=True);loss.backward();cgn=clip_grad(self.critic.parameters(),cfg.grad_clip)
                if scalar_finite(cgn) and scalar_finite(loss):self.opt_critic.step();self.counts['critic']+=1
                else:self.nonfinite+=1
                losses.append(float(loss.detach()))
        return {'branch_objectives':objective_means,'objective':sum(objective_means),
                'actor_grad_norm':float(gn),'critic_loss':sum(losses)/len(losses),'imagined_sources':source_counts.tolist(),
                'immediate':float(data['immediate'].mean()),'bootstrap':float(data['bootstrap'].mean())}

    def update_cts(self):
        cfg=self.cfg;losses=[]
        with frozen(self.wm,self.critic,self.policy.actor,self.policy.teacher_encoder):
            for _ in range(cfg.cts_updates):
                seed=self.buf.seeds(cfg.cts_batch,self.env.xi,source=1)
                priv=st.privileged(seed['s'],seed['a_prev'],seed['command'],seed['xi'],seed['relief'])
                with torch.no_grad():label=self.policy.teacher_encoder(self.policy.teacher_norm(priv))
                z,gate=self.policy.student_encoder(self.policy.history_norm(seed['obs']))
                latent=(z-label).square().mean();balance=(gate.mean(0)-1/8).square().mean()
                loss=latent+cfg.cts_balance*balance
                self.opt_cts.zero_grad(set_to_none=True);loss.backward();gn=clip_grad(self.policy.student_encoder.parameters(),cfg.grad_clip)
                if scalar_finite(gn) and scalar_finite(loss):self.opt_cts.step();self.counts['cts']+=1
                else:self.nonfinite+=1
                losses.append(float(loss.detach()))
        return sum(losses)/len(losses)

    def save(self,path=None):
        path=Path(path) if path else self.output/f'model_{self.it:08d}.pt'
        value={'schema':'physx-cts-source-port-v1','upstream_commit':UPSTREAM_COMMIT,'cfg':asdict(self.cfg),
               'policy':self.policy.state_dict(),'critic':self.critic.state_dict(),'world_model':self.wm.state_dict(),
               'optimizers':{k:getattr(self,'opt_'+k).state_dict() for k in ('wm','actor','critic','cts')},
               'iteration':self.it,'counts':self.counts,'lr_scale':self.lr_scale,'ramp_start':self.ramp_start,
               'curriculum':self.env.curriculum_state_dict(),
               'actor_frozen_until':self.actor_frozen_until,'rng':torch.get_rng_state(),
               'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
               'resume_mode':'fresh_simulator_episodes_and_replay; model/optimizer/normalizer_resume',
               'physics':'isaacsim_physx','consts':{k:v.detach().cpu() for k,v in vars(self.env.consts).items()}}
        partial=path.with_suffix('.partial');torch.save(value,partial);os.replace(partial,path)
        return path

    def load(self,path,continue_run=False):
        ck=torch.load(path,map_location=self.device,weights_only=False)
        if ck['schema']!='physx-cts-source-port-v1' or ck['physics']!='isaacsim_physx':raise ValueError('Old/Newton checkpoint forbidden')
        if ck['cfg'].get('robot','go2')!=self.cfg.robot:raise ValueError('Cross-robot checkpoint resume forbidden')
        for name,expected in ck['consts'].items():
            actual=getattr(self.env.consts,name)
            if not torch.allclose(actual,expected.to(actual),atol=1e-6,rtol=1e-6):
                raise ValueError('Checkpoint control constant differs from loaded PhysX: '+name)
        self.policy.load_state_dict(ck['policy']);self.wm.load_state_dict(ck['world_model'])
        reset_critic=self.cfg.reset_critic and not continue_run
        if not reset_critic:self.critic.load_state_dict(ck['critic'])
        for k in ('wm','actor','critic','cts'):
            if k!='critic' or not reset_critic:getattr(self,'opt_'+k).load_state_dict(ck['optimizers'][k])
        if reset_critic:self.actor_frozen_until=self.cfg.warmup_iterations+self.cfg.critic_warmup_iterations
        if continue_run:
            self.it=ck['iteration'];self.counts=ck['counts'];self.lr_scale=ck['lr_scale'];self.ramp_start=self.it
            self.actor_frozen_until=self.it+self.cfg.warmup_iterations
            # Retired evaluator metadata in immutable checkpoints is ignored.
            torch.set_rng_state(ck['rng'].cpu())
            if ck['cuda_rng']:torch.cuda.set_rng_state_all([v.cpu() for v in ck['cuda_rng']])
            if 'curriculum' in ck:self.env.load_curriculum_state_dict(ck['curriculum'])
            self.env.iteration=self.it
            self.env.steps.zero_()
            self.env.reset(self.env.ids);self.env._snapshot=self.env._observe()
        identity_path=self.output/'run_identity.json'
        identity=json.loads(identity_path.read_text())
        identity.update(init='continue_run' if continue_run else 'new_stage_resume',parent_checkpoint=str(path),parent_iteration=ck['iteration'],reset_critic=reset_critic,actor_frozen_until=self.actor_frozen_until)
        identity_path.write_text(json.dumps(identity,indent=2))
        # A reward-stage change needs a compatible critic in the rollback snapshot.
        self.last_good=self.save(self.output/'stage_start.pt') if not continue_run else Path(path)

    def rollback(self,path=None):
        path=path or self.last_good
        if path is None:return False
        ck=torch.load(path,map_location=self.device,weights_only=False)
        self.policy.load_state_dict(ck['policy']);self.critic.load_state_dict(ck['critic'])
        self.lr_scale*=.5;self.ramp_start=self.it;self.collapse_strikes=0
        # Keep the newly trained WM/replay/archive, discard diverging policy moments.
        self.opt_actor=torch.optim.Adam(self.actor_params,lr=self.cfg.lr,betas=(.9,.95))
        self.opt_critic=torch.optim.Adam(self.critic.parameters(),lr=self.cfg.lr)
        self.opt_cts=torch.optim.Adam(self.policy.student_encoder.parameters(),lr=self.cfg.cts_lr)
        return True

    def log(self,stats):
        falls=sum(m['falls'] for m in self.metrics)/max(1,len(self.metrics))
        own=max(1,self.provider_counts[0]+self.provider_counts[1])
        falls_per_min=falls/own*60/self.cfg.dt
        result={'iteration':self.it,'elapsed_seconds':time.monotonic()-self.start_time,'counts':self.counts.copy(),
                'metrics':{k:sum(m[k] for m in self.metrics)/len(self.metrics) for k in ('height','xy_rmse','yaw_rmse','reward','falls')},
                'falls_per_env_min':falls_per_min,'archive_size':self.archive.size,'nonfinite_skipped':self.nonfinite,**stats}
        result['terrain_levels']=self.env.terrain_level_statistics
        from .task import reward_weights
        from .commands import RANGE_ITERATIONS,RANGE_STAGES
        result['reward_weights']=reward_weights(self.it,self.cfg.stage)
        result['command_ranges']=RANGE_STAGES[sum(self.it>=v for v in RANGE_ITERATIONS)]
        falls_all=sum(m['falls_all'] for m in self.metrics)/max(1,len(self.metrics)) if self.metrics else falls
        self.metrics=[]
        result['falls_per_vector_step_all_providers']=falls_all
        if self.it>max(self.cfg.warmup_iterations,self.actor_frozen_until) and falls_all>self.cfg.collapse_falls:self.collapse_strikes+=1
        else:self.collapse_strikes=0
        result['rolled_back']=self.rollback() if self.collapse_strikes>=2 else False
        with (self.output/'training.jsonl').open('a') as f:f.write(json.dumps(result,allow_nan=False)+'\n')
        (self.output/'progress.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        print(json.dumps(result),flush=True)

    def run(self,iterations=None):
        self.env.iteration=self.it
        while self.buf.size<=self.cfg.wm_window+1:self.collect()
        end=self.cfg.iterations if iterations is None else self.it+iterations
        while self.it<end:
            if getattr(self,'stop_requested',False):
                self.save();break
            self.it+=1;stats={'lr':self.learning_rates()}
            self.env.iteration=self.it
            self.collect();stats['wm_loss']=self.fit_world_model()
            if self.it>self.cfg.warmup_iterations:stats.update(self.improve_policy())
            stats['cts_loss']=self.update_cts()
            if self.it==1 or self.it%self.cfg.log_interval==0 or self.it==end:self.log(stats)
            if self.it==1 or self.it%self.cfg.save_interval==0 or self.it==end:
                path=self.save()
                if self.it>self.cfg.warmup_iterations and not self.collapse_strikes:self.last_good=path
            if self.cfg.evaluation_python and self.it%self.cfg.evaluation_interval==0:
                path=self.save();self.evaluate_feedback(path)
                if getattr(self,'stop_requested',False):break
            if self.it==end and iterations is None:
                (self.output/'needs_review.json').write_text(json.dumps({'iteration':self.it,'reason':'Iteration review reached; native and official transfer evidence required before extending budget','B0':False}))
        return self.last_good

    def evaluate_feedback(self,path):
        from .robogauge_feedback import evaluate_checkpoint
        from .diagnostics import native_diagnostics,model_diagnostics
        out=self.output/f'robogauge_{self.it:08d}'
        out.mkdir(parents=True,exist_ok=True)
        jobs={'native':lambda:native_diagnostics(self),'model':lambda:model_diagnostics(self),
              'transfer':lambda:evaluate_checkpoint(path,out,self.it,self.cfg.evaluation_python)}
        for name,job in jobs.items():
            try:
                result=job()
                if name!='transfer':(out/(name+'.json')).write_text(json.dumps(result,indent=2))
            except Exception as error:
                # A diagnostic failure is actionable, but is not measured policy
                # collapse. Source training-collapse logic remains in log().
                record={'error':str(error),'iteration':self.it,'training_continues':True}
                (out/(name+'_error.json')).write_text(json.dumps(record,indent=2))
                print(json.dumps({'diagnostic_error':name,**record}),flush=True)
        (self.output/'latest_diagnostics.json').write_text(json.dumps({
            'iteration':self.it,'native':str(out/'native.json'),'transfer':str(out/'result.json'),
            'curriculum':'official iteration schedule','automatic_transfer_rollback':False,'B0':False},indent=2))

    def export(self,path):
        module=copy.deepcopy(StudentExport(self.policy)).eval().cpu()
        traced=torch.jit.trace(module,torch.zeros(1,450),check_trace=True);traced.save(str(path))
        return traced

```

### task.py · SHA256 7d3e30637056afe0adbe87cc0ba5b4c14f3df5c3ed38c594c62d1529380bb486

```python
"""Validated13-term task plus two explicitly imported Go2RLLab reward courses.

This is not the complete official15-term reward. Control constants and the
existing13 terms retain the validated PhysX baseline definitions and weights.
"""
import math
from dataclasses import dataclass
import torch
from . import state as st
from .terrain import ground

@dataclass
class RobotConsts:
    default_q: torch.Tensor
    action_scale: torch.Tensor
    kp: torch.Tensor
    kd: torch.Tensor
    effort: torch.Tensor
    soft_limits: torch.Tensor

WEIGHTS={'track_xy':2.,'track_yaw':1.,'upright':1.,'tilt_rate_l2':-.05,'hard_landing':-.5,
         'crouch':-30.,'joint_deviation_l2':-.1,'base_height_low':-10.,'terminated':-200.,
         'action_rate_l2':-.05,'dof_pos_limits':-1.,'joint_torques_l2':-2e-5,'joint_acc_l2':-2.5e-7}

OFFICIAL_REWARD_SOURCE='wty-yy/go2_rl_lab@6c95d9feef4814c25eb6536e61bb38e1d1d718d4'
BASE_HEIGHT_TARGET=.38
BASE_HEIGHT_CROP_X=(-.2,-.1,0.,.1,.2)
BASE_HEIGHT_CROP_Y=(-.1,0.,.1)

def reward_weights(iteration,stage='flat'):
    """Official linear courses on an explicit caller-owned curriculum clock.

    The original source uses common_step_counter//24. This approved port uses
    the SuperMimic logical learner iteration (one collection step per update),
    so it is not equivalent in physical sample count to a PPO24-step iteration.
    The caller passes current iteration, never a replay record's old timestamp.
    Scalar tensors keep this path differentiable/compilable without host reads.
    """
    if torch.is_tensor(iteration):
        vz_progress=(iteration/1500.).clamp(0.,1.)
        height_progress=(iteration/5000.).clamp(0.,1.)
    else:
        vz_progress=min(1.,max(0.,float(iteration)/1500.))
        height_progress=min(1.,max(0.,float(iteration)/5000.))
    return dict(WEIGHTS,terminated=-600. if stage=='terrain' else -200.,
                lin_vel_z_l2=-2.+2.*vz_progress,
                base_height_l2=-1.-9.*height_progress)

def crop_height_from_map(s,pose,field=None):
    """Known-terrain15-ray crop matching the official0.1m yaw-aligned grid.

    Real physics supplies measured mesh-ray crop height directly to reward().
    Imagination queries the mesh-derived height map at its predicted pose.
    These share the same height definition; map/raycast error remains measurable.
    """
    xs,ys=torch.meshgrid(pose.new_tensor(BASE_HEIGHT_CROP_X),pose.new_tensor(BASE_HEIGHT_CROP_Y),indexing='ij')
    x,y=xs.flatten(),ys.flatten();c,sn=pose[:,2:3].cos(),pose[:,2:3].sin()
    xy=torch.stack((pose[:,0:1]+c*x-sn*y,pose[:,1:2]+sn*x+c*y),-1)
    base_world_z=s[:,0]+ground(pose[:,:2],field)
    return base_world_z-ground(xy,field).mean(-1)

def fall_probability(s):
    tilt=torch.acos((-s[:,3]).clamp(-1+1e-4,1-1e-4))
    return torch.sigmoid((tilt-math.radians(70))/math.radians(5))

def fell(s):return -s[:,3]<math.cos(math.radians(70))

def reward(s0,s1,a,previous,command,c,stage='flat',soft=False,*,iteration=0,pose=None,field=None,
           height_above_crop=None,base_height_target=BASE_HEIGHT_TARGET):
    q=s1[:,st.Q]+c.default_q;lim=c.soft_limits
    tau=(c.kp*(a*c.action_scale+c.default_q-q)-c.kd*s1[:,st.QD]).clamp(-c.effort,c.effort)
    g=s1[:,st.GRAV]
    d=fall_probability(s1) if soft else fell(s1).float()
    velocity_body=st.velocity_body(s1)
    if height_above_crop is None:
        if stage=='terrain' and field is None:
            raise ValueError('Terrain reward requires measured crop height or a known terrain height map.')
        if pose is None:
            if field is not None or stage=='terrain':
                raise ValueError('Terrain reward requires measured crop height or predicted pose/height map.')
            crop_height=s1[:,0]
        else:
            crop_height=crop_height_from_map(s1,pose,field)
    else:
        crop_height=height_above_crop
    if crop_height.shape!=s1.shape[:1]:
        raise ValueError('Crop height must have one scalar per environment.')
    terms={'track_xy':torch.exp(-(velocity_body[:,:2]-command[:,:2]).square().sum(-1)/.25),
           'track_yaw':torch.exp(-(s1[:,9]-command[:,2]).square()/.25),
           'upright':torch.exp(-g[:,:2].square().sum(-1)/.16),
           'tilt_rate_l2':s1[:,7:9].square().sum(-1),
           'hard_landing':(-s1[:,6]).clamp(min=0).square()*(st.feet_world_z(s1).min(-1).values<.05).float(),
           'crouch':(.23-s1[:,0]).clamp(min=0),
           'joint_deviation_l2':s1[:,st.Q].square().sum(-1),
           'base_height_low':(.18-s1[:,0]).clamp(min=0),
           'terminated':d,'action_rate_l2':(a-previous).square().sum(-1),
           'dof_pos_limits':((lim[...,0]-q).clamp(min=0)+(q-lim[...,1]).clamp(min=0)).sum(-1),
           'joint_torques_l2':tau.square().sum(-1),
           'joint_acc_l2':((s1[:,st.QD]-s0[:,st.QD])/.02).square().sum(-1),
           'lin_vel_z_l2':velocity_body[:,2].square(),
           'base_height_l2':(crop_height-base_height_target).square()}
    weights=reward_weights(iteration,stage)
    return sum(weights[k]*v for k,v in terms.items())*.02,terms,d

```

### commands.py · SHA256 2f0430650082a93cf46fa2b700427b3a0a73f2591579f23825f9479d4d5d4267

```python
"""Functional port of Go2RLLab's pinned mixed-task command sampler.

Source: wty-yy/go2_rl_lab 6c95d9feef4814c25eb6536e61bb38e1d1d718d4,
cmd_utils/rl_gym_cmd.py and tasks/mixed/env_cfg.py:CommandsCfg.
The caller supplies learner iterations (num_steps_per_iter=1), not physics
substeps. This module contains no capability gates or grid curriculum.

All command memory is a packed floating tensor suitable for replay. Random
choices may be supplied explicitly; no state tensor is mutated or detached.
"""
from itertools import product
import math
import torch

SOURCE_COMMIT = '6c95d9feef4814c25eb6536e61bb38e1d1d718d4'
CMD_STATE_DIM = 17
REQUEST, COMMAND = slice(0, 3), slice(3, 6)
TIMER, ACCUMULATION, EPISODE_ELAPSED = 6, slice(7, 9), 9
LAST_LIMIT, COMMAND_COUNTER, MAX_DISTANCE = 10, 11, 12
ORIGIN, RANGE_STAGE, LIMIT_STAGE = slice(13, 15), 15, 16
COMMAND_NAMES = ('lin_vel_x', 'lin_vel_y', 'ang_vel_yaw')
RANGE_ITERATIONS = (5000, 15000, 25000, 35000, 45000)
RANGE_STAGES = (
    ((-.5, .5), (-.5, .5), (-1., 1.)),
    ((-1., 1.), (-1., 1.), (-2., 2.)),
    ((-2., 2.), (-1., 1.), (-3., 3.)),
    ((-3., 3.), (-1., 1.), (-3., 3.)),
    ((-4., 4.), (-1., 1.), (-3., 3.)),
    ((-4.5, 4.2), (-1., 1.), (-3., 3.)),
)
TERRAIN_RANGES = {
    'flat': ((-4.5, 4.2), (-1., 1.), (-3., 3.)),
    'wave': ((-1.5, 1.5), (-1., 1.), (-1.5, 1.5)),
    'slope_up': ((-1.5, 1.5), (-1., 1.), (-1.5, 1.5)),
    'slope_down': ((-1.5, 1.5), (-1., 1.), (-1.5, 1.5)),
    'rough_slope': ((-1.5, 1.5), (-1., 1.), (-1.5, 1.5)),
    'stairs_up': ((-1., 1.), (-1., 1.), (-1.5, 1.5)),
    'stairs_down': ((-1., 1.), (-1., 1.), (-1.5, 1.5)),
    'obstacles': ((-1., 1.), (-1., 1.), (-1.5, 1.5)),
    'wheels': ((-2., 1.5), (-1., 1.), (-1.5, 1.5)),
    'small_squares': ((-1., 1.), (-1., 1.), (-1.5, 1.5)),
}


def zero_command_prob(iteration):
    """Official linear zero-command ramp: 0 at iteration0, .1 at1500."""
    if isinstance(iteration, torch.Tensor):
        return (iteration / 1500.).clamp(0., 1.) * .1
    return min(max(float(iteration) / 1500., 0.), 1.) * .1


def coupled_clip(command):
    """Apply official priority rules after terrain clipping, without in-place ops."""
    x, y, yaw = command.unbind(-1)
    fast = x.abs() > 2.
    y = torch.where(fast, y.clamp(-.5, .5), y)
    yaw = torch.where(fast, yaw.clamp(-1., 1.), yaw)
    turning = yaw.abs() > 2.
    x = torch.where(turning, x.clamp(-2., 2.), x)
    y = torch.where(turning, y.clamp(-1., 1.), y)
    return torch.stack((x, y, yaw), -1)


class GridTerrainLookup:
    """Official half-open rectangular grid lookup; outside IDs are -1.

    Metadata axes match Isaac terrain grids: rows along x, columns along y.
    ``origin_xy`` is the lower grid corner, excluding the surrounding border;
    ``column_ids`` maps each column to an index in ``names``. The PhysX mesh
    adapter must supply these from its actual generator, not infer from robots.
    """
    def __init__(self, origin_xy, tile_size, rows, cols, column_ids, names, device='cpu'):
        self.origin_xy = torch.as_tensor(origin_xy, device=device, dtype=torch.float32)
        self.tile_size = torch.as_tensor(tile_size, device=device, dtype=torch.float32)
        self.rows, self.cols, self.names = int(rows), int(cols), tuple(names)
        self.column_ids = torch.as_tensor(column_ids, device=device, dtype=torch.long)
        if (self.column_ids.shape != (self.cols,) or self.origin_xy.shape != (2,)
                or self.tile_size.shape != (2,) or min(self.rows, self.cols) < 1
                or not bool((self.tile_size > 0).all())
                or not bool(((self.column_ids >= 0) & (self.column_ids < len(self.names))).all())):
            raise ValueError('Invalid terrain grid metadata.')

    def query(self, position_xy):
        cell = torch.floor((position_xy[..., :2] - self.origin_xy.to(position_xy))
                           / self.tile_size.to(position_xy)).long()
        valid = ((cell[..., 0] >= 0) & (cell[..., 0] < self.rows)
                 & (cell[..., 1] >= 0) & (cell[..., 1] < self.cols))
        ids = self.column_ids.to(position_xy.device)[cell[..., 1].clamp(0, self.cols - 1)]
        return torch.where(valid, ids, torch.full_like(ids, -1))


class OfficialCommands:
    """One functional command transition for both real and imagined rollouts.

    ``initial``/``reset`` resample without advancing episode time. ``advance``
    consumes one dt, resamples expired timers, clips requests at the current
    terrain, and updates displacement. Pass the post-physics XY position.
    ``random`` is optional [N,8] independent U[0,1) values; unused draws keep
    shapes static. Compared with upstream this changes RNG consumption, not
    sampling probabilities. The caller checkpoints its generator/global RNG.
    """
    resampling_time = 5.0
    command_area_length = 8.0
    num_steps_per_iter = 1
    limit_vel_prob = .2
    limit_vel_hold_prob = 0.
    limit_ang_vel_at_zero_command_prob = .2
    zero_command_prob = staticmethod(zero_command_prob)

    def __init__(self, dt=.02, episode_seconds=25., terrain_lookup=None):
        self.dt, self.episode_seconds = float(dt), float(episode_seconds)
        if not math.isfinite(self.dt) or not math.isfinite(self.episode_seconds) or self.dt <= 0 or self.episode_seconds <= 0:
            raise ValueError('Positive finite control dt and episode duration required.')
        self.max_episode_steps = math.ceil(self.episode_seconds / self.dt)
        self.terrain_lookup = terrain_lookup
        self.terrain_names = () if terrain_lookup is None else tuple(terrain_lookup.names)
        if set(self.terrain_names) - set(TERRAIN_RANGES):
            raise ValueError('Missing official command ranges for terrain names.')

    @staticmethod
    def command(state):
        return state[:, COMMAND]

    @staticmethod
    def timer(state):
        return state[:, TIMER:TIMER + 1]

    def _ranges(self, state, position_xy, stage):
        global_ranges = state.new_tensor(RANGE_STAGES)[stage.long()]
        if self.terrain_lookup is None:
            return global_ranges, global_ranges
        ids = self.terrain_lookup.query(position_xy)
        table = state.new_tensor([TERRAIN_RANGES['flat']] + [TERRAIN_RANGES[name] for name in self.terrain_names])
        terrain_ranges = table[ids + 1]
        local = torch.clamp(global_ranges, min=terrain_ranges[..., :1], max=terrain_ranges[..., 1:])
        return global_ranges, local

    def initial(self, position_xy, origins_xy, iteration=0, *, generator=None, random=None):
        state = position_xy.new_zeros((len(position_xy), CMD_STATE_DIM))
        return self.reset(state, torch.ones(len(state), device=state.device, dtype=torch.bool),
                          position_xy, origins_xy, iteration, generator=generator, random=random)

    def reset(self, state, mask, position_xy, origins_xy, iteration, *, generator=None, random=None):
        """Reset selected episodes; nonreset rows retain memory and raw requests.

        ``mask`` is [N] bool. A reset triggers the global iteration stage update
        just as the source manager's _resample does. All rows are re-clipped.
        """
        fresh = torch.cat((torch.zeros_like(state[:, :ORIGIN.start]), origins_xy[:, :2],
                           state[:, RANGE_STAGE:]), -1)
        base = torch.where(mask[:, None], fresh, state)
        return self._resample(base, mask, position_xy, iteration, generator=generator, random=random)

    def advance(self, state, position_xy, iteration, *, generator=None, random=None):
        elapsed = state[:, EPISODE_ELAPSED] + self.dt
        timer = state[:, TIMER] - self.dt
        updated = torch.cat((state[:, :TIMER], timer[:, None], state[:, ACCUMULATION],
                             elapsed[:, None], state[:, LAST_LIMIT:]), -1)
        result = self._resample(updated, timer <= 0., position_xy, iteration, generator=generator, random=random)
        distance = torch.linalg.vector_norm(position_xy[:, :2] - result[:, ORIGIN], dim=-1)
        maximum = torch.maximum(result[:, MAX_DISTANCE], distance)
        return torch.cat((result[:, :MAX_DISTANCE], maximum[:, None], result[:, ORIGIN.start:]), -1)

    def _resample(self, state, due, position_xy, iteration, *, generator=None, random=None):
        if state.ndim != 2 or state.shape[1] != CMD_STATE_DIM:
            raise ValueError('Expected packed command state [N,17].')
        # Lazy global curriculum activation matches the official manager:
        # stages change at the first resampling event at/after the iteration.
        clock = torch.as_tensor(iteration, device=state.device)
        current_stage = (clock[..., None] >= clock.new_tensor(RANGE_ITERATIONS)).sum(-1)
        current_limit_stage = (clock >= 25000).long()
        stage = torch.where(due.any(), current_stage, state[:, RANGE_STAGE]).expand(len(state))
        limit_stage = torch.where(due.any(), current_limit_stage, state[:, LIMIT_STAGE]).expand(len(state))
        global_ranges, ranges = self._ranges(state, position_xy, stage)
        u = torch.rand((len(state), 8), device=state.device, dtype=state.dtype, generator=generator) if random is None else random
        if u.shape != (len(state), 8):
            raise ValueError('Command random draws must have shape [N,8].')
        remaining = (.625 * self.command_area_length
                     - torch.linalg.vector_norm(state[:, ACCUMULATION], dim=-1) * self.resampling_time).clamp_min(0.)
        remaining_time = self.max_episode_steps * self.dt - state[:, EPISODE_ELAPSED] + self.dt
        lower = (remaining / (remaining_time + 1e-9)).clamp_min(0.)
        negative = (-lower[:, None] - ranges[:, :2, 0]).relu()
        positive = (ranges[:, :2, 1] - lower[:, None]).relu()
        draw = u[:, :2] * (negative + positive + 1e-6)
        xy = torch.where(draw < negative, ranges[:, :2, 0] + draw,
                         ranges[:, :2, 1] - positive + draw - negative)
        yaw = ranges[:, 2, 0] + u[:, 2] * (ranges[:, 2, 1] - ranges[:, 2, 0])
        sampled = torch.cat((xy, yaw[:, None]), -1)
        limited = u[:, 3] < self.limit_vel_prob
        early_combos = state.new_tensor(tuple(product((-1, 1), (-1, 1), (-1, 0, 1))))
        late_combos = state.new_tensor(tuple(product((-1, 1), (0,), (-1, 0, 1))))
        code = torch.where(limit_stage[:, None].bool(), late_combos[(u[:, 4] * 6).long().clamp_max(5)],
                           early_combos[(u[:, 4] * 12).long().clamp_max(11)])
        boundary = torch.where(code < 0, global_ranges[..., 0], global_ranges[..., 1])
        boundary = torch.where(code == 0, torch.zeros_like(boundary), boundary)
        # Upstream inverts the newly sampled value when the preceding command
        # was limited. It does not invert the preceding saved boundary command.
        limit_value = torch.where(state[:, LAST_LIMIT:LAST_LIMIT + 1].bool(), -sampled, boundary)
        sampled = torch.where(limited[:, None], limit_value, sampled)
        max_lin = global_ranges[:, :2].abs().flatten(1).amax(-1)
        zero_time = (self.episode_seconds - state[:, EPISODE_ELAPSED]
                     - remaining / (.8 * max_lin + 1e-9)).clamp(0., self.resampling_time)
        is_zero = ((u[:, 3] >= self.limit_vel_prob)
                   & (u[:, 3] < self.limit_vel_prob + zero_command_prob(iteration)) & (zero_time > 0.))
        zero_yaw = torch.where(u[:, 6] < .5, ranges[:, 2, 0], ranges[:, 2, 1])
        zero_yaw = torch.where(u[:, 5] < self.limit_ang_vel_at_zero_command_prob, zero_yaw, torch.zeros_like(zero_yaw))
        zero_command = torch.stack((torch.zeros_like(zero_yaw), torch.zeros_like(zero_yaw), zero_yaw), -1)
        sampled = torch.where(is_zero[:, None], zero_command, sampled)
        request = torch.where(due[:, None], sampled, state[:, REQUEST])
        actual = coupled_clip(torch.clamp(request, min=ranges[..., 0], max=ranges[..., 1]))
        new_timer = torch.where(is_zero, zero_time, torch.full_like(zero_time, self.resampling_time))
        timer = torch.where(due, new_timer, state[:, TIMER])
        accumulation = state[:, ACCUMULATION] + torch.where(due[:, None], actual[:, :2], torch.zeros_like(actual[:, :2]))
        last_limit = torch.where(due, limited.to(state.dtype), state[:, LAST_LIMIT])
        counter = state[:, COMMAND_COUNTER] + due.to(state.dtype)
        return torch.cat((request, actual, timer[:, None], accumulation,
                          state[:, EPISODE_ELAPSED:EPISODE_ELAPSED + 1], last_limit[:, None], counter[:, None],
                          state[:, MAX_DISTANCE:ORIGIN.stop], stage[:, None], limit_stage[:, None]), -1)

```

### terrain.py · SHA256 a6de15f8ae3db0fdd12476d6737b83d2eae70a329f15f1a059757b27d8507527

```python
"""Pinned source height-map sampling; PhysX mesh is the authoritative geometry."""
import torch
SCAN_X=tuple(-.42+.12*i for i in range(11))
SCAN_Y=tuple(-.36+.12*j for j in range(7))
SCAN_POINTS=77

class HeightField:
    def __init__(self,heights,origin,res):self.h,self.origin,self.res=heights,origin,res
    def sample(self,xy):
        gx=(xy[...,0]-self.origin[0])/self.res;gy=(xy[...,1]-self.origin[1])/self.res
        H,W=self.h.shape;gx=gx.clamp(0,W-1.001);gy=gy.clamp(0,H-1.001)
        x,y=gx.floor().long(),gy.floor().long();fx,fy=gx-x,gy-y
        return (self.h[y,x]*(1-fx)+self.h[y,x+1]*fx)*(1-fy)+(self.h[y+1,x]*(1-fx)+self.h[y+1,x+1]*fx)*fy

def ground(xy,field):return torch.zeros_like(xy[...,0]) if field is None else field.sample(xy)

def relief(pose,field):
    xs,ys=torch.meshgrid(pose.new_tensor(SCAN_X),pose.new_tensor(SCAN_Y),indexing='ij')
    x,y=xs.flatten(),ys.flatten();c,s=pose[:,2:3].cos(),pose[:,2:3].sin()
    xy=torch.stack((pose[:,0:1]+c*x-s*y,pose[:,1:2]+s*x+c*y),-1)
    return (ground(pose[:,:2],field)[:,None]-ground(xy,field)).clamp(-1.2,1.2)

```

### terrain_scene.py · SHA256 fd4c67e696305adc8b143d50cbf94443cd60ada95b9d57858f40beccf33f06ca

```python
"""Pinned Go2RLLab direct mesh, source distance progression, authoritative raycasts.
Source wty-yy/go2_rl_lab@6c95d9feef4814c25eb6536e61bb38e1d1d718d4.
"""
import hashlib
import numpy as np
import torch
import warp as wp
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.sim import RigidBodyMaterialCfg
from .terrain import HeightField
from .official_terrain_mesh import MIXED_MESH_TERRAIN_CFG

class CapturedTerrain(TerrainImporter):
    def __init__(self,cfg,*args,**kwargs):
        # Pinned custom wheel/square/height functions use global NumPy RNG.
        # Seed only the generation scope and preserve unrelated process RNG.
        previous=np.random.get_state()
        try:
            np.random.seed(int(cfg.terrain_generator.seed))
            super().__init__(cfg,*args,**kwargs)
        finally:
            np.random.set_state(previous)
    def import_mesh(self,name,mesh):
        self.collision_mesh=mesh.copy()
        return super().import_mesh(name,mesh)

def terrain_config(seed=17):
    gen=MIXED_MESH_TERRAIN_CFG.copy();gen.seed=int(seed)
    return TerrainImporterCfg(class_type=CapturedTerrain,prim_path='/World/Ground',
        terrain_type='generator',terrain_generator=gen,max_init_terrain_level=5,
        collision_group=-1,debug_vis=False,
        physics_material=RigidBodyMaterialCfg(friction_combine_mode='average',
            restitution_combine_mode='average',static_friction=1.,dynamic_friction=1.,restitution=0.))

class TerrainPositionLookup:
    """Source-normalized columns and half-open world-space cell lookup."""
    def __init__(self,cfg,device):
        self.names=tuple(cfg.sub_terrains)
        raw=np.asarray([value.proportion for value in cfg.sub_terrains.values()],dtype=np.float64)
        if not np.isfinite(raw).all() or (raw<0).any() or raw.sum()<=0:
            raise ValueError('Invalid source terrain proportions')
        normalized=raw/raw.sum();columns=np.arange(cfg.num_cols)/cfg.num_cols+.001
        ids=np.searchsorted(np.cumsum(normalized),columns,side='right')
        self.column_types=torch.as_tensor(ids,dtype=torch.long,device=device)
        self.size=torch.tensor(cfg.size,dtype=torch.float32,device=device)
        self.shape=torch.tensor([cfg.num_rows,cfg.num_cols],device=device)
        self.minimum_xy=-.5*self.size*self.shape
        self.metadata={'names':list(self.names),'raw_proportions':raw.tolist(),
            'raw_proportion_sum':float(raw.sum()),'normalized_proportions':normalized.tolist(),
            'column_type_ids':ids.tolist(),'column_counts':np.bincount(ids,minlength=len(raw)).tolist(),
            'rows':cfg.num_rows,'columns':cfg.num_cols,'tile_size_m':list(cfg.size),
            'border_width_m':cfg.border_width,'sub_terrain_border_width_m':cfg.sub_terrain_border_width,
            'max_init_terrain_level':5,'default_geometry_backend':'source_direct_mesh','seed':cfg.seed,
            'numpy_geometry_seed_scope':'saved_seeded_generated_restored'}
    def query(self,positions_w):
        cells=torch.floor((positions_w[...,:2]-self.minimum_xy)/self.size).long()
        inside=((cells>=0)&(cells<self.shape)).all(-1)&torch.isfinite(positions_w[...,:2]).all(-1)
        columns=cells[...,1].clamp(0,len(self.column_types)-1)
        return torch.where(inside,self.column_types[columns],-1)
    __call__=query

def terrain_level_decisions(maximum_distance,accumulated_commands,resampling_time,
                            zero_command_probability,tile_length=9.,sub_terrain_border_width=.5):
    """Exact TerrainLevelsVelGym inequalities; no invented iteration level ramp."""
    length=max(0.,tile_length-2.*sub_terrain_border_width)
    move_up=maximum_distance>length/2.
    target_distance=accumulated_commands.norm(dim=1)*resampling_time*(1.-zero_command_probability)
    move_down=(maximum_distance<target_distance*.5)&~move_up
    return move_up,move_down

@wp.kernel
def cast_height(mesh:wp.uint64,points:wp.array(dtype=wp.vec3),height:wp.array(dtype=float),valid:wp.array(dtype=wp.int32)):
    i=wp.tid();query=wp.mesh_query_ray(mesh,points[i],wp.vec3(0.,0.,-1.),100.)
    if query.result:
        height[i]=points[i][2]-query.t
        valid[i]=1

class CollisionHeightRaycaster:
    def __init__(self,mesh,device):
        self.device=device
        vertices=np.ascontiguousarray(mesh.vertices,dtype=np.float32)
        faces=np.ascontiguousarray(mesh.faces,dtype=np.int32)
        self.mesh=wp.Mesh(points=wp.array(vertices,dtype=wp.vec3,device=device),
                          indices=wp.array(faces.flatten(),dtype=int,device=device))
        self.identity={'vertices_sha256':hashlib.sha256(vertices.tobytes()).hexdigest(),
                       'faces_sha256':hashlib.sha256(faces.tobytes()).hexdigest(),
                       'vertex_count':len(vertices),'triangle_count':len(faces)}
    def cast(self,starts):
        shape=starts.shape[:-1];flat=starts.reshape(-1,3).contiguous().to(dtype=torch.float32)
        heights=torch.zeros(flat.shape[0],device=flat.device)
        valid=torch.zeros(flat.shape[0],dtype=torch.int32,device=flat.device)
        wp.launch(cast_height,dim=flat.shape[0],inputs=[self.mesh.id,wp.from_torch(flat,dtype=wp.vec3),
            wp.from_torch(heights),wp.from_torch(valid)],device=self.device)
        return heights.reshape(shape),valid.reshape(shape).bool()
    def height_crop(self,positions,yaw):
        """Exact full17x11 scan, source finite fallback, then5x3 crop."""
        x,y=torch.meshgrid(torch.arange(-8,9,device=positions.device)*.1,
                           torch.arange(-5,6,device=positions.device)*.1,indexing='ij')
        x,y=x.flatten(),y.flatten();c,s=yaw[:,None].cos(),yaw[:,None].sin()
        starts=torch.stack((positions[:,0:1]+c*x-s*y,positions[:,1:2]+s*x+c*y,
                            (positions[:,2:3]+20.).expand(-1,len(x))),-1)
        heights,valid=self.cast(starts)
        fallback=torch.where(valid,heights,torch.full_like(heights,torch.inf)).amin(-1)
        fallback=torch.where(torch.isfinite(fallback),fallback,positions[:,2]+19.)
        heights=torch.where(valid,heights,fallback[:,None])
        crop=(x.abs()<=.200001)&(y.abs()<=.150001)
        return positions[:,2]-heights[:,crop].mean(-1),valid[:,crop].all(-1)

def height_grid_axis(lower,upper,res):
    """Avoid float32 arange spacing drift over the source140x230m world."""
    count=int(np.ceil((upper-lower)/res))+1
    return (float(lower)+np.arange(count,dtype=np.float64)*res).astype(np.float32)

def field_from_mesh(terrain,device,res=.05):
    mesh=terrain.collision_mesh;lo,hi=mesh.bounds
    raycaster=CollisionHeightRaycaster(mesh,device)
    # Build coordinates in float64 before conversion: float32 arange starting at
    # -115m can accumulate centimetres of spacing error across the large grid.
    xs=height_grid_axis(lo[0],hi[0],res);ys=height_grid_axis(lo[1],hi[1],res)
    xs_t=torch.as_tensor(xs,device=device);ys_t=torch.as_tensor(ys,device=device)
    heights=torch.empty((len(ys)*len(xs),),device=device);misses=0
    # Bounded batches avoid a complete140x230m ray-start grid allocation on GPU.
    for first in range(0,heights.numel(),262144):
        index=torch.arange(first,min(first+262144,heights.numel()),device=device)
        x=xs_t[index.remainder(len(xs))];y=ys_t[torch.div(index,len(xs),rounding_mode='floor')]
        starts=torch.stack((x,y,torch.full_like(x,float(hi[2]+2.))),-1)
        h,valid=raycaster.cast(starts)
        interior=(x>lo[0]+res)&(x<hi[0]-res)&(y>lo[1]+res)&(y<hi[1]-res)
        if bool((~valid&interior).any()):raise ValueError('Interior ray misses in authoritative collision mesh')
        misses+=int((~valid).sum());heights[first:first+len(index)]=torch.where(valid,h,0.)
    field=HeightField(heights.reshape(len(ys),len(xs)),(float(lo[0]),float(lo[1])),res)
    field.raycaster=raycaster
    field.metadata=dict(raycaster.identity,resolution_m=res,shape=list(field.h.shape),
                        outer_boundary_misses=misses,collision_bounds=mesh.bounds.tolist())
    return field

```

### official_terrain_heightfield.py · SHA256 bf53c67d0cc6dd74822d02e67b2489a73526fd5d0d9c563c4b8b36adac2c271e

```python
"""Height-field generation backend and shared terrain configuration for the Go2 mixed task."""

from __future__ import annotations

import numpy as np
import trimesh

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.terrains.height_field import hf_terrains
from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh, height_field_to_mesh
from isaaclab.terrains.terrain_generator import TerrainGenerator
from isaaclab.utils.configclass import configclass


MESH_BASE_THICKNESS = 0.10
SPAWN_CENTER_HALF_EXTENT = 1.0
"""Half extent of the centered spawn region used to derive the terrain origin height."""


class PerSubTerrainSlopeThresholdGenerator(TerrainGenerator):
    """Terrain generator with RobotLab's per-sub-terrain slope handling."""

    def __init__(self, cfg, device: str = "cpu"):
        self._apply_sub_terrain_border_width(cfg)
        super().__init__(cfg, device)

    @staticmethod
    def _apply_sub_terrain_border_width(cfg) -> None:
        border_width = getattr(cfg, "sub_terrain_border_width", None)
        if border_width is None:
            return
        for sub_cfg in cfg.sub_terrains.values():
            if hasattr(sub_cfg, "border_width"):
                sub_cfg.border_width = border_width

    def _get_terrain_mesh(self, difficulty, cfg):
        override = getattr(cfg, "slope_threshold_override", None)
        stride = int(getattr(self.cfg, "mesh_simplification_stride", 1))
        stairs_representation = getattr(self.cfg, "stairs_representation", "heightfield")
        if stairs_representation not in ("heightfield", "boxes"):
            raise ValueError(
                "stairs_representation must be either 'heightfield' or 'boxes', "
                f"got {stairs_representation!r}."
            )
        original_slope_threshold = getattr(cfg, "slope_threshold", None)
        if override is not None:
            cfg.slope_threshold = override
        try:
            use_box_stairs = stairs_representation == "boxes" and "stair" in type(cfg).__name__.lower()
            if stride > 1 or use_box_stairs:
                sparse_mesh = self._generate_heightfield_mesh(difficulty, cfg)
            else:
                sparse_mesh = None
            if sparse_mesh is None:
                mesh, origin = super()._get_terrain_mesh(difficulty, cfg)
            else:
                mesh, origin = sparse_mesh
        finally:
            if override is not None:
                cfg.slope_threshold = original_slope_threshold

        if use_box_stairs:
            mesh = self._heightfield_stairs_to_boxes(mesh, cfg)
        elif stride > 1:
            mesh = self._simplify_heightfield_mesh(mesh, cfg, stride)
        return mesh, origin

    @staticmethod
    def _generate_heightfield_mesh(difficulty, cfg):
        """Rebuild a height-field mesh without trimesh vertex deduplication.

        Isaac Lab's ``height_field_to_mesh`` decorator constructs the regular
        grid and then lets ``trimesh`` process it, which merges vertices around
        vertical stair/obstacle faces.  The merged topology is inconvenient for
        structured decimation.  This is the same conversion with
        ``process=False``; all heights, slope correction, borders and origins
        are unchanged.
        """
        horizontal_scale = getattr(cfg, "horizontal_scale", None)
        vertical_scale = getattr(cfg, "vertical_scale", None)
        if horizontal_scale is None or vertical_scale is None or not hasattr(cfg, "border_width"):
            return None

        function = cfg.function
        if hasattr(function, "_resolve"):
            function = function._resolve()
        raw_function = getattr(function, "__wrapped__", function)
        if not callable(raw_function):
            return None

        work_cfg = cfg.copy()
        width_pixels = int(work_cfg.size[0] / horizontal_scale) + 1
        length_pixels = int(work_cfg.size[1] / horizontal_scale) + 1
        border_pixels = int(work_cfg.border_width / horizontal_scale) + 1
        heights = np.zeros((width_pixels, length_pixels), dtype=np.int16)
        sub_terrain_size = [
            (width_pixels - 2 * border_pixels) * horizontal_scale,
            (length_pixels - 2 * border_pixels) * horizontal_scale,
        ]
        terrain_size = work_cfg.size
        work_cfg.size = tuple(sub_terrain_size)
        z_gen, _ = raw_function(difficulty, work_cfg)
        heights[border_pixels:-border_pixels, border_pixels:-border_pixels] = z_gen

        vertices, triangles = convert_height_field_to_mesh(
            heights, horizontal_scale, vertical_scale, work_cfg.slope_threshold
        )
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
        x1 = int((terrain_size[0] * 0.5 - 1) / horizontal_scale)
        x2 = int((terrain_size[0] * 0.5 + 1) / horizontal_scale)
        y1 = int((terrain_size[1] * 0.5 - 1) / horizontal_scale)
        y2 = int((terrain_size[1] * 0.5 + 1) / horizontal_scale)
        origin = np.array(
            [0.5 * terrain_size[0], 0.5 * terrain_size[1], np.max(heights[x1:x2, y1:y2]) * vertical_scale]
        )
        transform = np.eye(4)
        transform[0:2, -1] = -terrain_size[0] * 0.5, -terrain_size[1] * 0.5
        mesh.apply_transform(transform)
        origin += transform[0:3, -1]
        return mesh, origin

    @staticmethod
    def _simplify_heightfield_mesh(mesh: trimesh.Trimesh, cfg, stride: int) -> trimesh.Trimesh:
        """Coarsen regular height-field meshes while retaining sharp terrain edges."""
        horizontal_scale = float(getattr(cfg, "horizontal_scale", 0.0))
        size = getattr(cfg, "size", None)
        if horizontal_scale <= 0.0 or size is None:
            return mesh

        num_rows = int(size[0] / horizontal_scale) + 1
        num_cols = int(size[1] / horizontal_scale) + 1
        if len(mesh.vertices) != num_rows * num_cols:
            # Mesh terrains (for example the flat plane) do not use the
            # height-field layout and are already minimal.
            return mesh

        grid = np.asarray(mesh.vertices).reshape(num_rows, num_cols, 3)
        row_indices = np.arange(0, num_rows, stride, dtype=np.int64)
        col_indices = np.arange(0, num_cols, stride, dtype=np.int64)
        if row_indices[-1] != num_rows - 1:
            row_indices = np.append(row_indices, num_rows - 1)
        if col_indices[-1] != num_cols - 1:
            col_indices = np.append(col_indices, num_cols - 1)

        # Preserve every grid line next to a real height discontinuity for
        # stairs and obstacles.  Continuous waves/slopes remain regular.
        cfg_name = type(cfg).__name__.lower()
        if "stair" not in cfg_name and "obstacle" not in cfg_name:
            # Keep wave, rough-slope and pyramid-slope meshes at their native
            # 10 cm sampling.  Their curvature/noise is part of the policy's
            # observation distribution and is not safely represented by a
            # coarse linear patch without a model-performance revalidation.
            return mesh

        heights = grid[:, :, 2]
        edge_threshold = max(0.04, 2.0 * float(getattr(cfg, "vertical_scale", 0.005)))
        row_edges = np.flatnonzero(np.any(np.abs(np.diff(heights, axis=0)) >= edge_threshold, axis=1))
        col_edges = np.flatnonzero(np.any(np.abs(np.diff(heights, axis=1)) >= edge_threshold, axis=0))
        row_indices = np.unique(np.concatenate((row_indices, row_edges, row_edges + 1)))
        col_indices = np.unique(np.concatenate((col_indices, col_edges, col_edges + 1)))

        coarse = grid[np.ix_(row_indices, col_indices)]
        coarse_rows, coarse_cols = coarse.shape[:2]
        vertices = coarse.reshape(-1, 3)
        cell_rows = np.arange(coarse_rows - 1, dtype=np.int64)[:, None]
        cell_cols = np.arange(coarse_cols - 1, dtype=np.int64)[None, :]
        base = cell_rows * coarse_cols + cell_cols
        next_row = base + coarse_cols
        faces = np.stack(
            (
                np.stack((base, next_row + 1, base + 1), axis=-1),
                np.stack((base, next_row, next_row + 1), axis=-1),
            ),
            axis=2,
        ).reshape(-1, 3)
        # Slope correction can place two neighboring samples at the same
        # 3-D point on a sharp edge.  Such zero-area triangles carry no
        # collision or raycast surface and should not be uploaded.
        face_vertices = vertices[faces]
        cross = np.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0])
        faces = faces[np.linalg.norm(cross, axis=1) > 1.0e-9]
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    @staticmethod
    def _heightfield_stairs_to_boxes(mesh: trimesh.Trimesh, cfg) -> trimesh.Trimesh:
        """Represent the quantized pyramid-stair surface with concentric box rings."""
        horizontal_scale = float(getattr(cfg, "horizontal_scale", 0.0))
        vertical_scale = float(getattr(cfg, "vertical_scale", 0.0))
        size = getattr(cfg, "size", None)
        if horizontal_scale <= 0.0 or vertical_scale <= 0.0 or size is None:
            raise ValueError("Box stairs require positive terrain scales and a configured size.")

        num_rows = int(size[0] / horizontal_scale) + 1
        num_cols = int(size[1] / horizontal_scale) + 1
        if len(mesh.vertices) != num_rows * num_cols:
            raise ValueError("Box stairs require a regular height-field mesh.")

        grid = np.asarray(mesh.vertices).reshape(num_rows, num_cols, 3)
        heights = np.unique(grid[:, :, 2])
        levels = []
        for height in heights:
            level_vertices = grid[np.isclose(grid[:, :, 2], height, atol=1.0e-7)]
            bounds = np.array(
                [
                    np.min(level_vertices[:, 0]),
                    np.max(level_vertices[:, 0]),
                    np.min(level_vertices[:, 1]),
                    np.max(level_vertices[:, 1]),
                ]
            )
            levels.append((float(height), bounds))

        # The generated pyramid consists of nested squares.  Sorting by area
        # reconstructs the visible outer-to-inner stair rings independently of
        # whether the terrain climbs upward or downward toward its center.
        levels.sort(
            key=lambda level: (level[1][1] - level[1][0]) * (level[1][3] - level[1][2]),
            reverse=True,
        )
        tolerance = max(1.0e-6, 0.01 * horizontal_scale)
        for (_, outer), (_, inner) in zip(levels, levels[1:]):
            if not (
                outer[0] < inner[0] - tolerance
                and outer[1] > inner[1] + tolerance
                and outer[2] < inner[2] - tolerance
                and outer[3] > inner[3] + tolerance
            ):
                raise ValueError("Box stairs require strictly nested rectangular height levels.")

        bottom_z = min(height for height, _ in levels) - vertical_scale
        boxes: list[trimesh.Trimesh] = []

        def add_box(x_min: float, x_max: float, y_min: float, y_max: float, top_z: float) -> None:
            if x_max - x_min <= tolerance or y_max - y_min <= tolerance:
                return
            extents = (x_max - x_min, y_max - y_min, top_z - bottom_z)
            center = (0.5 * (x_min + x_max), 0.5 * (y_min + y_max), 0.5 * (top_z + bottom_z))
            boxes.append(
                trimesh.creation.box(
                    extents=extents,
                    transform=trimesh.transformations.translation_matrix(center),
                )
            )

        def add_triangular_prism(top_vertices: np.ndarray) -> None:
            """Extrude one slope-correction corner triangle down to the common base."""
            top_vertices = top_vertices.copy()
            if np.cross(top_vertices[1] - top_vertices[0], top_vertices[2] - top_vertices[0])[2] < 0.0:
                top_vertices[[1, 2]] = top_vertices[[2, 1]]
            bottom_vertices = top_vertices.copy()
            bottom_vertices[:, 2] = bottom_z
            vertices = np.concatenate((top_vertices, bottom_vertices))
            faces = np.array(
                [
                    [0, 1, 2],
                    [3, 5, 4],
                    [0, 3, 4],
                    [0, 4, 1],
                    [1, 4, 5],
                    [1, 5, 2],
                    [2, 5, 3],
                    [2, 3, 0],
                ],
                dtype=np.int64,
            )
            boxes.append(trimesh.Trimesh(vertices=vertices, faces=faces, process=False))

        face_vertices = np.asarray(mesh.vertices)[np.asarray(mesh.faces)]
        face_centers = np.mean(face_vertices[:, :, :2], axis=1)
        horizontal_faces = np.ptp(face_vertices[:, :, 2], axis=1) < 1.0e-7
        for (top_z, outer), (_, inner) in zip(levels, levels[1:]):
            x_min, x_max, y_min, y_max = outer
            inner_x_min, inner_x_max, inner_y_min, inner_y_max = inner
            add_box(x_min, x_max, y_min, inner_y_min, top_z)
            add_box(x_min, x_max, inner_y_max, y_max, top_z)
            add_box(x_min, inner_x_min, inner_y_min, inner_y_max, top_z)
            add_box(inner_x_max, x_max, inner_y_min, inner_y_max, top_z)

            # Inverted height-field stairs retain two diagonal half-cells per
            # level because of Isaac Lab's corner slope correction.  Preserve
            # those exact triangles instead of expanding the lower box top
            # through them.
            corner_faces = (
                horizontal_faces
                & np.all(np.isclose(face_vertices[:, :, 2], top_z, atol=1.0e-7), axis=1)
                & (face_centers[:, 0] > inner_x_min + tolerance)
                & (face_centers[:, 0] < inner_x_max - tolerance)
                & (face_centers[:, 1] > inner_y_min + tolerance)
                & (face_centers[:, 1] < inner_y_max - tolerance)
            )
            for triangle in face_vertices[corner_faces]:
                add_triangular_prism(triangle)

        center_height, center = levels[-1]
        add_box(center[0], center[1], center[2], center[3], center_height)
        return trimesh.util.concatenate(boxes)


def flat_slab_terrain(
    difficulty: float,
    cfg: MeshFlatSlabTerrainCfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Create a closed flat slab whose top surface is at z=0."""
    del difficulty
    transform = trimesh.transformations.translation_matrix(
        (cfg.size[0] / 2.0, cfg.size[1] / 2.0, -cfg.thickness / 2.0)
    )
    slab = trimesh.creation.box(
        extents=(cfg.size[0], cfg.size[1], cfg.thickness),
        transform=transform,
    )
    origin = np.array((cfg.size[0] / 2.0, cfg.size[1] / 2.0, 0.0))
    return [slab], origin


@configclass
class MeshFlatSlabTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a volume-bearing flat mesh compatible with Newton."""

    function = flat_slab_terrain

    thickness: float = 0.1
    """Thickness below z=0 in meters."""


_COLUMN_BOX_FACES = np.array(
    (
        (0, 3, 2),
        (0, 2, 1),  # bottom (-z)
        (4, 5, 6),
        (4, 6, 7),  # top (+z)
        (0, 4, 7),
        (0, 7, 3),  # -x
        (1, 2, 6),
        (1, 6, 5),  # +x
        (0, 1, 5),
        (0, 5, 4),  # -y
        (3, 7, 6),
        (3, 6, 2),  # +y
    ),
    dtype=np.int64,
)


def _column_boxes_mesh(
    heights: np.ndarray,
    cell_size: float,
    base_z: float,
) -> trimesh.Trimesh:
    """Build one closed box per flat-topped column, vectorized over the grid.

    Column walls are not merged with their neighbours: adjacent boxes simply
    share coincident wall planes. This keeps every box watertight with exact
    flat tops and avoids the T-junctions of a merged column surface.
    """
    num_rows, num_cols = heights.shape
    rows = np.arange(num_rows, dtype=np.float64)[:, None]
    cols = np.arange(num_cols, dtype=np.float64)[None, :]
    x0 = np.broadcast_to(rows * cell_size, heights.shape)
    x1 = x0 + cell_size
    y0 = np.broadcast_to(cols * cell_size, heights.shape)
    y1 = y0 + cell_size
    z0 = np.full(heights.shape, float(base_z))
    z1 = heights.astype(np.float64)

    corners = np.stack(
        (
            np.stack((x0, y0, z0), axis=-1),
            np.stack((x1, y0, z0), axis=-1),
            np.stack((x1, y1, z0), axis=-1),
            np.stack((x0, y1, z0), axis=-1),
            np.stack((x0, y0, z1), axis=-1),
            np.stack((x1, y0, z1), axis=-1),
            np.stack((x1, y1, z1), axis=-1),
            np.stack((x0, y1, z1), axis=-1),
        ),
        axis=-2,
    )
    vertices = corners.reshape(-1, 3).astype(np.float32)
    offsets = (np.arange(num_rows * num_cols, dtype=np.int64) * 8)[:, None, None]
    faces = (_COLUMN_BOX_FACES[None] + offsets).reshape(-1, 3)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def wheels_terrain(
    difficulty: float,
    cfg: WheelsTerrainCfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a field of flat tyres around a tyre-free spawn disc.

    Tyres are placed on a square grid whose pitch keeps neighbouring tyres in
    contact. Every tyre overlapping the configured clear radius around the
    terrain center is skipped, so the robot spawns on flat ground and the tyres
    fill the rest of the tile. Inner radius grows and heights interpolate with
    the difficulty level.
    """
    level = float(np.clip(difficulty, 0.0, 1.0))
    size_x, size_y = (float(value) for value in cfg.size)
    center = np.array((size_x * 0.5, size_y * 0.5))
    ground = trimesh.creation.box(
        extents=(size_x, size_y, MESH_BASE_THICKNESS),
        transform=trimesh.transformations.translation_matrix(
            (center[0], center[1], -MESH_BASE_THICKNESS * 0.5)
        ),
    )
    meshes = [ground]

    inner_radius = float(np.interp(level, (0.0, 1.0), cfg.inner_radius_range))
    height_range = (1.0 - level) * np.asarray(cfg.easy_height_range) + level * np.asarray(
        cfg.height_range
    )
    pitch = 2.0 * cfg.outer_radius + level * cfg.max_gap
    skip_radius = cfg.clear_radius + cfg.outer_radius
    origin_height = 0.0
    num_cells = int(np.ceil(max(size_x, size_y) / pitch)) + 1
    for row in range(-num_cells, num_cells + 1):
        for col in range(-num_cells, num_cells + 1):
            offset = np.array(((row + 0.5) * pitch, (col + 0.5) * pitch))
            if np.linalg.norm(offset) < skip_radius:
                continue
            position = center + offset
            if not (
                cfg.outer_radius <= position[0] <= size_x - cfg.outer_radius
                and cfg.outer_radius <= position[1] <= size_y - cfg.outer_radius
            ):
                continue
            height = float(np.random.uniform(*height_range))
            tyre = trimesh.creation.annulus(
                r_min=inner_radius,
                r_max=cfg.outer_radius,
                height=height,
                sections=cfg.mesh_segments,
            )
            tyre.apply_translation((position[0], position[1], height * 0.5))
            meshes.append(tyre)
            if (
                abs(position[0] - center[0])
                <= SPAWN_CENTER_HALF_EXTENT + cfg.outer_radius
                and abs(position[1] - center[1])
                <= SPAWN_CENTER_HALF_EXTENT + cfg.outer_radius
            ):
                origin_height = max(origin_height, height)
    return meshes, np.array((center[0], center[1], origin_height))


def small_squares_terrain(
    difficulty: float,
    cfg: SmallSquaresTerrainCfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a closed block terrain of randomly raised square columns.

    Every square of ``cfg.square_size`` fills the whole tile and its height is
    sampled uniformly from ``[0, x]``, where ``x`` grows linearly from
    ``height_range[0]`` to ``height_range[1]`` with the difficulty level.
    """
    level = float(np.clip(difficulty, 0.0, 1.0))
    size_x, size_y = (float(value) for value in cfg.size)
    max_height = float(np.interp(level, (0.0, 1.0), cfg.height_range))
    num_rows = int(round(size_x / cfg.square_size))
    num_cols = int(round(size_y / cfg.square_size))
    heights = np.random.uniform(0.0, max_height, size=(num_rows, num_cols))
    mesh = _column_boxes_mesh(heights, cfg.square_size, -MESH_BASE_THICKNESS)

    half = max(1, int(round(SPAWN_CENTER_HALF_EXTENT / cfg.square_size)))
    center_row, center_col = num_rows // 2, num_cols // 2
    center_heights = heights[
        max(0, center_row - half) : min(num_rows, center_row + half),
        max(0, center_col - half) : min(num_cols, center_col + half),
    ]
    origin_height = float(np.max(center_heights))
    return [mesh], np.array((size_x * 0.5, size_y * 0.5, origin_height))


@configclass
class DirectMeshTerrainCfg(SubTerrainBaseCfg):
    """Base configuration for sub-terrains generated directly as mesh solids."""


@configclass
class WheelsTerrainCfg(DirectMeshTerrainCfg):
    """Configuration for a field of flat tyres around a tyre-free spawn disc."""

    function = wheels_terrain

    outer_radius: float = 0.30
    """Tyre outer radius in meters (0.6 m diameter)."""

    inner_radius_range: tuple[float, float] = (0.05, 0.25)
    """Tyre hole radius at difficulty 0 and 1; grows with the difficulty level."""

    easy_height_range: tuple[float, float] = (0.01, 0.03)
    """Per-tyre height range at difficulty 0."""

    height_range: tuple[float, float] = (0.15, 0.17)
    """Per-tyre height range at difficulty 1; interpolated from ``easy_height_range``."""

    max_gap: float = 0.0
    """Extra center-to-center gap at difficulty 1; 0 keeps neighbouring tyres in contact."""

    clear_radius: float = 1.0
    """Radius of the tyre-free disc around the terrain center."""

    mesh_segments: int = 32
    """Number of circumferential tyre segments."""


@configclass
class SmallSquaresTerrainCfg(DirectMeshTerrainCfg):
    """Configuration for a block terrain of randomly raised 10 cm squares."""

    function = small_squares_terrain

    square_size: float = 0.1
    """Edge length of every square column in meters."""

    height_range: tuple[float, float] = (0.01, 0.17)
    """Maximum square height at difficulty 0 and 1; heights are uniform in ``[0, x]``."""


@configclass
class Go2TerrainGeneratorCfg(terrain_gen.TerrainGeneratorCfg):
    """RobotLab terrain-generator options used by the mixed task."""

    class_type: type = PerSubTerrainSlopeThresholdGenerator
    sub_terrain_border_width: float | None = None
    # Keep the native 10 cm height-field mesh. Coarsening stair and obstacle
    # meshes changes Newton contact-force statistics at sharp boundaries.
    mesh_simplification_stride: int = 1
    # Preserve the original height-field stair surface instead of rebuilding
    # it from box rings, which amplifies calf contacts under Newton.
    stairs_representation: str = "heightfield"


def with_slope_threshold(sub_terrain_cfg, slope_threshold: float | None):
    """Attach a slope-threshold override to one height-field terrain."""
    sub_terrain_cfg.slope_threshold_override = slope_threshold
    return sub_terrain_cfg


def _height_field_with_origin(heights: np.ndarray, cfg) -> tuple[np.ndarray, np.ndarray]:
    """Attach a centered spawn origin to a raw height field for Isaac Lab 3.0."""
    center_x, center_y = (size // 2 for size in heights.shape)
    radius = max(1, int(SPAWN_CENTER_HALF_EXTENT / cfg.horizontal_scale))
    spawn_height = np.max(
        heights[max(0, center_x - radius) : center_x + radius, max(0, center_y - radius) : center_y + radius]
    )
    origin = np.array(
        [
            (heights.shape[0] - 1) * cfg.horizontal_scale / 2,
            (heights.shape[1] - 1) * cfg.horizontal_scale / 2,
            spawn_height * cfg.vertical_scale,
        ]
    )
    return heights, origin


@height_field_to_mesh
def robotlab_wave_terrain(difficulty: float, cfg) -> tuple[np.ndarray, np.ndarray]:
    """RobotLab wave terrain: sinusoidal waves plus fine roughness."""
    wave, _ = hf_terrains.wave_terrain.__wrapped__(difficulty, cfg)
    rough, _ = hf_terrains.random_uniform_terrain.__wrapped__(difficulty, cfg)
    return _height_field_with_origin(np.rint(wave + rough).astype(np.int16), cfg)


@height_field_to_mesh
def robotlab_rough_slope_terrain(difficulty: float, cfg) -> tuple[np.ndarray, np.ndarray]:
    """RobotLab rough slope terrain: pyramid slope plus fine roughness."""
    slope, _ = hf_terrains.pyramid_sloped_terrain.__wrapped__(difficulty, cfg)
    rough, _ = hf_terrains.random_uniform_terrain.__wrapped__(difficulty, cfg)
    return _height_field_with_origin(np.rint(slope + rough).astype(np.int16), cfg)


@configclass
class WaveTerrainCfg(terrain_gen.HfWaveTerrainCfg):
    function = robotlab_wave_terrain
    amplitude_range: tuple[float, float] = (0.1, 0.28)
    num_waves: int = 5
    noise_range: tuple[float, float] = (-0.05, 0.05)
    noise_step: float = 0.005
    downsampled_scale: float = 0.2


@configclass
class RoughSlopeTerrainCfg(terrain_gen.HfPyramidSlopedTerrainCfg):
    function = robotlab_rough_slope_terrain
    slope_range: tuple[float, float] = (0.1, 0.568)
    platform_width: float = 3.0
    noise_range: tuple[float, float] = (-0.05, 0.05)
    noise_step: float = 0.005
    downsampled_scale: float = 0.2


# Shared RobotLab terrain configuration for the mixed-terrain task.
MIXED_TERRAIN_CFG = Go2TerrainGeneratorCfg(
    size=(9.0, 9.0),
    border_width=25.0,
    sub_terrain_border_width=0.5,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    # Global default slope correction: 0.75 is approximately 36.9 degrees.
    # Each height-field terrain below overrides this with its recommended value.
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "wave": with_slope_threshold(
            WaveTerrainCfg(proportion=0.05),
            10.0,  # effectively disable slope correction for wave terrain
        ),
        "slope_up": with_slope_threshold(
            terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                proportion=0.10,
                slope_range=(0.1, 0.568),
                platform_width=3.0,
            ),
            10.0,  # effectively disable slope correction for slope-up terrain
        ),
        "slope_down": with_slope_threshold(
            terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.10,
                slope_range=(0.1, 0.568),
                platform_width=3.0,
            ),
            10.0,  # effectively disable slope correction for slope-down terrain
        ),
        "rough_slope": with_slope_threshold(
            RoughSlopeTerrainCfg(proportion=0.05),
            10.0,  # effectively disable slope correction for rough-slope terrain
        ),
        "stairs_up": with_slope_threshold(
            terrain_gen.HfInvertedPyramidStairsTerrainCfg(
                proportion=0.25,
                step_height_range=(0.05, 0.257),
                step_width=0.31,
                platform_width=3.0,
            ),
            0.25,  # preserve sharp stair edges with approximately 14.0-degree correction
        ),
        "stairs_down": with_slope_threshold(
            terrain_gen.HfPyramidStairsTerrainCfg(
                proportion=0.10,
                step_height_range=(0.05, 0.257),
                step_width=0.31,
                platform_width=3.0,
            ),
            0.25,  # preserve sharp stair edges with approximately 14.0-degree correction
        ),
        "obstacles": with_slope_threshold(
            terrain_gen.HfDiscreteObstaclesTerrainCfg(
                proportion=0.10,
                obstacle_width_range=(1.0, 2.0),
                obstacle_height_range=(0.05, 0.275),
                num_obstacles=20,
                platform_width=3.0,
            ),
            0.25,  # preserve obstacle edges with approximately 14.0-degree correction
        ),
        # Mesh terrains: height-field slope correction does not apply.
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.15),
        "wheels": WheelsTerrainCfg(proportion=0.05),
        "small_squares": SmallSquaresTerrainCfg(proportion=0.05),
    },
)

```

### official_terrain_mesh.py · SHA256 51e74c9073b9e53a9da440809315849e8ce9897647d0357cca67fae065ef2fed

```python
"""Direct-mesh generation backend for the Go2 mixed terrain.

Each sub-terrain is converted into a closed watertight triangular mesh at
generation time, giving sharp stair and obstacle edges without the regular
height-field tessellation. The height-field counterpart lives in ``terrain.py``.
"""

from __future__ import annotations

import copy
from collections.abc import Callable

import numpy as np
import trimesh

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import SubTerrainBaseCfg, TerrainImporter
from isaaclab.terrains.height_field import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh

from .official_terrain_heightfield import (
    MESH_BASE_THICKNESS,
    MIXED_TERRAIN_CFG,
    DirectMeshTerrainCfg,
    PerSubTerrainSlopeThresholdGenerator,
    RoughSlopeTerrainCfg,
    WaveTerrainCfg,
)


def _resolve_height_function(function: Callable) -> Callable:
    """Resolve config-style callables and remove the height-field decorator."""
    if hasattr(function, "_resolve"):
        function = function._resolve()
    return getattr(function, "__wrapped__", function)


def _height_field_with_border(
    difficulty: float,
    cfg: HfTerrainBaseCfg,
    horizontal_scale_override: float | None = None,
) -> tuple[np.ndarray, tuple[float, float], float, float]:
    """Build the same bordered height field used by Isaac Lab's decorator."""
    work_cfg = cfg.copy()
    horizontal_scale = float(
        horizontal_scale_override
        if horizontal_scale_override is not None
        else work_cfg.horizontal_scale
    )
    work_cfg.horizontal_scale = horizontal_scale
    vertical_scale = float(work_cfg.vertical_scale)
    width_pixels = int(work_cfg.size[0] / horizontal_scale) + 1
    length_pixels = int(work_cfg.size[1] / horizontal_scale) + 1
    border_pixels = int(work_cfg.border_width / horizontal_scale) + 1
    heights = np.zeros((width_pixels, length_pixels), dtype=np.int16)

    terrain_size = tuple(float(value) for value in work_cfg.size)
    sub_terrain_size = (
        (width_pixels - 2 * border_pixels) * horizontal_scale,
        (length_pixels - 2 * border_pixels) * horizontal_scale,
    )
    work_cfg.size = sub_terrain_size
    slope_threshold_override = getattr(work_cfg, "slope_threshold_override", None)
    if slope_threshold_override is not None:
        work_cfg.slope_threshold = slope_threshold_override
    height_function = _resolve_height_function(work_cfg.function)
    generated_heights, _ = height_function(difficulty, work_cfg)
    heights[border_pixels:-border_pixels, border_pixels:-border_pixels] = generated_heights
    return heights, terrain_size, horizontal_scale, vertical_scale


def _boundary_edges(num_rows: int, num_cols: int) -> np.ndarray:
    """Return counter-clockwise boundary edges for a regular grid."""
    edges: list[tuple[int, int]] = []
    for col in range(num_cols - 1):
        edges.append((col + 1, col))
        edges.append(((num_rows - 1) * num_cols + col, (num_rows - 1) * num_cols + col + 1))
    for row in range(num_rows - 1):
        edges.append((row * num_cols, (row + 1) * num_cols))
        edges.append(((row + 1) * num_cols + num_cols - 1, row * num_cols + num_cols - 1))
    return np.asarray(edges, dtype=np.int64)


def _height_field_origin(
    heights: np.ndarray,
    terrain_size: tuple[float, float],
    horizontal_scale: float,
    vertical_scale: float,
) -> np.ndarray:
    """Return the centered spawn origin used by Isaac Lab's height-field wrapper."""
    center_x_1 = int((terrain_size[0] * 0.5 - 1.0) / horizontal_scale)
    center_x_2 = int((terrain_size[0] * 0.5 + 1.0) / horizontal_scale)
    center_y_1 = int((terrain_size[1] * 0.5 - 1.0) / horizontal_scale)
    center_y_2 = int((terrain_size[1] * 0.5 + 1.0) / horizontal_scale)
    return np.array(
        (
            terrain_size[0] * 0.5,
            terrain_size[1] * 0.5,
            float(np.max(heights[center_x_1:center_x_2, center_y_1:center_y_2]))
            * vertical_scale,
        ),
        dtype=np.float64,
    )


def _effective_slope_threshold(cfg: HfTerrainBaseCfg) -> float | None:
    """Return the per-sub-terrain slope threshold override when present."""
    return (
        cfg.slope_threshold_override
        if getattr(cfg, "slope_threshold_override", None) is not None
        else cfg.slope_threshold
    )


def _close_height_field_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    grid_shape: tuple[int, int],
    terrain_size: tuple[float, float],
    base_z: float,
) -> trimesh.Trimesh:
    """Close one regular height-field surface with side walls and a bottom cap."""
    num_rows, num_cols = grid_shape
    boundary = _boundary_edges(num_rows, num_cols)
    bottom_indices: dict[int, int] = {}
    bottom_vertices: list[np.ndarray] = []
    for top_index in boundary.reshape(-1):
        if int(top_index) in bottom_indices:
            continue
        bottom_indices[int(top_index)] = len(vertices) + len(bottom_vertices)
        bottom_vertex = vertices[top_index].copy()
        bottom_vertex[2] = base_z
        bottom_vertices.append(bottom_vertex)

    center_bottom = len(vertices) + len(bottom_vertices)
    bottom_vertices.append(
        np.array((terrain_size[0] * 0.5, terrain_size[1] * 0.5, base_z), dtype=np.float32)
    )
    side_faces = np.empty((len(boundary) * 2, 3), dtype=np.int64)
    bottom_faces = np.empty((len(boundary), 3), dtype=np.int64)
    for index, (top_a, top_b) in enumerate(boundary):
        bottom_a = bottom_indices[int(top_a)]
        bottom_b = bottom_indices[int(top_b)]
        side_faces[2 * index] = (top_a, bottom_a, bottom_b)
        side_faces[2 * index + 1] = (top_a, bottom_b, top_b)
        bottom_faces[index] = (center_bottom, bottom_b, bottom_a)

    mesh = trimesh.Trimesh(
        vertices=np.vstack((vertices, np.asarray(bottom_vertices, dtype=np.float32))),
        faces=np.vstack((faces, side_faces, bottom_faces)),
        process=False,
    )
    mesh.fix_normals()
    if not mesh.is_watertight or not mesh.is_volume:
        raise RuntimeError("Generated mixed mesh is not a watertight volume.")
    return mesh


def _compression_indices(heights: np.ndarray, axis: int) -> np.ndarray:
    """Keep only grid lines that participate in a height discontinuity."""
    differences = np.diff(heights, axis=axis)
    changed_rows = np.any(differences != 0, axis=1 - axis)
    changed_indices = np.flatnonzero(changed_rows)
    selected = {0, heights.shape[axis] - 1}
    selected.update(changed_indices.tolist())
    selected.update((changed_indices + 1).tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def _closed_height_field_mesh(
    difficulty: float,
    cfg: HfTerrainBaseCfg,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Generate a closed mesh from one mixed-terrain height field."""
    smooth_terrain = isinstance(
        cfg,
        (WaveTerrainCfg, RoughSlopeTerrainCfg, terrain_gen.HfPyramidSlopedTerrainCfg),
    )
    heights, terrain_size, horizontal_scale, vertical_scale = _height_field_with_border(
        difficulty,
        cfg,
        horizontal_scale_override=0.2 if smooth_terrain else None,
    )
    vertices, faces = convert_height_field_to_mesh(
        heights,
        horizontal_scale,
        vertical_scale,
        _effective_slope_threshold(cfg),
    )
    base_z = float(np.min(heights)) * vertical_scale - MESH_BASE_THICKNESS
    mesh = _close_height_field_surface(
        vertices,
        faces,
        heights.shape,
        terrain_size,
        base_z,
    )
    return mesh, _height_field_origin(heights, terrain_size, horizontal_scale, vertical_scale)


def _box_obstacle_mesh(
    difficulty: float,
    cfg: terrain_gen.HfDiscreteObstaclesTerrainCfg,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Generate the obstacle height field as a union of axis-aligned boxes."""
    heights, terrain_size, horizontal_scale, vertical_scale = _height_field_with_border(
        difficulty, cfg
    )
    row_indices = _compression_indices(heights, axis=0)
    column_indices = _compression_indices(heights, axis=1)
    compressed = heights[np.ix_(row_indices, column_indices)]

    cell_heights = compressed[:-1, :-1]
    x_edges = row_indices * horizontal_scale
    y_edges = column_indices * horizontal_scale
    base_z = float(np.min(heights)) * vertical_scale - MESH_BASE_THICKNESS
    height_levels = np.unique(cell_heights)
    z_edges = np.concatenate(([base_z], height_levels.astype(np.float64) * vertical_scale))
    occupied = cell_heights[:, :, None] >= height_levels[None, None, :]
    num_rows, num_cols, num_levels = occupied.shape

    vertices: list[tuple[float, float, float]] = []
    vertex_indices: dict[tuple[int, int, int], int] = {}
    faces: list[tuple[int, int, int]] = []

    def vertex(row: int, col: int, level: int) -> int:
        key = (row, col, level)
        if key not in vertex_indices:
            vertex_indices[key] = len(vertices)
            vertices.append(
                (
                    float(x_edges[row]),
                    float(y_edges[col]),
                    float(z_edges[level]),
                )
            )
        return vertex_indices[key]

    def add_quad(a: int, b: int, c: int, d: int) -> None:
        faces.extend(((a, b, c), (a, c, d)))

    for row, col, level in np.argwhere(occupied):
        if row == 0 or not occupied[row - 1, col, level]:
            add_quad(
                vertex(row, col, level),
                vertex(row, col, level + 1),
                vertex(row, col + 1, level + 1),
                vertex(row, col + 1, level),
            )
        if row == num_rows - 1 or not occupied[row + 1, col, level]:
            add_quad(
                vertex(row + 1, col, level),
                vertex(row + 1, col + 1, level),
                vertex(row + 1, col + 1, level + 1),
                vertex(row + 1, col, level + 1),
            )
        if col == 0 or not occupied[row, col - 1, level]:
            add_quad(
                vertex(row, col, level),
                vertex(row + 1, col, level),
                vertex(row + 1, col, level + 1),
                vertex(row, col, level + 1),
            )
        if col == num_cols - 1 or not occupied[row, col + 1, level]:
            add_quad(
                vertex(row, col + 1, level),
                vertex(row, col + 1, level + 1),
                vertex(row + 1, col + 1, level + 1),
                vertex(row + 1, col + 1, level),
            )
        if level == 0 or not occupied[row, col, level - 1]:
            add_quad(
                vertex(row, col, level),
                vertex(row, col + 1, level),
                vertex(row + 1, col + 1, level),
                vertex(row + 1, col, level),
            )
        if level == num_levels - 1 or not occupied[row, col, level + 1]:
            add_quad(
                vertex(row, col, level + 1),
                vertex(row + 1, col, level + 1),
                vertex(row + 1, col + 1, level + 1),
                vertex(row, col + 1, level + 1),
            )

    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    return mesh, _height_field_origin(heights, terrain_size, horizontal_scale, vertical_scale)


def _stairs_mesh(
    difficulty: float,
    cfg: terrain_gen.HfPyramidStairsTerrainCfg,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Convert the exact discrete stair height field into closed box rings."""
    heights, terrain_size, horizontal_scale, vertical_scale = _height_field_with_border(
        difficulty, cfg
    )
    vertices, faces = convert_height_field_to_mesh(
        heights,
        horizontal_scale,
        vertical_scale,
        _effective_slope_threshold(cfg),
    )
    surface = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh = PerSubTerrainSlopeThresholdGenerator._heightfield_stairs_to_boxes(surface, cfg)
    return mesh, _height_field_origin(heights, terrain_size, horizontal_scale, vertical_scale)


def _closed_flat_mesh(
    difficulty: float,
    cfg: SubTerrainBaseCfg,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Generate a closed flat slab with its walkable surface at ``z=0``."""
    del difficulty
    size_x, size_y = (float(value) for value in cfg.size)
    mesh = trimesh.creation.box(
        extents=(size_x, size_y, MESH_BASE_THICKNESS),
        transform=trimesh.transformations.translation_matrix(
            (size_x * 0.5, size_y * 0.5, -MESH_BASE_THICKNESS * 0.5)
        ),
    )
    return mesh, np.array((size_x * 0.5, size_y * 0.5, 0.0), dtype=np.float64)


def _stairs_crease_segments(mesh: trimesh.Trimesh) -> np.ndarray | None:
    """Collect step creases of one stair tile, dropping slab and tile perimeter edges."""
    selected = np.degrees(mesh.face_adjacency_angles) > 30.0
    if not np.any(selected):
        return None
    segments = mesh.vertices[mesh.face_adjacency_edges[selected]]
    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    segments = segments[lengths > 1.0e-6]
    bounds = mesh.bounds
    segments = segments[segments[:, :, 2].max(axis=1) > bounds[0, 2] + 1.0e-3]
    on_x = (np.abs(segments[:, :, 0] - bounds[0, 0]) < 1.0e-3) | (
        np.abs(segments[:, :, 0] - bounds[1, 0]) < 1.0e-3
    )
    on_y = (np.abs(segments[:, :, 1] - bounds[0, 1]) < 1.0e-3) | (
        np.abs(segments[:, :, 1] - bounds[1, 1]) < 1.0e-3
    )
    segments = segments[~((on_x[:, 0] & on_x[:, 1]) | (on_y[:, 0] & on_y[:, 1]))]
    return segments if len(segments) > 0 else None


class MixedMeshTerrainGenerator(PerSubTerrainSlopeThresholdGenerator):
    """Generate a closed triangular mesh for each configured mixed sub-terrain."""

    def __init__(self, cfg, device: str = "cpu"):
        self._stair_outline_chunks: list[np.ndarray] = []
        super().__init__(cfg=cfg, device=device)
        if self._stair_outline_chunks:
            segments = np.concatenate(self._stair_outline_chunks)
            transform = np.eye(4)
            transform[:2, -1] = -cfg.size[0] * cfg.num_rows * 0.5, -cfg.size[1] * cfg.num_cols * 0.5
            segments += transform[:3, -1]
            self.terrain_mesh.stair_outline_segments = segments

    def _get_terrain_mesh(self, difficulty: float, cfg: SubTerrainBaseCfg):
        work_cfg = copy.deepcopy(cfg)
        if isinstance(work_cfg, terrain_gen.HfPyramidStairsTerrainCfg):
            mesh, origin = _stairs_mesh(difficulty, work_cfg)
        elif isinstance(work_cfg, terrain_gen.HfDiscreteObstaclesTerrainCfg):
            mesh, origin = _box_obstacle_mesh(difficulty, work_cfg)
        elif isinstance(work_cfg, DirectMeshTerrainCfg):
            meshes, origin = work_cfg.function(difficulty, work_cfg)
            mesh = trimesh.util.concatenate(meshes)
        elif isinstance(work_cfg, HfTerrainBaseCfg):
            mesh, origin = _closed_height_field_mesh(difficulty, work_cfg)
        else:
            mesh, origin = _closed_flat_mesh(difficulty, work_cfg)

        transform = np.eye(4)
        transform[:2, 3] = -work_cfg.size[0] * 0.5, -work_cfg.size[1] * 0.5
        mesh.apply_transform(transform)
        return mesh, np.asarray(origin, dtype=np.float64) + transform[:3, 3]

    def _add_sub_terrain(
        self,
        mesh: trimesh.Trimesh,
        origin: np.ndarray,
        row: int,
        col: int,
        sub_terrain_cfg: SubTerrainBaseCfg,
    ) -> None:
        """Store step creases of stair tiles once the tile is placed in the world frame."""
        super()._add_sub_terrain(mesh, origin, row, col, sub_terrain_cfg)
        if isinstance(sub_terrain_cfg, terrain_gen.HfPyramidStairsTerrainCfg):
            segments = _stairs_crease_segments(mesh)
            if segments is not None:
                self._stair_outline_chunks.append(segments)


MIXED_MESH_TERRAIN_CFG = copy.deepcopy(MIXED_TERRAIN_CFG)
MIXED_MESH_TERRAIN_CFG.class_type = MixedMeshTerrainGenerator
MIXED_MESH_TERRAIN_CFG.use_cache = False
MIXED_MESH_TERRAIN_CFG.color_scheme = "none"


```

### physx_env.py · SHA256 476e64ae492b45e84a00e4512e6eae1e80364e5570a884d3abb70fbc11faa86d

```python
"""Isaac Lab3 / Isaac Sim PhysX adapter. No Newton task or learner imports.

All public learner tensors use leg-major ordering and XYZW quaternions. Motor
parameters follow the pinned source; raw actions are not tanh-squashed.
"""
import math
import torch
from . import state as st
from .task import RobotConsts,reward,fell
from .commands import OfficialCommands
from .terrain import relief,ground

def tensor(value):return value.torch if hasattr(value,'torch') else value


class PlaneTerrainLookup:
    names=('flat',)
    def query(self,xy):return torch.zeros(xy.shape[:-1],dtype=torch.long,device=xy.device)
    __call__=query

def quat_inverse(q,v):
    xyz,w=q[...,:3],q[...,3:4]
    return v-2*w*torch.cross(xyz,v,dim=-1)+2*torch.cross(xyz,torch.cross(xyz,v,dim=-1),dim=-1)

def motor_values(device):
    inertia=torch.tensor([.000111842*(6.33*(1.5 if j=='calf' else 1))**2 for _ in st.LEGS for j in ('hip','thigh','calf')],device=device)
    kp=inertia*(2*math.pi*10)**2;kd=4*inertia*(2*math.pi*10)
    effort=torch.tensor([23.7,23.7,45.43]*4,device=device)
    return inertia,kp,kd,effort,.25*effort/kp

class PhysXEnv:
    def __init__(self,cfg,sim,usd_path):
        import isaaclab.sim as su
        from isaaclab.assets import ArticulationCfg,AssetBaseCfg
        from isaaclab.actuators import DCMotorCfg, IdealPDActuatorCfg
        from .robots import profile,as2_motor_values
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.utils import configclass,instantiate
        from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg,PhysxArticulationCfg
        self.cfg,self.sim,self.device,self.num_envs=cfg,sim,cfg.device,cfg.num_envs
        self.field=None;self.terrain_lookup=PlaneTerrainLookup();self.terrain_names=('flat',)
        self.terrain_metadata={'names':['flat'],'geometry':'infinite_ground_plane'}
        self.evaluation_mode=False;self.command_override=None
        self.profile=profile(cfg.robot)
        inertia,kp,kd,effort,scale=(motor_values(cfg.device) if cfg.robot=='go2' else as2_motor_values(cfg.device))
        def mapping(v):return {n:float(x) for n,x in zip(st.JOINT_NAMES,v)}
        velocity={f'.*_{j}_joint':v for j,v in zip(('hip','thigh','calf'),self.profile.velocity_limits)}
        actuator_args=dict(joint_names_expr=['.*'],stiffness=mapping(kp),damping=mapping(kd),armature=mapping(inertia),friction=0.,actuator_effort_limit=mapping(effort),actuator_velocity_limit=velocity,joint_velocity_limit=velocity)
        actuator=(DCMotorCfg(saturation_effort=mapping(2*effort),**actuator_args) if cfg.robot=='go2' else IdealPDActuatorCfg(**actuator_args))
        robot_cfg=ArticulationCfg(prim_path='{ENV_REGEX_NS}/Robot',joint_ordering=st.JOINT_NAMES,
            spawn=su.UsdFileCfg(usd_path=usd_path,activate_contact_sensors=False,
                rigid_props=PhysxRigidBodyCfg(disable_gravity=False,linear_damping=0.,angular_damping=0.,max_depenetration_velocity=1.),
                articulation_props=PhysxArticulationCfg(enabled_self_collisions=True,solver_position_iteration_count=4,solver_velocity_iteration_count=0)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0,0,self.profile.initial_height),joint_pos={'.*L_hip_joint':self.profile.left_hip,'.*R_hip_joint':-self.profile.left_hip,'.*_thigh_joint':.9,'.*_calf_joint':-1.8},joint_vel={'.*':0.}),
            soft_joint_pos_limit_factor=.9,
            actuators={'legs':actuator})
        @configclass
        class SceneCfg(InteractiveSceneCfg):
            ground=AssetBaseCfg(prim_path='/World/Ground',spawn=su.GroundPlaneCfg(physics_material=su.RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=1.,restitution=0.)))
            robot=robot_cfg
        scene_cfg=SceneCfg(num_envs=cfg.num_envs,env_spacing=3.,replicate_physics=True)
        if cfg.stage=='terrain':
            from .terrain_scene import terrain_config
            scene_cfg.ground=None;scene_cfg.terrain=terrain_config(cfg.seed)
        self.scene=instantiate(scene_cfg);self.robot=self.scene['robot'];sim.reset()
        if 'isaaclab_physx' not in type(self.robot).__module__:raise RuntimeError(f'Wrong backend: {type(self.robot)}')
        if tuple(self.robot.joint_names)!=st.JOINT_NAMES:raise RuntimeError('Joint ordering mismatch')
        self.consts=RobotConsts(tensor(self.robot.data.default_joint_pos)[0].clone(),scale,kp,kd,effort,tensor(self.robot.data.soft_joint_pos_limits)[0].clone())
        self.body_ids={part:[list(self.robot.body_names).index(f'{leg}_{part}') for leg in st.LEGS] for part in ('hip','thigh','calf','foot') if all(f'{leg}_{part}' in self.robot.body_names for leg in st.LEGS)}
        if not all(k in self.body_ids for k in ('hip','thigh','calf')):raise RuntimeError(f'Incomplete limb body mapping: {self.robot.body_names}')
        if cfg.robot=='as2' and 'foot' not in self.body_ids:raise RuntimeError('As2 requires actual fixed foot bodies; Go2 foot offsets are invalid')
        self.origins=tensor(self.scene.env_origins).clone()
        if cfg.stage=='terrain':
            from .terrain_scene import field_from_mesh,TerrainPositionLookup
            self.field=field_from_mesh(self.scene.terrain,cfg.device)
            self.origins=tensor(self.scene.terrain.env_origins).clone()
            self.terrain_lookup=TerrainPositionLookup(self.scene.terrain.cfg.terrain_generator,cfg.device)
            self.terrain_names=self.terrain_lookup.names
            self.terrain_metadata=dict(self.terrain_lookup.metadata,collision_heightfield=self.field.metadata)
        n=cfg.num_envs;dev=cfg.device
        self.xi=torch.zeros(n,st.CONTEXT_DIM,device=dev);self.xi[:,0]=1.
        self.command_sampler=OfficialCommands(cfg.dt,cfg.episode_seconds,terrain_lookup=self.terrain_lookup)
        self.cmd_state=self.command_sampler.initial(self.origins[:,:2],self.origins[:,:2],iteration=0)
        self.command=self.cmd_state[:,3:6];self.timer=self.cmd_state[:,6:7]
        self.hist=torch.zeros(n,450,device=dev);self.last=torch.zeros(n,12,device=dev)
        self.steps=torch.zeros(n,dtype=torch.long,device=dev);self.episode=torch.zeros(n,1,dtype=torch.int32,device=dev)
        self.own_mask=torch.ones(n,dtype=torch.bool,device=dev)
        self.delay=torch.randint(3,(n,),device=dev);self.target_ring=self.consts.default_q.expand(3,n,12).clone();self.ring_ptr=0
        self.ids=torch.arange(n,device=dev);self.iteration=0
        self._terrain_max_levels=torch.full((len(self.terrain_names),),-1.,device=dev)
        self.terrain_level_statistics={}
        # Payload is a fixed rigid mass at the existing base COM, changed only at reset.
        if not 0.<=cfg.payload_max_kg<=2.:raise ValueError('Payload must be in0..2kg')
        self._mass_view=True
        self._base_id=list(self.robot.body_names).index('base')
        self._nominal_masses=tensor(self.robot.data.body_mass).clone()
        self.actual_base_mass_kg=self._nominal_masses[:,self._base_id].clone()
        self.reset(self.ids);self._snapshot=self._observe()

    def _observe(self,measured_crop=None):
        d=self.robot.data;rp=tensor(d.root_link_pose_w);q=rp[:,3:7];pos=rp[:,:3]
        yaw=torch.atan2(2*(q[:,3]*q[:,2]+q[:,0]*q[:,1]),1-2*(q[:,1].square()+q[:,2].square()))
        vel=tensor(d.root_link_vel_w);g=torch.zeros_like(pos);g[:,2]=-1
        grav=quat_inverse(q,g);vy=st.yaw_inverse(vel[:,:3],yaw);wb=quat_inverse(q,vel[:,3:])
        body=tensor(d.body_link_pose_w);points=[]
        # Source points are hip joint, knee joint and foot. Link origins coincide
        # with hip/thigh/calf joints; foot may be a merged fixed link in PhysX.
        for part in ('thigh','calf','foot'):
            if part in self.body_ids:pw=body[:,self.body_ids[part],:3]
            else:
                bp=body[:,self.body_ids['calf']];offset=torch.zeros_like(bp[:,:,:3]);offset[:,:,2]=-self.profile.calf_length
                qc=bp[:,:,3:7];pw=bp[:,:,:3]+quat_inverse(torch.cat((-qc[:,:,:3],qc[:,:,3:]),-1),offset)
            points.append(st.yaw_inverse(pw-pos[:,None,:],yaw[:,None]).flatten(1))
        s=torch.cat(((pos[:,2]-ground(pos[:,:2],self.field))[:,None],grav,vy,wb,
                     tensor(d.joint_pos)-self.consts.default_q,tensor(d.joint_vel),*points),-1)
        pose=torch.cat((pos[:,:2],yaw[:,None]),-1)
        if self.field is None:
            height_above_crop=pos[:,2];height_crop_valid=torch.ones(self.num_envs,dtype=torch.bool,device=self.device)
            terrain_id=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
            assigned_id=terrain_id;level=terrain_id
        else:
            height_above_crop,height_crop_valid=(self.field.raycaster.height_crop(pos,yaw)
                if measured_crop is None else measured_crop)
            terrain_id=self.terrain_lookup.query(pos)
            assigned_id=self.terrain_lookup.column_types[self.scene.terrain.terrain_types]
            level=self.scene.terrain.terrain_levels
        return {'s':s,'pose':pose,'a_prev':self.last.clone(),'obs':self.hist.clone(),'command':self.command.clone(),
                'timer':self.timer.clone(),'episode':self.episode.clone(),'relief':relief(pose,self.field),
                'cmd_state':self.cmd_state.clone(),'height_above_crop':height_above_crop,
                'height_crop_valid':height_crop_valid,'terrain_id':terrain_id,'assigned_terrain_id':assigned_id.clone(),
                'terrain_level':level.clone(),'episode_steps':self.steps.clone(),'xi':self.xi.clone(),
                'actual_base_mass_kg':self.actual_base_mass_kg.clone(),'payload_kg':self.xi[:,4].clone()}

    def snapshot(self):
        if self._snapshot is None:self._snapshot=self._observe()
        if self.command_override is not None:
            self._command_views()
            self._snapshot['command']=self.command.clone();self._snapshot['cmd_state']=self.cmd_state.clone()
        return self._snapshot

    def reset(self,ids):
        if not len(ids):return
        n=len(ids);dev=self.device
        if self.cfg.stage=='terrain':
            terrain=self.scene.terrain
            # Initial assignments remain the source max-init level5; only completed
            # training episodes can promote/demote, independent of training iteration.
            eligible=ids[self.steps[ids]>0]
            if len(eligible) and not self.evaluation_mode:
                from .terrain_scene import terrain_level_decisions
                gen=terrain.cfg.terrain_generator
                up,down=terrain_level_decisions(self.cmd_state[eligible,12],self.cmd_state[eligible,7:9],
                    self.command_sampler.resampling_time,self.command_sampler.zero_command_prob(self.iteration),
                    gen.size[0],gen.sub_terrain_border_width)
                terrain.update_env_origins(eligible,up,down)
            self.origins[ids]=tensor(terrain.env_origins)[ids]
            self._record_terrain_levels()
        pose=tensor(self.robot.data.default_root_pose)[ids].clone();pose[:,:3]+=self.origins[ids]
        pose[:,:2]+=(torch.rand(n,2,device=dev)-.5)*.5
        yaw=(torch.rand(n,device=dev)*2-1)*math.pi;pose[:,3:]=0;pose[:,5]=torch.sin(yaw/2);pose[:,6]=torch.cos(yaw/2)
        pos=self.consts.default_q.expand(n,12).clone()
        self.robot.write_root_pose_to_sim_index(root_pose=pose,env_ids=ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n,6,device=dev),env_ids=ids)
        self.robot.write_joint_position_to_sim_index(position=pos,env_ids=ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(pos),env_ids=ids)
        self.robot.reset(ids)
        self.hist[ids]=0;self.last[ids]=0;self.steps[ids]=0;self.episode[ids]+=1
        self.target_ring[:,ids]=pos;self.delay[ids]=torch.randint(3,(n,),device=dev)
        mask=torch.zeros(self.num_envs,dtype=torch.bool,device=dev);mask[ids]=True
        all_positions=tensor(self.robot.data.root_link_pose_w)[:,:2].clone()
        all_positions[ids]=pose[:,:2]
        self.cmd_state=self.command_sampler.reset(self.cmd_state,mask,all_positions,self.origins[:,:2],self.iteration)
        self._command_views()
        if not self.evaluation_mode:self.set_payload(ids,torch.rand(n,device=dev)*self.cfg.payload_max_kg)
        self._snapshot=None

    def _command_views(self):
        if self.command_override is not None:
            if self.command_override.shape!=(self.num_envs,3):raise ValueError('command_override must be[N,3]')
            self.cmd_state[:,0:3]=self.command_override;self.cmd_state[:,3:6]=self.command_override
        self.command=self.cmd_state[:,3:6];self.timer=self.cmd_state[:,6:7]

    def set_payload(self,ids,payload):
        """Apply a point payload at existing COM only at an episode boundary."""
        if not len(ids):return
        if bool((self.steps[ids]!=0).any()):raise ValueError('Payload may change only after reset, never inside an episode')
        payload=torch.as_tensor(payload,device=self.device,dtype=self.xi.dtype).expand(len(ids))
        if not bool(torch.isfinite(payload).all()) or bool(((payload<0)|(payload>2.)).any()):
            raise ValueError('Payload must be finite and in0..2kg')
        masses=(self._nominal_masses[ids,self._base_id]+payload)[:,None]
        self.robot.set_masses_index(masses=masses,body_ids=[self._base_id],env_ids=ids)
        actual=tensor(self.robot.data.body_mass)[ids,self._base_id]
        if not torch.allclose(actual,masses[:,0],rtol=1e-6,atol=1e-6):raise RuntimeError('PhysX payload mass readback differs')
        self.actual_base_mass_kg[ids]=actual
        self.xi[ids,4]=actual-self._nominal_masses[ids,self._base_id]
        if getattr(self,'_snapshot',None) is not None:
            self._snapshot['xi']=self.xi.clone();self._snapshot['payload_kg']=self.xi[:,4].clone()
            self._snapshot['actual_base_mass_kg']=self.actual_base_mass_kg.clone()

    def _record_terrain_levels(self):
        terrain=self.scene.terrain;levels=terrain.terrain_levels.float()
        assigned=self.terrain_lookup.column_types[terrain.terrain_types]
        self.terrain_level_statistics={'mean_level':float(levels.mean())}
        for index,name in enumerate(self.terrain_names):
            values=levels[assigned==index]
            if len(values):self._terrain_max_levels[index]=torch.maximum(self._terrain_max_levels[index],values.max())
            self.terrain_level_statistics[name+'/mean_level']=float(values.mean()) if len(values) else -1.
            self.terrain_level_statistics[name+'/max_level_reached']=float(self._terrain_max_levels[index])

    def terrain_type_at(self,xy):
        return torch.zeros(xy.shape[:-1],dtype=torch.long,device=xy.device) if self.terrain_lookup is None else self.terrain_lookup.query(xy)

    def curriculum_state_dict(self):
        return {'terrain_levels':None if self.field is None else self.scene.terrain.terrain_levels.clone(),
                'max_levels_reached':self._terrain_max_levels.clone()}

    def load_curriculum_state_dict(self,state):
        self._terrain_max_levels.copy_(state['max_levels_reached'])
        if self.field is not None and state['terrain_levels'] is not None:
            terrain=self.scene.terrain;terrain.terrain_levels.copy_(state['terrain_levels'])
            terrain.env_origins[:]=terrain.terrain_origins[terrain.terrain_levels,terrain.terrain_types]
            self.origins[:]=terrain.env_origins
            self._record_terrain_levels()

    @torch.no_grad()
    def step(self,a):
        self._command_views()
        before=self.snapshot()['s'];target=self.consts.default_q+a*self.consts.action_scale
        for _ in range(round(self.cfg.dt/self.cfg.physics_dt)):
            self.target_ring[self.ring_ptr]=target
            delayed=self.target_ring[(self.ring_ptr-self.delay)%3,self.ids];self.ring_ptr=(self.ring_ptr+1)%3
            self.robot.set_joint_position_target_index(target=delayed)
            self.scene.write_data_to_sim();self.sim.step(render=False);self.scene.update(self.cfg.physics_dt)
        self.steps+=1;after=self._observe();s=after['s']
        r,terms,_=reward(before,s,a,self.last,self.command,self.consts,self.cfg.stage,
                         iteration=self.iteration,height_above_crop=after['height_above_crop'],base_height_target=self.profile.reward_height)
        falls=fell(s);timeout=self.steps>=round(self.cfg.episode_seconds/self.cfg.dt);done=falls|timeout
        if not torch.isfinite(s).all():raise FloatingPointError('Nonfinite PhysX state')
        own=self.own_mask;vb=st.velocity_body(s)
        metrics={'height':float(s[own,0].mean()),'xy_rmse':float((vb[own,:2]-self.command[own,:2]).square().mean().sqrt()),
                 'yaw_rmse':float((s[own,9]-self.command[own,2]).square().mean().sqrt()),'reward':float(r[own].mean()),'falls':int((falls&own).sum()),'falls_all':int(falls.sum())}
        self.last=a.clone()
        if self.command_override is None:
            self.cmd_state=self.command_sampler.advance(self.cmd_state,after['pose'][:,:2],self.iteration)
        self._command_views()
        self.hist=st.push_history(self.hist,st.frame(s,a,self.command,noise=True))
        reset_ids=done.nonzero().flatten();self.reset(reset_ids)
        cached_crop=None if len(reset_ids) else (after['height_above_crop'],after['height_crop_valid'])
        self._snapshot=self._observe(measured_crop=cached_crop)
        # Return terminal state for labels; replay masks all reset transitions.
        return after,done,metrics

```

### train.py · SHA256 9a170c5ef8d5307b510dc297bf21e80a9e7b3649d94b1aaa3cb12f5f4f08d3bb

```python
"""Only supported launch entry: Isaac Sim plus PhysX. Old checkpoints rejected."""
import argparse,json,os,signal,sys,time,hashlib,subprocess
from pathlib import Path
from .config import TrainCfg
os.environ.setdefault('OMNI_KIT_ACCEPT_EULA','YES')

def main():
    from isaaclab.app import add_launcher_args,launch_simulation
    from isaaclab.app.sim_launcher import _make_physics_cfg
    from isaaclab.sim import SimulationCfg,SimulationContext,RigidBodyMaterialCfg
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--usd',required=True);parser.add_argument('--resume');parser.add_argument('--continue-run',action='store_true')
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--smoke-iterations',type=int,default=8)
    parser.add_argument('--probe-only',action='store_true');parser.add_argument('--probe-steps',type=int,default=1000)
    add_launcher_args(parser);args=parser.parse_args()
    cfg=TrainCfg.from_record(json.loads(Path(args.config).read_text()));cfg.validate(production=not args.smoke)
    code_root=Path(__file__).resolve().parents[1]
    version=subprocess.run(['git','-C',str(code_root),'rev-parse','HEAD'],capture_output=True,text=True)
    code_commit=version.stdout.strip() if version.returncode==0 else None
    if not args.smoke:
        dirty=subprocess.run(['git','-C',str(code_root),'status','--porcelain'],capture_output=True,text=True)
        if code_commit is None or dirty.returncode or dirty.stdout.strip():raise RuntimeError('Formal launch requires a clean versioned release')
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA','YES')
    import torch
    torch.manual_seed(cfg.seed);torch.set_num_threads(8)
    torch.set_float32_matmul_precision('high')
    physics=_make_physics_cfg('isaacsim_physx')
    # Exact pinned Go2RLLab PhysX preset; default patch buffers overflow with
    #4096 robots on the complete direct-mesh terrain during actual diagnostics.
    physics.gpu_max_rigid_patch_count=2*1024*1024
    physics.gpu_collision_stack_size=512*1024*1024
    physics.enable_external_forces_every_iteration=True
    sim_cfg=SimulationCfg(dt=cfg.physics_dt,device=cfg.device,render_interval=4,physics=physics,physics_material=RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=1.,restitution=0.))
    with launch_simulation(sim_cfg,args):
        from .physx_env import PhysXEnv
        from .driver import MoeCtsDriver,FlatExpert
        from .learner import Learner
        sim=SimulationContext(sim_cfg);env=PhysXEnv(cfg,sim,args.usd)
        driver=MoeCtsDriver(env,cfg.driver_policy) if cfg.driver_frac else None
        out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
        from .physx_env import tensor
        masses=tensor(env.robot.data.body_mass).sum(-1)
        contract={'physics_backend':type(env.robot).__module__,'usd_path':args.usd,'usd_sha256':hashlib.sha256(Path(args.usd).read_bytes()).hexdigest(),
                  'joint_names':list(env.robot.joint_names),'body_names':list(env.robot.body_names),'torch':torch.__version__,
                  'total_mass_range_kg':[float(masses.min()),float(masses.max())],'context_dim':5,'wm_state_dim':70,'wm_input_dim':176,'wm_output_dim':73,'teacher_input_dim':167,'critic_input_dim':169,
                  'student_input_dim':450,'shared_actor_input_dim':77,'policy_hz':50,'physics_hz':200,
                  'motor':{k:v.tolist() for k,v in vars(env.consts).items()},'payload_model':'rigid point mass at existing base COM, sampled only on reset','payload_max_kg':cfg.payload_max_kg}
        contract['curriculum_clock']='learner iterations; collect1; not PPO24-step equivalent'
        contract['robot']=cfg.robot
        contract['reward_height_target_m']=env.profile.reward_height
        contract['terrain']=None if env.field is None else {'allocation':env.terrain_lookup.metadata,'collision_height_map':env.field.metadata}
        contract['actual_base_mass_range_kg']=[float(env.actual_base_mass_kg.min()),float(env.actual_base_mass_kg.max())]
        contract['reward_scope']='baseline13 plus pinned body-height and vertical-velocity definitions/courses; not full official15-term reward'
        contract['physx_preset']={k:getattr(physics,k) for k in ('gpu_max_rigid_patch_count','gpu_collision_stack_size','enable_external_forces_every_iteration')}
        (out/'physics_contract.json').write_text(json.dumps(contract,indent=2))
        identity={'pid':os.getpid(),'started_unix':time.time(),'backend':type(env.robot).__module__,'mode':'smoke' if args.smoke else 'formal','command':sys.argv,'code_commit':code_commit,'config_sha256':hashlib.sha256(Path(args.config).read_bytes()).hexdigest()}
        (out/'process.json').write_text(json.dumps(identity,indent=2))
        if args.probe_only:
            from .validation import physical_probe
            result=physical_probe(env,driver,args.probe_steps)
            (out/'probe.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
            return
        expert=FlatExpert(cfg.expert_policy,cfg.device) if cfg.expert_frac else None
        if args.smoke and cfg.stage=='terrain':
            import runpy
            audit=runpy.run_path(str(code_root/'tests/terrain_runtime_audit.py'))['audit_environment'](env)
            (out/'terrain_audit.json').write_text(json.dumps(audit,indent=2))
        learner=Learner(cfg,env,out,driver,expert)
        if args.resume:learner.load(args.resume,args.continue_run)
        def stop(signum,frame):
            learner.stop_requested=True
        learner.stop_requested=False
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        if args.smoke:
            from .validation import learner_smoke
            result=learner_smoke(learner,args.smoke_iterations)
            (out/'smoke.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
        else:
            learner.run();learner.export(out/'student_latest.pt')

if __name__=='__main__':main()

```

### launch.py · SHA256 964ae1addb21f36a754aab5e81f50a1bfb10b213e95c02eb74c967477e923206

```python
"""Separate robot selection, explicit stage and isolated output; no implicit launch."""
import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from .config import TrainCfg

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--robot',required=True,choices=('go2','as2'))
    p.add_argument('--stage',choices=('flat','terrain'),default='flat')
    p.add_argument('--output',required=True)
    p.add_argument('--usd')
    p.add_argument('--resume')
    p.add_argument('--driver',default='')
    p.add_argument('--flat-expert',default='')
    p.add_argument('--evaluation-python',default='')
    p.add_argument('--iterations',type=int,default=60000)
    p.add_argument('--print-config',action='store_true')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    cfg_fn=TrainCfg.terrain if a.stage=='terrain' else TrainCfg
    cfg=cfg_fn(robot=a.robot,iterations=a.iterations,driver_policy=a.driver,
               expert_policy=a.flat_expert,evaluation_python=a.evaluation_python,
               reset_critic=bool(a.resume and a.stage=='terrain'))
    cfg.validate()
    usd=Path(a.usd) if a.usd else root/'assets'/a.robot/('usd/go2.usd' if a.robot=='go2' else 'as2.usd')
    if not usd.is_file():raise FileNotFoundError(usd)
    if a.print_config:
        print(json.dumps(dict(config=asdict(cfg),usd=str(usd)),indent=2));return
    # Same lock as the existing supervisor: cannot overlap the current Go2 run.
    import fcntl
    lock_path=Path.home()/'Documents/go2_supermimic_lab/physx_training.lock'
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
        conf=out/'config.json';conf.write_text(json.dumps(asdict(cfg),indent=2))
        command=[sys.executable,'-m','go2_physx_cts.train','--config',str(conf),'--usd',str(usd),'--output',str(out)]
        if a.resume:command+=['--resume',a.resume]
        raise SystemExit(subprocess.call(command,cwd=root,env=dict(os.environ,PYTHONPATH=str(root))))

if __name__=='__main__':main()

```

### supervise.py · SHA256 cd8112b480527bff3e37e6f9af14199cbc99230c809040d799b58a6b9e556607

```python
"""Authorized mixed-terrain stage from the measured flat baseline; no time cap."""
import argparse,json,os,subprocess,sys,time,fcntl,hashlib
from pathlib import Path
from dataclasses import asdict
from .config import TrainCfg

def main():
    p=argparse.ArgumentParser()
    for name in ('output','usd','driver','resume','evaluation-python'):p.add_argument('--'+name,required=True)
    p.add_argument('--actor-microbatch',type=int,default=8192)
    p.add_argument('--iterations',type=int,default=60000)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    lock_path=Path.home()/'Documents/go2_supermimic_lab/physx_training.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        checkpoint=Path(a.resume).resolve()
        cfg=TrainCfg.terrain(driver_policy=a.driver,expert_policy=str(checkpoint),
            actor_microbatch=a.actor_microbatch,payload_max_kg=2.,
            reset_critic=True,evaluation_python=a.evaluation_python,iterations=a.iterations)
        cfg.validate(production=True)
        stage_out=out/'terrain';stage_out.mkdir()
        conf=stage_out/'config.json';conf.write_text(json.dumps(asdict(cfg),indent=2))
        command=[sys.executable,'-m','go2_physx_cts.train','--config',str(conf),
                 '--output',str(stage_out),'--usd',a.usd,'--resume',str(checkpoint)]
        state={'supervisor_pid':os.getpid(),'started_unix':time.time(),'physics':'isaacsim_physx',
               'hard_wall_clock_cap':None,'stage':'terrain','status':'launching',
               'parent_checkpoint':str(checkpoint),'parent_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
               'initial_iteration_review':cfg.iterations,'final_velocity_stage_iteration':45000,
               'review_reason':'60000 covers all official velocity stages plus15000 iterations at final range; review native and transfer evidence separately',
               'command':command}
        with (stage_out/'console.log').open('w') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            state.update(status='running_terrain',learner_pid=child.pid)
            (out/'supervisor.json').write_text(json.dumps(state,indent=2))
            code=child.wait()
        state.update(last_exit_code=code,ended_unix=time.time(),status='failed' if code else 'iteration_review_required')
        (out/'supervisor.json').write_text(json.dumps(state,indent=2))

if __name__=='__main__':main()

```
