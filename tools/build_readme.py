"""Rebuild the single project manual from prose plus executable parameter sources."""
from pathlib import Path
import ast,hashlib,json,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]

INTRO='''# Go2 / As2 · PhysX + SuperMimic + CTS

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
'''

def build():
    parts=[INTRO]
    tree=ast.parse((ROOT/'go2_physx_cts/config.py').read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='TrainCfg')
    parts+=['|字段|平地默认|\n|---|---|\n']
    for x in cls.body:
        if isinstance(x,ast.AnnAssign):parts.append(f'|`{x.target.id}`|`{ast.unparse(x.value)}`|\n')
    parts.append('\n地形覆盖值与参数检查逻辑见config完整源码。所有CLI参数见launch/train/supervise源码；未使用参数不宣称生效。\n')
    for robot,urdf in [('Go2',ROOT/'assets/go2/urdf/go2.urdf'),('As2',ROOT/'assets/as2/official/as2_description/urdf/as2.urdf')]:
        doc=ET.parse(urdf).getroot(); rows=[]
        for link in doc.findall('link'):
            rows.append({'link':link.get('name'),'inertial':ET.tostring(link.find('inertial'),encoding='unicode') if link.find('inertial') is not None else None,'collisions':[ET.tostring(c,encoding='unicode') for c in link.findall('collision')]})
        rows += [{'joint':j.attrib,'definition':ET.tostring(j,encoding='unicode')} for j in doc.findall('joint')]
        parts.append(f'\n## {robot}完整资产物理参数\n\n```json\n'+json.dumps(rows,indent=2,ensure_ascii=False)+'\n```\n')
    for file in ['assets/as2/asset_parameters.json','assets/as2/runtime_check.json','assets/go2/runtime_check.json','assets/manifest.json','TRAINING_RUNTIME.json','ROBOGAUGE_RUNTIME.json','verification.json']:
        p=ROOT/file
        if p.exists():parts.append(f'\n## 实测或依赖记录：{file}\n\n```json\n'+p.read_text()+'\n```\n')
    parts.append('\n## 可执行机制与全部固定参数源码附录\n\n以下直接取自当前代码，包含未暴露为CLI的阈值、常数和完整计算；SHA256绑定对应文件。正文是解释，附录用于逐项核对，不是另一套配置。\n')
    for file in ['config','robots','networks','moe','state','world_model','buffer','imagine','learner','task','commands','terrain','terrain_scene','official_terrain_heightfield','official_terrain_mesh','physx_env','train','launch','supervise']:
        path=ROOT/'go2_physx_cts'/f'{file}.py';raw=path.read_bytes()
        parts.append(f'\n### {path.name} · SHA256 {hashlib.sha256(raw).hexdigest()}\n\n```python\n'+raw.decode().replace('\r\n','\n')+'\n```\n')
    (ROOT/'README.md').write_text(''.join(parts),encoding='utf8')

if __name__=='__main__':build()
