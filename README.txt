Go2 PhysX CTS / source-aligned SuperMimic port

Implementation reference: https://github.com/Zhengtianjin/go2-supermimic
Pinned commit: d9eee9b3f248e5300e636c433255db2c1c3e086a
UPSTREAM.json binds the downloaded source files to verified Git blob SHA1 and SHA256.

This is an independent implementation. It does not import the retired Newton
SMCTS package. Required physics is Isaac Lab3 with Isaac Sim6.2 / PhysX. The
reference repository uses mjlab/MuJoCo Warp; its physics results do not certify
this port.

Training: python -m go2_physx_cts.supervise --output RUN_DIRECTORY --usd GO2_USD
          --driver OFFICIAL_MOE_TORCHSCRIPT --evaluation-python ROBOGAUGE_PYTHON
All paths must identify real assets. The supervisor uses an exclusive file lock.
Training remains stopped under the current evaluation-only authorization.
Official sim2sim measurements require native/transfer review before curriculum
or rollback decisions. An evaluation alone cannot mark the flat curriculum ready.

Mechanism checks: python -m unittest discover -s tests -v
Physical/update smoke: python -m go2_physx_cts.train --config CONFIG_JSON
                      --output SCRATCH --usd GO2_USD --smoke
Official/standing positive control: same command plus --probe-only.

The production contract is4096 envs, collect1, actor1, replay1000/env,
8192 imagined starts x32steps, WM2048 windows x8steps,2 WM updates on flat and4
on terrain. Teacher/student imagination allocation is6144/2048 with one full
batch mean (0.75/0.25), no extra averaging by branch. Both sample all providers.
Only teacher encoder and shared actor are optimized by imagined return. The
student encoder receives latent distillation on real student-source histories.
The frozen critic has physical inputs, not actor-owned latent inputs. Freezing
parameters does not cut state/action input gradients.

Replay contains every provider's real executed actions and episode context.
There is no expert-action cloning loss and no separate WM-only reference pool.
Flat online provider: mixed-terrain MoE10%; source flat default has no driver.
Terrain recipe: mixed-terrain MoE10% plus this task's validated flat expert10%.
The latter provides velocity skills, not demonstrated jumping/ducking skills.

Task adaptations: body-frame(vx,vy,wz) commands, blind45x10 CTS history,
167-D privileged teacher,169-D physical/branch critic,70-D model state,
176-D WM input /73-D output. The fifth context entry is fixed0..1kg rigid payload
at the original base COM, sampled only at episode reset. Actions are linear,
unsquashed12-D joint targets with source PD/motor scales. The deployed export
requires the corresponding default pose, scales and PD in physics_contract.json.

The nominal contact coefficient is1 and COM randomization is zero in this first
port. Source sliding/spinning/rolling friction and COM randomization are not
claimed reproduced: PhysX contact parameters differ from MuJoCo. Encoder-bias
randomization is also not claimed. The terrain recipe's sizes and proportions
are ported to Isaac Lab generators; its triangles are not byte-identical mjlab
meshes. A differentiable height map is raycast from the actual PhysX mesh.

The source's WM, autoregressive loss, TD(lambda), soft survival, rare-window
selection/archive, running normalization, OU noise, actor LR ramp, optimizer
settings and clips are reproduced. CTS creates an additional20x1024 student
update schedule. Actual timing includes it.

There is no wall-time cutoff. The training callback now exports a frozen
checkpoint and invokes the isolated pinned RoboGauge runtime. After recording
the official measurements it stops for native/transfer review: the retired
custom-MuJoCo scores no longer control rollback or terrain promotion.

Checkpoints include all four models/optimizers, normalization and RNG metadata.
Like the source, resume starts new simulator episodes and fresh replay; it is
not a bit-identical full-simulator/replay continuation. Old checkpoints are
intentionally incompatible. No training capability or B0 is claimed by smoke.

Sim2sim reference: https://github.com/wty-yy/RoboGauge
Pinned commit: add2c7ee0185ea8fd93d242f86e1c8cac6e9913b
Runtime: Python3.11.18, MuJoCo3.2.3, dm-control1.0.23, Torch2.4.1+cu124,
NumPy1.26.4. Full dependency lock and measured comparisons are in
research/runs/ROBOGAUGE_MIGRATION_20261029. Do not use the training runtime's
MuJoCo3.12.0. robogauge_source.json verifies every upstream archive file.

Export in the training Python:
python -m go2_physx_cts.robogauge_export --checkpoint CHECKPOINT
       --output EXPORT_DIRECTORY --iteration ITERATION
Evaluate in the isolated official Python, with this repository on PYTHONPATH:
python -m go2_physx_cts.robogauge_bridge --exports EXPORT_DIRECTORY
       --output REPORT_DIRECTORY --iterations ITERATION
       --video

BasePipeline.run and MujocoSimulator load/step/PD/reset are unmodified upstream.
The extensions provide a stateful CTS model ABI, simulator-only teacher inputs,
fixed3D velocity goals and raw metrics. Student receives45 inputs, keeps450-D
term-major history internally and embeds frozen normalization. Teacher receives
45+167 values and is not a deployable blind policy. Exported outputs/history/reset
are verified against checkpoint golden sequences in both Torch runtimes.

The only control contract reads PD/default pose/action scale from the checkpoint.
Missing or malformed constants are errors; no fallback to official control defaults.
The official XML mass, COM, inertia, contacts, joint passive properties, limits
and motor ctrlrange remain unchanged. Experimental parameter switches are removed.
No inertial rewriting or private simulator/PD loop remains. Training USD nominal
mass16.087kg differs from official XML15.206408kg; torque-speed limits, contact
solvers and initialization also differ. Do not call these physically identical.

The default fixed-command protocol disables optional official noise/delay for
diagnosis; --official-noise-delay restores both. Both protocols retain official
startup settling. Neither constitutes full RoboGauge benchmark certification.
