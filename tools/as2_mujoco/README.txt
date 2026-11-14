Independent As2 MuJoCo multi-terrain evaluation

This tool ports the validated AS2_MUJOCO_20261110 experiment into the repository.
It is independent of training and does not add an automatic As2 training callback.
The existing Go2 training evaluator is unchanged.

Pinned reference: RoboGauge add2c7ee0185ea8fd93d242f86e1c8cac6e9913b.
Evaluation runtime: Python3.11, torch2.4.1+cu124, MuJoCo3.2.3, dm-control1.0.23.
MuJoCo must remain below3.3.0; the tool verifies the pinned runtime/source files.
Training runtime is used only for CPU checkpoint export and terrain preparation.
Existing assets/as2/official/as2_description supplies the meshes and training URDF.
official_as2.xml is the previously verified official-derived snapshot; its hash
and the original scripts are recorded in PORT_MANIFEST.json. The fixed-foot mass
and inertia are already merged in its calf inertias. No weights are bundled.

From the repository root, run:

python3 tools/as2_mujoco/run.py --checkpoint runs/AS2_BLIND_WINDOWED_s17_20261103_224658_175/model_00120000.pt --output /absolute/new/report/directory

Use a NEW output directory. Nothing overwrites old results or original weights.
Use --check for a38-case acceptance suite (all ten terrains,seed17,0.5m/s,both
branches), or omit it for the original354-case suite. Both include real MuJoCo
dynamics, not pose replay. Use --iteration for another checkpoint iteration;
the stored checkpoint iteration must agree. --train-python and --eval-python
override the existing runtimes under ~/Documents/go2_supermimic_runtime/.

Full suite: levels2/5, seeds17/29/43, speeds0.5/1/1.5m/s, plus flat4.2/5m/s.
Each episode is at most12s. Non-flat finishes at x=3.5m before terrain scan
crosses the tile edge; flat lasts12s. Startup settling remains official.
No payload, added observation noise or action delay. First1s is excluded from
tracking means. Failures always remain in success-rate denominators.

Original As2 checkpoint control is mandatory: PD40/40/60,Kd2/2/3, torque50/50/75,
action0.3125, armature0.026/0.026/0.038, joint damping/frictionloss0. Loaded
parameters/order/sensors are checked and saved per case. MuJoCo does not have
the training hard joint velocity limit; exceedance is measured, not hidden.
Contact penetration truncations remain enabled and are separate from falls.

Outputs: execution.json, export/geometry/runtime verification, summary.json,
all_results.json, all_cases.csv, per-case loaded_contract.json, metrics.json,
trajectory.npz, configs and logs. Summary means may include pre-failure samples;
always consider survival/course completion alongside tracking errors.

The policy is frozen. Teacher has privileged terrain rays; student is blind.
This is a project-specific As2 adapter using the official RoboGauge pipeline,
not an author-certified As2 benchmark or a claim of real-robot performance.
