Maintained evidence tools for the137 Go2 environment

audit_physx.py: load the actual training USD and read initialized PhysX parameters.
evaluate_physx_evidence.py: frozen teacher/student native evaluation, with pose and slip traces.
interface_audit.py: compare actual loaded sensors, anatomical frames and checkpoint observation ABI.
runtime_validate.py: verify pinned dependencies and exported golden history/reset sequences.
render_native.py: render measured PhysX trajectories using official RoboGauge meshes; no dynamics.
video_qa.py: decode recorded videos and create inspection sheets.

These tools retain the explicit137 runtime/evidence paths for the frozen20261029
baseline. Run physical tools in the PhysX runtime; interface/runtime/render/video
tools in the isolated official RoboGauge runtime. They do not start training.
The standard policy export/evaluation APIs are go2_physx_cts.robogauge_export,
robogauge_bridge and robogauge_feedback. No parameter-profile selector remains.
