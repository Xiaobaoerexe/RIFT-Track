# Blind As2 window-feedback revision — 2026-11-03

Current launch: `python3 run_blind_as2.py` on server175. It launches only As2 from
fresh initialization,120000 iterations,no wall-clock cutoff,no external policies.
Shared GPU lock,clean Git checkout and exact launch/config identity are required.

## Preserved training contract
Isaac Sim/PhysX; three velocity commands; blind student,8-expert student encoder;
unchanged teacher/shared Actor/Critic/WM architectures.4096 real environments,
one vector collection step per iteration,8192 imagination paths x32;
6144 teacher/2048 student paths with teacher_loss.mean()+student_loss.mean().
WM2048x8,2 updates before2500 and4 afterward; CTS20x1024; Critic5x4.
Payload remains0..2kg at base COM,constant within each episode.
Both provider fractions are0 in the As2 configuration.

Warmup0..1000; Critic1000..1500; Actor ramp1500..2500.
Flat until2500; flat40% and three slopes20% each until5000; ten terrains afterward.
Speed stages5000/10000/20000/30000/35000 and official level rules are retained.

## Replaced mechanisms
Automatic weight rollback and single-step candidate-LR selection are removed.
Actor and teacher share one optimizer; their original ramp never restarts.
Every1000 iterations compares20 complete50-iteration windows. Changed validation
cohorts/curriculum/allocation contexts invalidate comparisons. Two successive
comparable deteriorating or recovering windows and at least2000 iterations
between changes are required. Adjustment factor1.3; no PPO KL controller.

WM2e-4 through40000,bounds[5e-5,2e-4]; afterward fixed heldout eight-step error
drives bounded reduction/recovery. Actor/teacher2e-4 after ramp,bounds[5e-5,2e-4].
Critic cap2e-4/floor5e-5. CTS3e-4 through40000,normal floor1e-4; later reduction
requires isolated action-matching deterioration with a stable teacher target,
or broad multi-terrain teacher/student stability. Student stability decay is
spaced by5000 iterations and does not require the cohort-refresh boundary.
The exact windows,values,decisions and reasons are stored in lr_windows.jsonl.
A nonfinite update stops with a diagnostic checkpoint; no automatic retry/rollback.

Critic loss is branch-equal2*Huber(delta=1),preserving MSE scale near zero.
CTS loss is latent MSE+.01 gate balance+.1*ramp*quality-weighted action Huber.
The new action term ramps2500..5000 and remains off on each terrain until fixed
teacher evaluation supports it:quality=clip((teacher_pass-.5)/.3,0,1).
It uses this experiment's student replay states. Teacher and Actor parameters
are frozen during CTS,while Actor input gradients reach the student encoder.

Flat commands above2m/s blend a broader velocity-tracking kernel:
alpha=.25*clip(abs(vx_command)-2,0,1);
tracking=(1-alpha)*exp(-XYerror2/.25)+alpha*exp(-XYerror2/1).
Other terrain and speeds<=2m/s keep the prior formula exactly; reward peak stays1.
All other official15 reward weights,As2 height target and ramps are unchanged.

## Fixed PhysX allocation evidence
Every5000 iterations evaluates paired teacher/student cases on levels2 and5,
payload0/max,5 fixed seeds,common velocity cases. Allocation uses only currently
unlocked commands; high-speed probes are recorded separately. The terrain score
is the worse branch's success rate. This is10-second tracking/stability validation,
not proof of a whole obstacle-course traversal or RoboGauge sim2sim.

20000 establishes allocation baseline;25000 is first possible redistribution.
Two comparable evaluations below50% make a recipient;both>=80% without a>=10pp
drop make a donor. Recipients gain up to10%,donors contribute evenly up to10%.
Neutral terrains stay unchanged;total4096,floor min(200,original count),
cap2xoriginal. Actual reassignment occurs only on each selected episode reset.
This adapts the depth allocator at b160cfeb7dee395bc2b54726c2853cfc4aa551dc.

The WM validation pool excludes64 fixed environments from gradient-supervised
WM window sampling/archive; normalization statistics still use all collected data.
Actor imagination may use all replay states. Cohorts/error scaling are fixed
inside comparison blocks and rebuilt at curriculum/allocation/5000 boundaries.

## Provenance and validation
Previous blind commit532e78de09ca23c875004fb2bbfd406073d741d1 preserved under
archive/go2-blind-120k-20261103.
SuperMimic reference d9eee9b3f248e5300e636c433255db2c1c3e086a.
Go2RLLab reference6c95d9feef4814c25eb6536e61bb38e1d1d718d4.
New scheduler,action distillation gating and high-speed blend are project
hypotheses,not claimed verbatim upstream implementations or proven capability.
As2 has no verified RoboGauge adapter;native results cannot certify sim2sim.

See reports/as2_revision_20261103/validation.json and physics_contract.json.
36 unit tests:34 pass,2 skip missing upstream mesh checkout;6 standalone
analytical timeout/gradient checks pass. Full4096-environment As2 smoke passed
24 mature iterations plus restore,export,diagnostic and allocation checks.
Last12 iterations average0.6255s,median0.6271s;GPU sample mean65%,29103.5MiB.
These short measurements estimate20.85h of updates for120k,excluding evaluations,
saving and possible later throughput changes. They do not demonstrate locomotion.

Formal run identity and release commit will be recorded in runs/as2_launch_*.json and the run process.json.
