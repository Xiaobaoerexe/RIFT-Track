"""CPU command regression, including execution of the pinned source sampler.

Run: python tests/test_official_commands.py --source /path/to/pinned/repo
No Isaac import, optimizer update, server access, or simulated capability claim.
"""
import argparse
import ast
import copy
import contextlib
import hashlib
import io
from itertools import product
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from go2_physx_cts import commands as c

DEFAULT_SOURCE = Path('C:/WINDOWS/Temp/go2_rl_lab_source_6c95d9fe/go2_rl_lab-6c95d9feef4814c25eb6536e61bb38e1d1d718d4')


class ScriptedTorch:
    def __init__(self, uniforms, integers):
        self.uniforms, self.integers = list(uniforms), list(integers)

    def __getattr__(self, name):
        return getattr(torch, name)

    def rand(self, size, **kwargs):
        value = self.uniforms.pop(0)
        assert len(value) == size
        return value.clone()

    def randint(self, low, high, shape, **kwargs):
        value = self.integers.pop(0)
        assert tuple(value.shape) == tuple(shape) and bool(((value >= low) & (value < high)).all())
        return value.clone()


def source_oracle(source, state, pos, iteration, u, lookup, episode_seconds=25.):
    """Execute original _resample and its helpers with controlled random draws."""
    path = source / 'source/go2_rl_lab/go2_rl_lab/cmd_utils/rl_gym_cmd.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == '6f1392c77229a6852a43d3eda36a75b09bc94441b0678d567bd7d5433537f242'
    tree = ast.parse(path.read_text())
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RLGymVelCommand')
    keep = {'_resample', '_apply_limit_commands', '_add_limit_yaw_to_zero_commands', '_get_current_scale',
            '_get_max_lin_vel', '_apply_terrain_command_limits', '_update_env_command_ranges',
            '_update_command_range_curriculum', '_update_limit_vel_curriculum', '_resolve_limit_vel_stage',
            '_build_limit_vel_comb'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    nodes += [n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name in keep]
    namespace = {'torch': torch, 'product': product, 'Sequence': list, '_COMMAND_NAMES': c.COMMAND_NAMES}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + nodes,
                            type_ignores=[])), str(path), 'exec'), namespace)
    oracle = type('PinnedSource', (), {k: namespace[k] for k in keep})()
    n = len(state)
    oracle.cfg = SimpleNamespace(resampling_time=5., command_area_length=8., dynamic_resample_commands=True,
        limit_vel_prob=.2, limit_vel_hold_prob=0., limit_vel_invert_when_continuous=True,
        limit_ang_vel_at_zero_command_prob=.2, num_steps_per_iter=1,
        zero_command_curriculum={'start_iter': 0, 'end_iter': 1500, 'start_value': 0., 'end_value': .1},
        command_range_curriculum=[{'iter': it, **dict(zip(c.COMMAND_NAMES, bounds))}
                                  for it, bounds in zip(c.RANGE_ITERATIONS, c.RANGE_STAGES[1:])][::-1],
        limit_vel_curriculum=[{'iter': 25000, 'lin_vel_x': [-1, 1], 'lin_vel_y': [0], 'ang_vel_yaw': [-1, 0, 1]},
                              {'iter': 0, 'lin_vel_x': [-1, 1], 'lin_vel_y': [-1, 1], 'ang_vel_yaw': [-1, 0, 1]}],
        command_clip_rules=[{'command': 'lin_vel_x', 'limit': 2., 'clip': {'lin_vel_y': (-.5, .5), 'ang_vel_yaw': (-1., 1.)}},
                            {'command': 'ang_vel_yaw', 'limit': 2., 'clip': {'lin_vel_x': (-2., 2.), 'lin_vel_y': (-1., 1.)}}])
    oracle._env = SimpleNamespace(common_step_counter=iteration, max_episode_length=round(episode_seconds/.02),
        max_episode_length_s=episode_seconds, episode_length_buf=state[:, c.EPISODE_ELAPSED] / .02, step_dt=.02)
    oracle.device='cpu';oracle.num_envs=n;oracle._terrain_lookup=lookup
    oracle.robot=SimpleNamespace(data=SimpleNamespace(root_pos_w=SimpleNamespace(torch=pos)))
    oracle.command_ranges=dict(zip(c.COMMAND_NAMES,c.RANGE_STAGES[0]));oracle._global_range_values=None
    oracle._env_command_ranges=torch.zeros(n,3,2)
    oracle.env_command_ranges={name:oracle._env_command_ranges[:,i] for i,name in enumerate(c.COMMAND_NAMES)}
    oracle._terrain_max_ranges=torch.tensor([c.TERRAIN_RANGES['flat']]+[c.TERRAIN_RANGES[name] for name in lookup.names])
    oracle.velocity_command_b=state[:,c.COMMAND].clone();oracle._requested_velocity_command_b=state[:,c.REQUEST].clone()
    oracle.commands_xy_accumulation=state[:,c.ACCUMULATION].clone();oracle.time_left=state[:,c.TIMER].clone()
    oracle.last_is_limit_vel=state[:,c.LAST_LIMIT].bool().clone();oracle.command_counter=state[:,c.COMMAND_COUNTER].clone()
    oracle.zero_command_prob=0.;oracle.limit_vel={};oracle.max_lin_vel=.5
    # All source arithmetic/control flow remains original. Only random streams
    # are assigned to the equivalent per-row draw slots of the functional port.
    stage=sum(iteration>=it for it in c.RANGE_ITERATIONS);bounds=torch.tensor(c.RANGE_STAGES[stage])
    remaining=(5.-state[:,c.ACCUMULATION].norm(dim=-1)*5.).clamp_min(0.)
    zero_time=(episode_seconds-state[:,c.EPISODE_ELAPSED]-remaining/(.8*bounds[:2].abs().max()+1e-9)).clamp(0,5.)
    limited=u[:,3]<.2;changed=limited & ~state[:,c.LAST_LIMIT].bool()
    zero=(u[:,3]>=.2)&(u[:,3]<.2+c.zero_command_prob(iteration))&(zero_time>0)
    add=zero&(u[:,5]<.2)
    uniforms=[u[:,i] for i in range(4)]
    if bool(zero.any()):uniforms.append(u[zero,5])
    if bool(add.any()):uniforms.append(u[add,6])
    integers=[(u[changed,4]*(6 if iteration>=25000 else 12)).long()] if bool(limited.any()) else []
    proxy=ScriptedTorch(uniforms,integers);namespace['torch']=proxy
    with contextlib.redirect_stdout(io.StringIO()):
        oracle._resample(torch.arange(n))
    assert not proxy.uniforms and not proxy.integers
    return oracle


def run(source):
    torch.manual_seed(4);checks=[]
    text=(source/'source/go2_rl_lab/go2_rl_lab/cmd_utils/rl_gym_cmd.py').read_text()
    cfg=next(node for node in ast.parse(text).body if isinstance(node,ast.ClassDef) and node.name=='RLGymVelCommandCfg')
    literal={}
    for node in cfg.body:
        if isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name):
            try:literal[node.target.id]=ast.literal_eval(node.value)
            except (ValueError,TypeError):pass
    assert tuple(stage['iter'] for stage in literal['command_range_curriculum'])==c.RANGE_ITERATIONS
    assert tuple(tuple(tuple(stage[name]) for name in c.COMMAND_NAMES) for stage in literal['command_range_curriculum'])==c.RANGE_STAGES[1:]
    assert {name:tuple(tuple(values[key]) for key in c.COMMAND_NAMES) for name,values in literal['terrain_max_command_ranges'].items()}==c.TERRAIN_RANGES
    assert literal['resampling_time']==5. and literal['limit_vel_prob']==.2 and literal['limit_vel_hold_prob']==0.
    assert c.zero_command_prob(0)==0 and c.zero_command_prob(750)==.05 and c.zero_command_prob(1500)==.1
    checks.append('pinned_config_literal_stages_terrain_limits_and_probabilities')
    lookup=c.GridTerrainLookup((0.,0.),(8.,8.),2,3,[0,1,2],('flat','stairs_up','wheels'))
    sampler=c.OfficialCommands(.02,25.,lookup)
    pos=torch.tensor([[1.,1.],[1.,9.],[1.,17.],[-1.,4.],[17.,3.],[8.,8.],[2.,12.],[5.,5.]])
    origins=torch.zeros_like(pos);u=torch.rand(8,8)
    # Exercise fresh boundary, consecutive boundary, zero, yaw-only-zero, and
    # ordinary dynamics in one batch, on in-grid and outside-grid positions.
    u[:,3]=torch.tensor([.05,.1,.22,.25,.9,.8,.7,.26]);u[2,5]=.1;u[3,5]=.9;u[7,5]=.1
    u[2,6]=.1;u[7,6]=.8
    base=pos.new_zeros((8,c.CMD_STATE_DIM));base[:,c.EPISODE_ELAPSED]=torch.arange(8).float()
    base[1,c.LAST_LIMIT]=1.;base[:,c.ACCUMULATION]=torch.tensor([.3,.4]);base[:,c.COMMAND_COUNTER]=2
    for iteration in (0,749,1500,4999,5000,14999,15000,24999,25000,35000,45000,50000):
        before=base.clone();oracle=source_oracle(source,base,pos,iteration,u,lookup)
        result=sampler._resample(base,torch.ones(8,dtype=torch.bool),pos,iteration,random=u)
        for actual,expected in ((result[:,c.REQUEST],oracle._requested_velocity_command_b),
            (result[:,c.COMMAND],oracle.velocity_command_b),(result[:,c.TIMER],oracle.time_left),
            (result[:,c.ACCUMULATION],oracle.commands_xy_accumulation),
            (result[:,c.LAST_LIMIT].bool(),oracle.last_is_limit_vel),(result[:,c.COMMAND_COUNTER],oracle.command_counter)):
            torch.testing.assert_close(actual,expected,rtol=2e-6,atol=2e-6)
        assert torch.equal(base,before)
    checks.append('original_pinned_resample_parity_12_iterations_all_sampling_branches')
    short=c.OfficialCommands(.02,20.,lookup)
    ending=base.clone();ending[:,c.EPISODE_ELAPSED]=torch.tensor([0.,5.,10.,15.,19.98,19.9,18.,12.])
    ending[:,c.ACCUMULATION]=0.
    oracle=source_oracle(source,ending,pos,5000,u,lookup,20.)
    actual=short._resample(ending,torch.ones(8,dtype=torch.bool),pos,5000,random=u)
    torch.testing.assert_close(actual[:,c.COMMAND],oracle.velocity_command_b,rtol=2e-6,atol=2e-6)
    torch.testing.assert_close(actual[:,c.TIMER],oracle.time_left,rtol=2e-6,atol=2e-6)
    checks.append('configured_20s_episode_remaining_time_and_late_zero_suppression')
    assert lookup.query(torch.tensor([[0.,0.],[15.99,23.99],[16.,0.],[0.,24.],[-.001,0.],[8.,8.]])).tolist()==[0,2,-1,-1,-1,1]
    checks.append('half_open_geometry_lookup_edges_and_outside_flat')
    # A raw high-speed request survives crossing into stairs and returning.
    state=base[:1].clone();state[:,c.REQUEST]=torch.tensor([4.,.9,2.9]);state[:,c.TIMER]=4.
    state[:,c.RANGE_STAGE]=4.;state[:,c.LIMIT_STAGE]=1
    clipped=sampler.advance(state,torch.tensor([[1.,9.]]),35000,random=u[:1])
    torch.testing.assert_close(clipped[:,c.COMMAND],torch.tensor([[1.,.9,1.5]]))
    restored=sampler.advance(clipped,torch.tensor([[1.,1.]]),35000,random=u[:1])
    torch.testing.assert_close(restored[:,c.COMMAND],torch.tensor([[4.,.5,1.]]))
    torch.testing.assert_close(restored[:,c.REQUEST],state[:,c.REQUEST])
    checks.append('raw_request_restored_after_terrain_crossing_and_priority_coupling')
    # Iteration stages apply only at a resampling event; boundary combination
    # removes lateral boundary requests at25k, without removing continuous vy.
    unchanged=sampler.advance(state,torch.tensor([[1.,1.]]),45000,random=u[:1])
    assert unchanged[0,c.RANGE_STAGE]==4
    state[:,c.TIMER]=0.;state[:,c.LAST_LIMIT]=0
    late=sampler.advance(state,torch.tensor([[1.,1.]]),45000,random=u[:1])
    assert late[0,c.RANGE_STAGE]==5 and late[0,1]==0
    checks.append('official_iteration_stages_lazy_activation_and_25k_boundary_combinations')
    initial=sampler.initial(pos,origins,0,random=u)
    mask=torch.tensor([True,False,False,False,False,False,False,False])
    initial[:,c.TIMER]=.5;initial[:,c.MAX_DISTANCE]=3.
    reset=sampler.reset(initial,mask,pos,origins,0,random=u)
    assert reset[0,c.COMMAND_COUNTER]==1 and reset[0,c.MAX_DISTANCE]==0 and reset[0,c.EPISODE_ELAPSED]==0
    torch.testing.assert_close(reset[1:],initial[1:])
    checks.append('selective_reset_preserves_other_episode_state')
    # Timer crosses zero at dt, episode time and max-distance advance, no hidden
    # mutation, and a copied replay command state reproduces the next step.
    initial[:,c.TIMER]=.01;replayed=initial.clone();u2=torch.rand(8,8)
    a=sampler.advance(initial,pos,1500,random=u2);b=sampler.advance(replayed,pos,1500,random=u2)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.testing.assert_close(a[:,c.EPISODE_ELAPSED],initial[:,c.EPISODE_ELAPSED]+.02)
    checks.append('packed_replay_roundtrip_next_transition_and_timing')
    # Gradients through continuous command history and physical displacement
    # remain available; only terrain IDs/discrete random choices are discrete.
    differentiable=base.clone().requires_grad_();xy=pos.clone().requires_grad_()
    ordinary=u.clone();ordinary[:,3]=.9
    out=sampler.advance(differentiable,xy,15000,random=ordinary)
    loss=out[:,c.COMMAND].square().sum()+out[:,c.MAX_DISTANCE].sum()
    loss.backward()
    assert bool(torch.isfinite(differentiable.grad).all()) and bool(torch.isfinite(xy.grad).all())
    assert differentiable.grad[:,c.ACCUMULATION].abs().sum()>0 and xy.grad.abs().sum()>0
    checks.append('continuous_history_and_position_autograd_finite_nonzero')
    # Exact generator checkpoint resume for the port's own random draw order.
    generator=torch.Generator().manual_seed(57);saved=generator.get_state()
    a=sampler.advance(initial,pos,1500,generator=generator);generator.set_state(saved)
    b=sampler.advance(initial,pos,1500,generator=generator);torch.testing.assert_close(a,b,rtol=0,atol=0)
    checks.append('generator_state_restore_reproduces_command_transition')
    compiled=torch.compile(sampler.advance,backend='eager',fullgraph=True)
    actual=compiled(initial,pos,1500,random=u2)
    torch.testing.assert_close(actual,sampler.advance(initial,pos,1500,random=u2),rtol=0,atol=0)
    checks.append('torch_compile_fullgraph_without_tensor_host_sync')
    result={'status':'PASS','kind':'CPU_code_source_parity_not_robot_capability','checks':checks,
            'source_commit':c.SOURCE_COMMIT,'cmd_state_dim':c.CMD_STATE_DIM}
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    run(parser.parse_args().source)
