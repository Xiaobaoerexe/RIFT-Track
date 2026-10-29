"""Check the two imported rewards against pinned source, preserving baseline13.

python tests/test_reward_curriculum.py --source-dir /pinned/source/go2_rl_lab/go2_rl_lab/tasks/mixed \
    --baseline-task /saved/pre-change/task.py --output result.json
Synthetic formula tests; real terrain-ray agreement requires the PhysX smoke.
"""
import argparse
import ast
import __future__
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from go2_physx_cts import state as st
from go2_physx_cts import task

SOURCE_DIR = Path(os.environ['GO2RLLAB_REWARD_SOURCE']) if os.environ.get('GO2RLLAB_REWARD_SOURCE') else None
BASELINE_TASK = Path(os.environ['GO2_REWARD_BASELINE_TASK']) if os.environ.get('GO2_REWARD_BASELINE_TASK') else None
torch.set_num_threads(2)


class SceneEntity:
    def __init__(self, name, **kwargs):
        self.name = name
        self.__dict__.update(kwargs)


def source_functions(path, names, extra=None):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise AssertionError('Pinned source function set differs.')
    namespace = {'torch': torch, 'SceneEntityCfg': SceneEntity, **(extra or {})}
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(path), 'exec', flags=__future__.annotations.compiler_flag), namespace)
    return namespace


class RewardCurriculum(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if SOURCE_DIR is None:
            raise unittest.SkipTest('Run this source-comparison suite explicitly with --source-dir.')
        if not (SOURCE_DIR / 'env_cfg.py').is_file():
            raise RuntimeError('Provide the pinned Go2RLLab tasks/mixed source directory.')
        cls.source = source_functions(SOURCE_DIR / 'mdp/rewards.py',
            ['_get_base_height', '_velocity_command_b', 'lin_vel_z_l2', 'base_height_l2',
             'track_lin_vel_xy_exp', 'track_ang_vel_z_exp'],
            {'get_cropped_ray_hit_heights': lambda env, *args: env.crop_hits})
        cls.course = source_functions(SOURCE_DIR / 'mdp/curriculums.py', ['gradual_reward_weight_modification'])

    def fixture(self, n=64, dtype=torch.float32):
        torch.manual_seed(917)
        s = torch.randn(n, 70, dtype=dtype) * .2
        s[:, 0] = .2 + torch.rand(n, dtype=dtype) * .3
        roll = torch.linspace(-.7, .6, n, dtype=dtype)
        pitch = torch.linspace(.6, -.7, n, dtype=dtype)
        s[:, 1:4] = torch.stack((pitch.sin(), -pitch.cos() * roll.sin(), -pitch.cos() * roll.cos()), -1)
        s[:, 4:7] = torch.randn(n, 3, dtype=dtype)
        # Independent inverse-pitch then inverse-roll matrix, not the function under test.
        cp, sp, cr, sr = pitch.cos(), pitch.sin(), roll.cos(), roll.sin()
        matrix = torch.stack((cp, torch.zeros_like(cp), -sp,
                              sr * sp, cr, sr * cp,
                              cr * sp, -sr, cr * cp), -1).reshape(n, 3, 3)
        body_velocity = (matrix @ s[:, 4:7, None]).squeeze(-1)
        command = torch.randn(n, 3, dtype=dtype)
        crop_hits = torch.randn(n, 15, dtype=dtype) * .03 + .1
        root_pos = torch.zeros(n, 3, dtype=dtype)
        root_pos[:, 2] = s[:, 0] + .1
        env = SimpleNamespace(scene={'robot': SimpleNamespace(data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=root_pos), root_lin_vel_b=SimpleNamespace(torch=body_velocity),
            root_ang_vel_b=SimpleNamespace(torch=s[:, 7:10])))},
            command_manager=SimpleNamespace(get_command=lambda name: command), crop_hits=crop_hits)
        c = task.RobotConsts(torch.zeros(12, dtype=dtype), torch.ones(12, dtype=dtype) * .25,
            torch.ones(12, dtype=dtype) * 25, torch.ones(12, dtype=dtype) * .5,
            torch.ones(12, dtype=dtype) * 24, torch.tensor([[-3., 3.]] * 12, dtype=dtype))
        return s, command, crop_hits, root_pos, env, c

    def test_official_formulas_and_fixed_tracking_width(self):
        s, command, crop, pos, env, c = self.fixture()
        a = torch.randn(len(s), 12)
        height = pos[:, 2] - crop.mean(-1)
        _, terms, _ = task.reward(s, s, a, a * .8, command, c, iteration=750, height_above_crop=height)
        expected = {
            'lin_vel_z_l2': self.source['lin_vel_z_l2'](env),
            'base_height_l2': self.source['base_height_l2'](env, .38, sensor_cfg=SceneEntity('height_scanner')),
            'track_xy': self.source['track_lin_vel_xy_exp'](env, .5, 'base_velocity'),
            'track_yaw': self.source['track_ang_vel_z_exp'](env, .5, 'base_velocity')}
        for name, value in expected.items():
            self.assertTrue(torch.allclose(terms[name], value, atol=2e-6, rtol=2e-6), name)
        self.assertGreater(float((s[:, 6].square() - terms['lin_vel_z_l2']).abs().max()), .1)

    def test_source_course_parameters_and_boundary_clock(self):
        tree = ast.parse((SOURCE_DIR / 'env_cfg.py').read_text(encoding='utf-8'))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'CurriculumCfg')
        parameters = {}
        for node in cls.body:
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            kw = {item.arg: item.value for item in node.value.keywords}
            if 'params' in kw:
                values = ast.literal_eval(kw['params'])
                if values.get('term_name') in ('lin_vel_z_l2', 'base_height_l2'):
                    parameters[values['term_name']] = values
        self.assertEqual(set(parameters), {'lin_vel_z_l2', 'base_height_l2'})
        for iteration in (-1, 0, 1, 749, 750, 1499, 1500, 1501, 2499, 2500, 4999, 5000, 5001, 40000):
            for name, params in parameters.items():
                cfg = SimpleNamespace(weight=None)
                manager = SimpleNamespace(get_term_cfg=lambda term: cfg, set_term_cfg=lambda term, value: None)
                env = SimpleNamespace(common_step_counter=iteration * 24 + 23, reward_manager=manager)
                expected = self.course['gradual_reward_weight_modification'](env, None, **params)
                actual = task.reward_weights(iteration)[name]
                self.assertAlmostEqual(actual, expected, places=12)
                self.assertAlmostEqual(float(task.reward_weights(torch.tensor(float(iteration)))[name]), expected, places=5)
        self.assertEqual(task.reward_weights(0)['track_xy'], 2.)
        self.assertEqual(task.reward_weights(0)['track_yaw'], 1.)

    def test_all_existing_coefficients_unchanged(self):
        expected = {'track_xy': 2., 'track_yaw': 1., 'upright': 1., 'tilt_rate_l2': -.05,
            'hard_landing': -.5, 'crouch': -30., 'joint_deviation_l2': -.1, 'base_height_low': -10.,
            'terminated': -200., 'action_rate_l2': -.05, 'dof_pos_limits': -1.,
            'joint_torques_l2': -2e-5, 'joint_acc_l2': -2.5e-7}
        self.assertEqual(task.WEIGHTS, expected)
        for stage in ('flat', 'terrain'):
            for iteration in (0, 750, 1500, 2500, 5000, 40000):
                weights = task.reward_weights(iteration, stage)
                self.assertEqual(len(weights), 15)
                for name, value in expected.items():
                    self.assertEqual(weights[name], -600. if name == 'terminated' and stage == 'terrain' else value)

    def test_crop_geometry_definition_and_gradient(self):
        class Field:
            def sample(self, xy):
                return .17 * xy[..., 0].square() + .08 * xy[..., 1].square() + .03 * xy[..., 0] * xy[..., 1]
        field = Field()
        s, _, _, _, _, _ = self.fixture(2, torch.float64)
        pose = torch.tensor([[.3, -.2, .1], [-.4, .5, .7]], dtype=torch.float64, requires_grad=True)
        s.requires_grad_(True)
        self.assertTrue(torch.autograd.gradcheck(lambda ss, pp: task.crop_height_from_map(ss, pp, field), (s, pose)))
        self.assertTrue(torch.equal(task.crop_height_from_map(s, pose, None), s[:, 0]))
        self.assertEqual(len(task.BASE_HEIGHT_CROP_X) * len(task.BASE_HEIGHT_CROP_Y), 15)
        # Flat-gradient limit and yaw-sensitive curved terrain both remain connected.
        gradient = torch.autograd.grad(task.crop_height_from_map(s, pose, field).sum(), (s, pose))
        self.assertTrue(torch.equal(gradient[0][:, 0], torch.ones(2, dtype=torch.float64)))
        self.assertGreater(float(gradient[1][:, 2].abs().max()), 1e-6)

    def test_baseline_term_values_and_total_delta(self):
        if BASELINE_TASK is None:
            self.skipTest('Provide --baseline-task to compare preserved13 terms against pre-change code.')
        spec = importlib.util.spec_from_file_location('go2_physx_cts._reward_audit_baseline', BASELINE_TASK)
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        s1, command, crop, pos, _, c = self.fixture()
        s0 = s1 + .002 * torch.randn_like(s1)
        a, previous = torch.randn(len(s1), 12), torch.randn(len(s1), 12)
        height = pos[:, 2] - crop.mean(-1)
        for stage in ('flat', 'terrain'):
            for soft in (False, True):
                old_r, old_terms, old_done = legacy.reward(s0, s1, a, previous, command, c, stage, soft)
                for iteration in (0, 750, 1500, 5000):
                    new_r, terms, done = task.reward(s0, s1, a, previous, command, c, stage, soft,
                                                    iteration=iteration, height_above_crop=height)
                    self.assertTrue(torch.equal(done, old_done))
                    for name in old_terms:
                        self.assertTrue(torch.equal(terms[name], old_terms[name]), name)
                    weights = task.reward_weights(iteration, stage)
                    delta = .02 * (weights['lin_vel_z_l2'] * terms['lin_vel_z_l2'] + weights['base_height_l2'] * terms['base_height_l2'])
                    self.assertTrue(torch.allclose(new_r, old_r + delta, atol=3e-6, rtol=2e-6))

    def test_missing_terrain_geometry_and_height_shape_rejected(self):
        s, command, _, _, _, c = self.fixture()
        a = torch.zeros(len(s), 12)
        with self.assertRaises(ValueError):
            task.reward(s, s, a, a, command, c, 'terrain')
        with self.assertRaises(ValueError):
            task.reward(s, s, a, a, command, c, height_above_crop=s[:, :1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--baseline-task', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    SOURCE_DIR, BASELINE_TASK = args.source_dir, args.baseline_task
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RewardCurriculum))
    record = {'status': 'PASS' if result.wasSuccessful() else 'FAIL', 'tests': result.testsRun,
              'skipped': len(result.skipped), 'scope': 'Source function and configuration comparison on synthetic tensors; not locomotion evidence.',
              'source': task.OFFICIAL_REWARD_SOURCE,
              'hashes': {name: hashlib.sha256((SOURCE_DIR / name).read_bytes()).hexdigest()
                         for name in ('env_cfg.py', 'mdp/rewards.py', 'mdp/curriculums.py')}}
    if BASELINE_TASK:
        record['baseline_task_sha256'] = hashlib.sha256(BASELINE_TASK.read_bytes()).hexdigest()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
