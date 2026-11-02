"""Exercise real adapter methods without loading the simulator."""
import copy
import unittest
from types import SimpleNamespace
import torch
from go2_physx_cts.physx_env import PhysXEnv
from go2_physx_cts.commands import task_assignments
from go2_physx_cts.task_allocation import BASE_COUNTS,redistribute


class Transfers(unittest.TestCase):
    def env(self):
        e=PhysXEnv.__new__(PhysXEnv);e.num_envs=4096;e.device='cpu';e.cfg=SimpleNamespace(seed=17);e.iteration=25000
        e.task_ids=task_assignments(4096,25000,'cpu');e.terrain_ids=torch.zeros(4096,dtype=torch.long)
        e.terrain_levels=torch.zeros_like(e.terrain_ids);e._assign_terrains()
        e.terrain_levels[e.task_ids!=6]=3;e.recovery_kind=torch.full_like(e.task_ids,-1)
        e.pending_task=torch.full_like(e.task_ids,-1);e.pending_terrain=e.pending_task.clone();e.pending_level=e.pending_task.clone();e.allocation_target=None
        return e

    def test_queue_then_only_reset_ids_change(self):
        e=self.env();before=e.task_ids.clone();target=redistribute(list(BASE_COUNTS),[3,6],[0,1,2])
        e.request_task_allocation(target);self.assertTrue(torch.equal(e.task_ids,before))
        ids=(e.pending_task>=0).nonzero().flatten();first=ids[::2];second=ids[1::2]
        for i in ids:
            t=int(e.pending_task[i]);self.assertEqual(int(e.pending_level[i]),5 if t==6 else 3)
        e._apply_task_transfers(first)
        self.assertTrue(torch.equal(e.task_ids[second],before[second]))
        self.assertTrue((e.pending_task[first]==-1).all());self.assertTrue((e.pending_task[second]>=0).all())
        e._apply_task_transfers(second);self.assertEqual(e.allocation_status()['actual'],target)
        self.assertEqual(e.allocation_status()['pending'],0)
        self.assertEqual(torch.bincount(before,minlength=10).tolist(),list(BASE_COUNTS))

    def test_pending_checkpoint_fields(self):
        e=self.env();e._phase=3;e._terrain_max_levels=torch.zeros(14)
        class Grid:
            def state_dict(self):return {'grid':17}
            def load_state_dict(self,s):assert s=={'grid':17}
        e.command_sampler=Grid();e.ids=torch.arange(4096);e._place_arenas=lambda ids:None
        target=redistribute(list(BASE_COUNTS),[3],[0,1]);e.request_task_allocation(target)
        saved=copy.deepcopy(e.curriculum_state_dict());pending=e.pending_task.clone()
        e._apply_task_transfers(e.ids);e.load_curriculum_state_dict(saved)
        self.assertTrue(torch.equal(e.pending_task,pending));self.assertEqual(e.allocation_target,target)
        e._apply_task_transfers(e.ids);self.assertEqual(e.allocation_status()['actual'],target)

    def test_overlapping_requests_and_wrong_floor_rejected(self):
        e=self.env();target=redistribute(list(BASE_COUNTS),[3],[0,1]);e.request_task_allocation(target)
        with self.assertRaises(ValueError):e.request_task_allocation(target)
        e=self.env();target=list(BASE_COUNTS);target[0]+=166;target[1]-=166
        with self.assertRaises(ValueError):e.request_task_allocation(target)


if __name__=='__main__':unittest.main()
