"""AS2 two-cohort allocator with task0 subgroup confinement."""
from types import SimpleNamespace
import unittest
import torch
from go2_physx_cts.terrain_allocation import TerrainAllocator,redistribute
from go2_physx_cts.physx_env import PhysXEnv
from go2_physx_cts.commands import TERRAIN_NAMES

class Allocation(unittest.TestCase):
    def test_source4096_conservation_floor_cap_matching(self):
        base=[410]*9+[406];counts=base[:]
        for _ in range(20):
            result=redistribute(counts,[0,1],[2,3,4],base)
            self.assertEqual(sum(result),4096)
            self.assertTrue(all(200<=a<=2*b for a,b in zip(result,base)))
            self.assertEqual(result[5:],counts[5:]);counts=result
        allocator=TerrainAllocator(base);self.assertEqual(allocator.minimum,200)
        report={str(i):dict(valid=True,cohort_id='fixed',rate=.3 if i<2 else .9) for i in range(10)}
        self.assertFalse(allocator.observe(20000,report,base)['changed'])
        self.assertTrue(allocator.observe(25000,report,base)['changed'])
        for value in report.values():value['cohort_id']='changed'
        self.assertFalse(allocator.observe(30000,report,base)['changed'])

    def test_event_subgroup_scaled_floor_and_zero_terrain_exclusion(self):
        base=[82]*10+[0]*3;allocator=TerrainAllocator(base)
        self.assertEqual(allocator.minimum,40)
        report={str(i):dict(valid=True,cohort_id='fixed',rate=.3 if i==0 else .9) for i in range(13)}
        allocator.observe(20000,report,base)
        result=allocator.observe(25000,report,base)['target']
        self.assertEqual(sum(result),820);self.assertGreater(result[0],82)
        self.assertEqual(result[10:],[0,0,0]);self.assertTrue(all(40<=n<=164 for n in result[:10]))
        other=TerrainAllocator(base);other.load_state_dict(allocator.state_dict())
        self.assertEqual(other.state_dict(),allocator.state_dict())

    def env(self,variant):
        env=object.__new__(PhysXEnv);env.cfg=SimpleNamespace(variant=variant);env.device='cpu';env.iteration=25000
        env.task_ids=torch.tensor([0]*10+[1]*3+[2]*3+[3]*3+[4]*3)
        env.terrain_ids=torch.tensor([0]*5+[1]*5+[7]*6+[10,11,12]+[7]*3)
        env.terrain_names=TERRAIN_NAMES;env.num_envs=len(env.task_ids)
        env.terrain_levels=torch.zeros(env.num_envs,dtype=torch.long);env.ids=torch.arange(env.num_envs)
        env.pending_terrain=torch.full((env.num_envs,),-1,dtype=torch.long);env.pending_level=env.pending_terrain.clone()
        env.task_assignment_version=0;env.allocation_target=None
        return env

    def test_queue_only_moves_task0_and_applies_at_reset(self):
        env=self.env('events');before=env.task_ids.clone();terrain_before=env.terrain_ids.clone()
        target=env.allocation_counts();self.assertEqual(sum(target),10)
        target[0]-=2;target[1]+=2;self.assertTrue(env.queue_allocation(target))
        self.assertTrue((env.task_ids[env.pending_terrain>=0]==0).all())
        self.assertTrue(torch.equal(env.terrain_ids,terrain_before))
        env._apply_terrain_transfers(env.ids)
        self.assertEqual(env.allocation_counts(),target)
        self.assertTrue(torch.equal(env.task_ids,before));self.assertTrue(torch.equal(env.terrain_ids[10:],terrain_before[10:]))
        illegal=target[:];illegal[0]-=1;illegal[10]+=1
        with self.assertRaises(ValueError):env.queue_allocation(illegal)

    def test_simple_total_includes_all_terrains(self):
        env=self.env('simple');counts=env.allocation_counts()
        self.assertEqual(sum(counts),env.num_envs);self.assertEqual(counts[10:],[1,1,1])

if __name__=='__main__':unittest.main()
