"""Time/geometry feasibility only: perfect tracking is not a physics success claim."""
import unittest
from unittest.mock import patch
import torch
from go2_physx_cts.commands import OfficialCommands,VelocityGrid,special_speed_contract,COMMAND,GRID_BIN
from go2_physx_cts import task
from go2_physx_cts.calibration import build_budget

class Reachability(unittest.TestCase):
    def test_dynamic_contract(self):
        eight=special_speed_contract(8.);six=special_speed_contract(6.)
        self.assertAlmostEqual(eight['max_distance_m'],3.83)
        self.assertEqual(eight['minimum_speed_mps'],.6)
        self.assertEqual(eight['initial_extent_mps'],1.)
        self.assertEqual(six['minimum_speed_mps'],.8)
        self.assertEqual(special_speed_contract(4.)['minimum_speed_mps'],1.4)
        with self.assertRaises(ValueError):special_speed_contract(2.)

    def test_all_levels_minimum_speed_leaves_stability_time(self):
        for episode in (8.,6.,4.):
            contract=special_speed_contract(episode);v=contract['minimum_speed_mps']
            for level in range(10):
                gap_start=6.-(.2+.07*level)-1.4
                gap_distance=8.4-.8-gap_start
                for distance,stability in ((gap_distance,.64),(1.8,.32),(1.6,.32)):
                    self.assertLessEqual(.5+distance/v+stability,episode)

    def test_initial_distribution_and_failed_frontier_cannot_make_deadline_unreachable(self):
        grid=VelocityGrid();n=4096;tasks=torch.full((n,),3);terrains=torch.full((n,),10)
        command,bins=grid.sample(tasks,terrains,5000)
        self.assertGreaterEqual(float(command[:,0].min()),.6-1e-6)
        self.assertLessEqual(float(command[:,0].max()),1.+1e-6)
        self.assertTrue((command[:,1:]==0).all())
        self.assertTrue((grid.grid[bins,0]>=.75).any())
        for iteration in (6000,7000,8000):
            grid.record(tasks,terrains,bins,torch.zeros(n),torch.ones(n,dtype=torch.bool));grid.update(iteration)
            command,bins=grid.sample(tasks,terrains,iteration)
            self.assertGreaterEqual(float(command[:,0].min()),.6-1e-6)
            self.assertGreaterEqual(float(grid.extent[3,10,0]),1.)
        short=VelocityGrid(special_episode_seconds=4.)
        with self.assertRaises(ValueError):short.sample(tasks,terrains,5000)

    def test_minimum_speed_reaches_hard_gap_region_and_finishes_timer(self):
        # Synthetic perfect motion with stable support: tests episode-clock and
        # success-region compatibility, not the ability to cross a real gap.
        n=10;levels=torch.arange(n);speed=special_speed_contract(8.)['minimum_speed_mps']
        start=6.-(.2+.07*levels)-1.4
        s=torch.zeros(n,70);s[:,0]=.3;s[:,3]=-1.;s[:,4]=speed
        pose=torch.zeros(n,7);pose[:,0]=start;pose[:,2]=.3;pose[:,6]=1.
        goal=torch.zeros(n,3);goal[:,0]=8.4
        context=task.initial_task_context(pose,goal,episode_limit=8.,target_radius=.8,stable_seconds=.64)
        a=torch.zeros(n,12);command=torch.tensor([speed,0.,0.]).expand(n,-1)
        c=task.RobotConsts(torch.zeros(12),torch.ones(12),torch.ones(12),torch.ones(12),torch.ones(12),torch.tensor([[-3.,3.]]*12))
        telemetry={'applied_torque':a,'joint_acceleration':a,'foot_velocity':torch.zeros(n,4,3),'log_contact_force':torch.zeros(n,8),'contact_prob':torch.cat((torch.ones(n,4),torch.zeros(n,10)),-1)}
        measured=build_budget([{'net_return':50.,'qualified_complete':True}],{'policy_sha256':'test-fixture','robot':'as2'})
        success_time=torch.zeros(n);paid=torch.zeros(n,dtype=torch.long)
        with patch.object(task,'_BUDGET',measured):
            for step in range(1,401):
                nxt=pose.clone();nxt[:,0]=start+max(0.,step*.02-.5)*speed
                _,_,context,failure,success=task.task_reward(s,s,a,a,command,c,task_id=torch.full((n,),3),terrain_id=torch.full((n,),10),context=context,pose0=pose,pose=nxt,telemetry=telemetry,previous2=a,height_above_crop=s[:,0],variant='events')
                success_time=torch.where(success.bool()&(success_time==0),torch.full_like(success_time,step*.02),success_time)
                paid+=success.long();pose=nxt
        self.assertTrue((paid==1).all());self.assertTrue((success_time>0).all())
        self.assertLessEqual(float(success_time.max()),8.)
        self.assertAlmostEqual(float(success_time.max()),7.52,places=4)

if __name__=='__main__':unittest.main()
