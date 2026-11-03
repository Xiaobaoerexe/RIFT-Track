import unittest,torch
from go2_physx_cts.task import reward,RobotConsts,WEIGHTS
class Reward(unittest.TestCase):
 def track(self,v,cmd,flat):
  s=torch.zeros(1,70);s[:,0]=.38;s[:,3]=-1;s[:,4]=v
  s.requires_grad_(True)
  c=RobotConsts(torch.zeros(12),torch.ones(12)*.25,torch.ones(12)*25,torch.ones(12),torch.ones(12)*24,torch.tensor([[-3.,3.]]*12))
  a=torch.zeros(1,12)
  r,terms,_=reward(s,s,a,a,torch.tensor([[cmd,0.,0.]]),c,height_above_crop=s[:,0],aux=torch.zeros(1,40),previous_previous=a,on_flat=torch.tensor([flat]))
  return s,terms['track_xy']
 def test_preserved_other_terrain_and_low_speed(self):
  for cmd in (.5,1.,2.,3.,4.2):
   s,r=self.track(.2,cmd,False)
   self.assertAlmostEqual(float(r.detach()),float(torch.exp(torch.tensor(-(.2-cmd)**2/.25))),places=6)
  for cmd in (.5,1.,2.):
   _,a=self.track(.2,cmd,False);_,b=self.track(.2,cmd,True);self.assertTrue(torch.equal(a,b))
 def test_high_speed_gradient_and_maximum(self):
  s,a=self.track(2.7,4.2,False);ga=torch.autograd.grad(a,s)[0][0,4]
  s,b=self.track(2.7,4.2,True);gb=torch.autograd.grad(b,s)[0][0,4]
  self.assertGreater(float(gb),float(ga)*10);self.assertGreater(float(gb),0)
  _,r=self.track(4.2,4.2,True);self.assertAlmostEqual(float(r.detach()),1.)
  self.assertEqual(len(WEIGHTS),15)
if __name__=='__main__':unittest.main()
