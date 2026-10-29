import unittest
from go2_physx_cts.config import TrainCfg
from go2_physx_cts.robots import profile, as2_motor_values

class RobotContracts(unittest.TestCase):
    def test_old_config_is_go2(self):
        self.assertEqual(TrainCfg.from_record({'stage':'flat'}).robot,'go2')
    def test_as2_cannot_use_go2_driver_or_evaluator(self):
        for change in ({'driver_frac':.1,'driver_policy':'go2.pt'},
                       {'expert_frac':.1,'expert_policy':'go2.pt'},
                       {'evaluation_python':'python'}):
            with self.assertRaises(ValueError):TrainCfg(robot='as2',**change).validate()
    def test_as2_geometry_and_motors(self):
        self.assertIsNone(profile('as2').calf_length)
        self.assertEqual(profile('as2').left_hip,-.1)
        inertia,kp,kd,effort,scale=as2_motor_values('cpu')
        self.assertEqual(kp[:3].tolist(),[40.,40.,60.])
        self.assertEqual(effort[:3].tolist(),[50.,50.,75.])
        self.assertTrue((scale==.3125).all())
    def test_as2_flat_validates(self):TrainCfg(robot='as2').validate()

if __name__=='__main__':unittest.main()
