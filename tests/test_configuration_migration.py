import unittest
from go2_physx_cts.config import TrainCfg

class ConfigurationMigration(unittest.TestCase):
    def test_historical_metadata_does_not_restore_control_overrides(self):
        cfg=TrainCfg.from_record({'evaluation_xml':'obsolete.xml',
            'evaluation_profile':'official_pose','adaptive_iterations':True,
            'seed':17,'num_envs':4096})
        self.assertEqual(cfg.seed,17)
        self.assertFalse(hasattr(cfg,'evaluation_profile'))
        self.assertFalse(hasattr(cfg,'evaluation_xml'))

    def test_unknown_configuration_still_fails(self):
        with self.assertRaises(TypeError):
            TrainCfg.from_record({'unrecognized_control_override':1})

if __name__=='__main__':unittest.main()
