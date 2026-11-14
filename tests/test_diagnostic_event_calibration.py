"""Native event-value reports use complete real episodes, not imagined labels."""
import unittest
import torch
from go2_physx_cts.diagnostics import event_value_calibration


def histories(n,t):
    return {key:torch.zeros((n,t),dtype=torch.float32 if key in ('prediction','reward') else torch.bool)
        for key in ('prediction','reward','valid','done','pre_completed','completed','success','failure','timeout')}


class EventCalibration(unittest.TestCase):
    def test_complete_mc_strata_and_short_episode_exclusion(self):
        h=histories(4,5);h['valid'][:]=True
        # Teacher crossing: one success, then real mode-correction rewards.
        h['reward'][0]=torch.tensor([0.,20.,-.2,-.2,-.2]);h['done'][0,4]=True
        h['success'][0,1]=True;h['completed'][0,1:]=True;h['pre_completed'][0,2:]=True
        h['timeout'][0,4]=True
        h['prediction'][0]=torch.tensor([19.4,19.4,-.6,-.4,-.2])+2
        # Student failure ends at t=1; malicious post-reset reward must not leak.
        h['reward'][1]=torch.tensor([0.,-5.,1000.,1000.,1000.]);h['done'][1,1]=True
        h['failure'][1,1]=True;h['prediction'][1,:2]=-5
        # Unfinished teacher recovery cannot supply an MC target, even with bonus.
        h['reward'][2,1]=30.;h['success'][2,1]=True
        # Student recovery observed through timeout.
        h['done'][3,4]=True;h['timeout'][3,4]=True;h['prediction'][3]=0
        report=event_value_calibration(h,torch.tensor([3,3,4,4]),torch.tensor([0,1,0,1]),gamma=1.)
        rows={(r['branch'],r['task_id']):r for r in report['rows']}
        cross=rows['teacher',3]
        self.assertEqual(cross['complete_episodes'],1)
        self.assertEqual(cross['groups']['all']['states'],5)
        self.assertAlmostEqual(cross['groups']['all']['mae'],2.,places=5)
        self.assertAlmostEqual(cross['groups']['all']['bias'],2.,places=5)
        self.assertAlmostEqual(cross['groups']['all']['correlation'],1.,places=6)
        self.assertEqual(cross['groups']['before_success_32']['states'],1)
        self.assertEqual(cross['groups']['at_success']['states'],1)
        self.assertEqual(cross['groups']['after_success']['states'],3)
        self.assertEqual(cross['groups']['at_timeout']['states'],1)
        fail=rows['student',3]
        self.assertEqual(fail['groups']['all']['states'],2)
        self.assertEqual(fail['groups']['all']['mae'],0.)
        self.assertEqual(fail['groups']['before_failure_32']['states'],1)
        self.assertIsNone(fail['groups']['all']['correlation'])
        unfinished=rows['teacher',4]
        self.assertEqual(unfinished['excluded_truncated_episodes'],1)
        self.assertEqual(unfinished['excluded_truncated_states'],5)
        self.assertEqual(unfinished['groups']['all']['states'],0)
        self.assertIsNone(unfinished['groups']['all']['mae'])
        self.assertIsNone(unfinished['hard_success_rate_complete'])

    def test_discounting_and_nonfinite_prediction_are_explicit(self):
        h=histories(1,4);h['valid'][:]=True;h['done'][0,3]=True;h['timeout'][0,3]=True
        h['reward'][0]=torch.tensor([0.,0.,0.,10.])
        h['prediction'][0]=torch.tensor([7.29,8.1,9.,float('nan')])
        report=event_value_calibration(h,torch.tensor([4]),torch.tensor([1]),gamma=.9)
        row=report['rows'][0]['groups']['all']
        self.assertEqual(row['states'],3);self.assertEqual(row['nonfinite_states'],1)
        self.assertLess(row['mae'],1e-6)

    def test_pre_success_window_is_bounded(self):
        h=histories(1,80);h['valid'][:]=True;h['done'][0,-1]=True;h['success'][0,60]=True
        report=event_value_calibration(h,torch.tensor([3]),torch.tensor([0]))
        self.assertEqual(report['rows'][0]['groups']['before_success_32']['states'],32)


if __name__=='__main__':unittest.main()
