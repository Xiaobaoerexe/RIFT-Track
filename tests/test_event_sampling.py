"""Success archive retention, exact capture history and own-source partitions."""
import copy
import unittest
import torch
from go2_physx_cts.buffer import Replay
from go2_physx_cts.event_sampling import EventSeedArchive,sample_event_seeds


def fill(replay,step,*,success=False,episode=None,terminal=False):
    values={key:torch.zeros((replay.n,value.shape[-1]),dtype=value.dtype)
            for key,value in replay.data.items()}
    episode=step//8 if episode is None else episode
    values['episode'].fill_(episode)
    values['source'][:,0]=torch.arange(replay.n)%2
    values['task_ids'][:,0]=3+(torch.arange(replay.n)//2)%2
    values['frame_index'].fill_(step)
    values['depth_frame_index'].fill_(step//10*10)
    values['depth_scene_ids'][:,0]=torch.arange(replay.n)+100
    values['depth_pose'][:,0]=step
    values['obs'][:,0]=step
    values['s'][:,0]=step
    values['s_next'][:,0]=step+1
    values['next_contact_prob'][:,:4]=1
    values['success'].fill_(success);values['terminated'].fill_(terminal)
    values['done'].fill_(terminal)
    values['event_reward'].fill_(10 if success else (-5 if terminal else 0))
    replay.add(**values)


class EventSampling(unittest.TestCase):
    def replay(self):return Replay(8,32,'cpu',holdout_modulus=100003)

    def test_no_success_uniform_fallback_and_failure_near_pool(self):
        replay=self.replay();archive=EventSeedArchive(16,history_steps=4,burnin=2)
        for step in range(10):
            fill(replay,step,episode=1,terminal=step==8);archive.observe(replay)
        self.assertEqual(archive.total_admitted,0)
        seed=sample_event_seeds(replay,archive,64,source=1,burnin=2)
        self.assertEqual(len(seed['s']),64);self.assertTrue((seed['source']==1).all())
        self.assertEqual(archive.last_sample_stats['long_success_history'],0)
        self.assertGreater(archive.last_sample_stats['near_event'],0)
        self.assertEqual(int(archive.size.sum()),0)

    def test_archive_survives_ring_overwrite_and_exact_prefix(self):
        replay=self.replay();archive=EventSeedArchive(16,history_steps=4,burnin=2)
        for step in range(10):
            fill(replay,step,episode=1,success=step==8);archive.observe(replay)
        self.assertEqual(archive.total_admitted,4)
        for step in range(10,60):fill(replay,step,episode=2)
        seed=archive.sample(128,1,2,'cpu')
        self.assertTrue((seed['episode']==1).all())
        self.assertTrue(((seed['s'][:,0]>=4)&(seed['s'][:,0]<=8)).all())
        prefix=seed['prefix'];self.assertTrue(prefix['valid'].all())
        torch.testing.assert_close(prefix['obs'][:,:,0],seed['s'][:,0,None]-torch.tensor([2.,1.]))
        torch.testing.assert_close(seed['depth_pose'][:,0],seed['s'][:,0])
        self.assertTrue((seed['depth_scene_ids'][:,0]==seed['trajectory_ids']+100).all())
        self.assertTrue((seed['source']==1).all())
        self.assertTrue((seed['success'].reshape(-1)==(seed['s'][:,0]==8)).all())

    def test_early_success_masks_other_episodes(self):
        replay=self.replay();archive=EventSeedArchive(16,history_steps=4,burnin=2)
        for step in range(6):fill(replay,step,episode=1)
        fill(replay,6,episode=2,success=True);archive.observe(replay)
        seed=archive.sample(16,1,2,'cpu')
        self.assertTrue((seed['s'][:,0]==6).all());self.assertFalse(seed['prefix']['valid'].any())
        self.assertTrue(seed['prefix']['reset'].all())

    def test_only_success_and_no_heldout_enter(self):
        replay=self.replay();archive=EventSeedArchive(16,history_steps=4,burnin=2)
        fill(replay,0,episode=1,success=True)
        replay.heldout[:,(replay.ptr-1)%replay.L]=True
        self.assertEqual(archive.observe(replay),0)
        fill(replay,1,episode=2,terminal=True)
        self.assertEqual(archive.observe(replay),0)
        self.assertEqual(archive.storage_bytes,0)

    def test_mixture_counts_and_roundtrip(self):
        replay=self.replay();archive=EventSeedArchive(16,history_steps=4,burnin=2)
        for step in range(12):
            fill(replay,step,episode=1,success=step==8);archive.observe(replay)
        checkpoint=copy.deepcopy(archive.state_dict())
        restored=EventSeedArchive(16,history_steps=4,burnin=2);restored.load_state_dict(checkpoint)
        seed=sample_event_seeds(replay,restored,64,source=1,burnin=2)
        stats=restored.last_sample_stats
        self.assertEqual((stats['uniform'],stats['near_event'],stats['long_success_history']),(32,16,16))
        self.assertEqual(len(seed['prefix']['obs']),64)
        self.assertEqual(sum(row['states'] for row in stats['tasks'].values()),64)
        self.assertTrue((seed['source']==1).all())
        self.assertEqual(restored.total_admitted,archive.total_admitted)

    def test_bounded_even_with_many_successes(self):
        replay=self.replay();archive=EventSeedArchive(8,history_steps=4,burnin=2)
        for step in range(30):
            fill(replay,step,episode=step+1,success=True);archive.observe(replay)
        self.assertEqual(archive.size.tolist(),[2,2,2,2])
        self.assertEqual(archive.total_admitted,120)
        self.assertLess(archive.storage_bytes,1_000_000)
        with self.assertRaises(ValueError):archive.sample(1,1,3,'cpu')


if __name__=='__main__':unittest.main()
