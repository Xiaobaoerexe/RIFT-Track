"""Storage placement must not change training samples, labels, or reset history."""
import unittest
import torch

from go2_physx_cts.buffer import Replay,Archive,PackedTransport,FLOAT_FIELDS
from go2_physx_cts.event_sampling import EventSeedArchive,sample_event_seeds


def transition(replay,step):
    values={key:torch.zeros((replay.n,value.shape[-1]),dtype=value.dtype,device=replay.device)
            for key,value in replay.data.items()}
    env=torch.arange(replay.n,device=replay.device)
    for key in FLOAT_FIELDS:
        values[key].fill_(step+.123)
    values['episode'][:,0]=step//7+1
    values['source'][:,0]=env%2
    values['task_ids'][:,0]=3+(env//2)%2
    values['scene_ids'][:,0]=env+100
    values['frame_index'].fill_(step)
    values['depth_frame_index'].fill_(step//10*10)
    values['depth_scene_ids'][:,0]=env+100
    values['next_contact_prob'].zero_();values['next_contact_prob'][:,:4]=1
    values['success'].fill_(step%7==5)
    values['done'].fill_(step%7==6);values['terminated'].fill_(step%7==6)
    return values


class StorageContracts(unittest.TestCase):
    def equal(self,left,right):
        self.assertEqual(set(left),set(right))
        for key in left:
            if isinstance(left[key],dict):self.equal(left[key],right[key])
            else:
                self.assertEqual(left[key].dtype,right[key].dtype,key)
                torch.testing.assert_close(left[key].cpu(),right[key].cpu(),rtol=0,atol=0,msg=key)

    def exercise(self,device):
        on_device=Replay(16,32,device,holdout_modulus=4)
        on_cpu=Replay(16,32,device,holdout_modulus=4,storage_device='cpu')
        events=[EventSeedArchive(16,history_steps=4,burnin=2) for _ in range(2)]
        for step in range(46):
            values=transition(on_device,step)
            for replay,event in zip((on_device,on_cpu),events):
                replay.add(**values);torch.manual_seed(step);event.observe(replay)
        self.equal(on_device.data,on_cpu.data)
        torch.testing.assert_close(on_device.event_history.cpu(),on_cpu.event_history)
        torch.testing.assert_close(on_device.heldout.cpu(),on_cpu.heldout)
        self.assertEqual(on_device.storage_bytes,on_cpu.storage_bytes)
        self.assertEqual(on_cpu.metadata()['storage_device'],'cpu')
        operations=(lambda r:r.seeds(48,source=1,burnin=8),
                    lambda r:r.seeds(48,source=0,burnin=8,validation=True),
                    lambda r:r.sequences(32,loss_steps=8,burnin=4,source=1),
                    lambda r:r.windows(32,8,event=True),
                    lambda r:r.windows(32,8,validation=True))
        for operation in operations:
            torch.manual_seed(987);first=operation(on_device)
            torch.manual_seed(987);second=operation(on_cpu)
            self.equal(first,second)
            self.assertTrue(all(v.device==on_device.device for v in second.values() if torch.is_tensor(v)))
        torch.manual_seed(49);first=sample_event_seeds(on_device,events[0],64,1,2)
        torch.manual_seed(49);second=sample_event_seeds(on_cpu,events[1],64,1,2)
        self.equal(first,second)
        self.assertEqual(events[0].size.tolist(),events[1].size.tolist())
        torch.manual_seed(12);windows=on_device.windows(24,8)
        archives=[Archive(20,8,device),Archive(20,8,device,storage_device='cpu')]
        for archive in archives:
            archive.add(windows,tick=30);archive.add({k:v[:12] for k,v in windows.items()},tick=40)
        torch.manual_seed(52);first=archives[0].sample(48)
        torch.manual_seed(52);second=archives[1].sample(48)
        self.equal(first,second)
        self.assertEqual(archives[0].statistics(50),archives[1].statistics(50))
        if device!='cpu':
            self.assertGreater(on_cpu.transport.bytes_h2d,0)
            self.assertGreater(on_cpu.transport.bytes_d2h,0)
            self.assertLessEqual(on_cpu.transport.metadata()['pinned_bytes'],16*1024*1024)

    def test_cpu_same_device_explicit_storage(self):self.exercise('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA transfer coverage runs on target servers')
    def test_cuda_target_cpu_storage_exact_sampling(self):self.exercise('cuda')

    def test_identity_does_not_allocate_staging(self):
        transport=PackedTransport(128);data={'a':torch.arange(32),'b':torch.ones(3,dtype=torch.bool)}
        self.assertIs(transport.move(data,'cpu'),data)
        self.assertEqual(transport.metadata()['pinned_bytes'],0)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA transfer coverage runs on target servers')
    def test_chunked_transport_exact_dtypes_and_staging_reuse(self):
        transport=PackedTransport(128)
        original={'a':torch.arange(1001,dtype=torch.int64),'b':torch.arange(57,dtype=torch.float32).reshape(3,19),
                  'half':torch.tensor([float('inf'),-0.,.125],dtype=torch.float16),
                  'bool':torch.arange(101)%3==0,'empty':torch.empty((0,3),dtype=torch.int16)}
        moved=transport.move(original,'cuda')
        # Immediate reuse exercises event protection, including different byte lengths.
        other=transport.move({'a':torch.arange(2003,dtype=torch.int32)},'cuda')
        self.equal(original,transport.move(moved,'cpu'))
        torch.testing.assert_close(other['a'].cpu(),torch.arange(2003,dtype=torch.int32))
        self.assertEqual(transport.metadata()['pinned_bytes'],128)


if __name__=='__main__':unittest.main()
