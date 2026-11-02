"""Resource conservation, complete cohorts and delayed episode transfers."""
import copy
import random
import unittest
from go2_physx_cts.task_allocation import (BASE_COUNTS,SEEDS,PAYLOADS,TERRAIN_WEIGHTS,
    TaskAllocator,redistribute,summarize_cohort)


def report(it,values=None,cohort='fixed'):
    values=values or [.3,.9,.9,.4,.4,.6,.6,.6,.6,.6]
    return {'iteration':it,'tasks':{str(i):dict(valid=True,rate=v,cohort_id=cohort) for i,v in enumerate(values)}}


class Allocation(unittest.TestCase):
    def test_two_reports_start_25k(self):
        a=TaskAllocator()
        self.assertFalse(a.observe(report(20000),BASE_COUNTS)['changed'])
        r=a.observe(report(25000),BASE_COUNTS)
        self.assertTrue(r['changed']);self.assertEqual(sum(r['target']),4096)
        self.assertEqual(r['target'][1:3],[329,329])
        self.assertEqual(sum(r['target'][i]-BASE_COUNTS[i] for i in (0,3,4)),72)

    def test_missing_baseline_and_offcycle(self):
        a=TaskAllocator()
        self.assertIsNone(a.observe(report(19999),BASE_COUNTS))
        self.assertFalse(a.observe(report(25000),BASE_COUNTS)['changed'])
        self.assertIsNone(a.observe(report(26000),BASE_COUNTS))
        self.assertTrue(a.observe(report(30000),BASE_COUNTS)['changed'])
        self.assertIsNone(a.observe(report(30000),BASE_COUNTS))

    def test_threshold_and_donor_regression(self):
        a=TaskAllocator();a.observe(report(20000,[.49,.9,.8,.6,.6,.6,.6,.6,.6,.6]),BASE_COUNTS)
        r=a.observe(report(25000,[.5,.8,.8,.6,.6,.6,.6,.6,.6,.6]),BASE_COUNTS)
        self.assertEqual(r['states']['0'],'hold');self.assertEqual(r['states']['1'],'hold')
        self.assertEqual(r['states']['2'],'donor');self.assertFalse(r['changed'])

    def test_incomparable_missing_and_pending_hold(self):
        for kind in ('cohort','invalid','gap','pending'):
            a=TaskAllocator();a.observe(report(20000),BASE_COUNTS)
            r=report(30000 if kind=='gap' else 25000,cohort='changed' if kind=='cohort' else 'fixed')
            if kind=='invalid':
                for t in r['tasks'].values():t['valid']=False
            self.assertFalse(a.observe(r,BASE_COUNTS,pending=kind=='pending')['changed'])

    def test_save_resume(self):
        a=TaskAllocator();a.observe(report(20000),BASE_COUNTS)
        b=TaskAllocator();b.load_state_dict(a.state_dict())
        self.assertEqual(a.observe(report(25000),BASE_COUNTS),b.observe(report(25000),BASE_COUNTS))

    def test_conservation_floor_cap_equal_donation(self):
        rng=random.Random(17);counts=list(BASE_COUNTS)
        for _ in range(500):
            ids=list(range(10));rng.shuffle(ids);poor=ids[:rng.randrange(1,5)];good=[i for i in ids[5:] if i not in poor]
            new=redistribute(counts,poor,good)
            self.assertEqual(sum(new),4096);self.assertGreaterEqual(min(new),200)
            self.assertTrue(all(n<=2*b for n,b in zip(new,BASE_COUNTS)))
            self.assertTrue(all(counts[i]-new[i]<=int(counts[i]*.1) for i in good))
            self.assertTrue(all(new[i]-counts[i]<=int(counts[i]*.1+.5) for i in poor))
            counts=new
        self.assertEqual(redistribute(list(BASE_COUNTS),list(range(10)),[]),list(BASE_COUNTS))
        self.assertEqual(redistribute(list(BASE_COUNTS),[],list(range(10))),list(BASE_COUNTS))

    def test_invalid_counts(self):
        with self.assertRaises(ValueError):redistribute([409.6]*10,[],[])
        with self.assertRaises(ValueError):redistribute(BASE_COUNTS,[1],[1])

    def cohort(self):
        exp=[];rows=[]
        for t,weights in enumerate(TERRAIN_WEIGHTS):
            for terrain in weights:
                for branch in ('teacher','student'):
                    for s in SEEDS:
                        for p in PAYLOADS:
                            key=f'{t}/{terrain}/{branch}/{s}/{p}'
                            exp.append(dict(case_id=key,task_id=t,terrain_id=terrain,branch=branch,seed=s,payload_kg=p))
                            rows.append(dict(allocation_case_id=key,iteration=25000,cohort_role='task_allocation',task_id=t,
                                             xy_rmse=.1,yaw_rmse=.1,height=.3,**{'pass':branch=='teacher' or terrain==7}))
        return exp,rows

    def test_coverage_and_fixed_weight_min_branch(self):
        exp,rows=self.cohort();self.assertEqual(len(exp),500)
        result=summarize_cohort(25000,rows,exp)
        self.assertTrue(all(t['valid'] for t in result['tasks'].values()))
        self.assertEqual(result['tasks']['0']['rate'],.4)
        self.assertEqual(result['tasks']['6']['rate'],.5)
        older=[dict(r,iteration=20000) for r in rows]
        self.assertEqual(summarize_cohort(25000,rows+older,exp),result)

    def test_bad_results_do_not_masquerade_as_failure(self):
        exp,rows=self.cohort()
        for kind in ('missing','duplicate','error','nan','missing_map'):
            x=copy.deepcopy(rows);e=copy.deepcopy(exp)
            if kind=='missing':x.pop(0)
            if kind=='duplicate':x.append(x[0])
            if kind=='error':x[0]['error']='test'
            if kind=='nan':x[0]['height']=float('nan')
            if kind=='missing_map':e=[r for r in e if not (r['task_id']==0 and r['terrain_id']==7)];x=[r for r in x if r['allocation_case_id'] in {z['case_id'] for z in e}]
            result=summarize_cohort(25000,x,e)
            self.assertFalse(result['tasks']['0']['valid'],kind)
            self.assertIsNone(result['tasks']['0']['rate'])
            self.assertTrue(result['tasks']['1']['valid'])

    def test_real_early_fall_counts_as_failure(self):
        exp,rows=self.cohort()
        rows[0].update(survived=False,samples=0,survival_seconds=.4,xy_rmse=None,yaw_rmse=None,height=None,**{'pass':False})
        self.assertTrue(summarize_cohort(25000,rows,exp)['tasks']['0']['valid'])


if __name__=='__main__':unittest.main()
