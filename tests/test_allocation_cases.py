import copy
import tempfile
import unittest
from pathlib import Path
from go2_physx_cts.allocation_evaluation import plan_cases
from go2_physx_cts.task_allocation import TERRAIN_WEIGHTS


class Cases(unittest.TestCase):
    def test_complete_deterministic_cohort_and_course_identity(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'terrain.xml';p.write_text('<mujoco/>')
            manifest={'maps':[]}
            for task,terrains in enumerate(TERRAIN_WEIGHTS):
                for terrain in terrains:
                    for kind in range(6 if task==6 else 1):
                        row=dict(task_id=task,terrain_id=terrain,level=5 if task==6 else 0,xml=str(p),label=f'kind{kind}')
                        if task==6:row['recovery_qpos']=[kind]*19
                        manifest['maps'].append(row)
            a,missing=plan_cases(20000,manifest,{'physics':'fixed'})
            b,_=plan_cases(25000,manifest,{'physics':'fixed'})
            self.assertFalse(missing);self.assertEqual(len(a),500)
            self.assertEqual([r['case_id'] for r,m in a],[r['case_id'] for r,m in b])
            self.assertEqual(len({r['case_id'] for r,m in a}),500)
            changed,_=plan_cases(30000,manifest,{'physics':'fixed'})
            for task in range(10):
                same=[r['case_id'] for r,m in b if r['task_id']==task]==[r['case_id'] for r,m in changed if r['task_id']==task]
                self.assertEqual(same,task not in (0,1))
            m=copy.deepcopy(manifest);m['maps']=[r for r in m['maps'] if not (r['task_id']==6 and r['terrain_id']==8)]
            _,missing=plan_cases(25000,m,{})
            self.assertEqual(missing,[dict(task_id=6,terrain_id=8,reason='missing_exact_map')])


if __name__=='__main__':unittest.main()
