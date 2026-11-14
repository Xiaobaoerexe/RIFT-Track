"""Aggregate real independent MuJoCo cases; keep failure categories explicit."""
from pathlib import Path
import argparse,csv,json,statistics,collections
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--workers',default='0,1,2');p.add_argument('--expected',type=int,required=True);a=p.parse_args()
 out=Path(a.output).resolve();rows=[]
 for worker in map(int,a.workers.split(',')):
  record=json.loads((out/f'results_{worker}.json').read_text());assert record['complete'];rows+=record['rows']
 assert len(rows)==a.expected,(len(rows),a.expected)
 def stats(rr):
  result={'cases':len(rr),'survived':sum(r['survived'] for r in rr),'tracking_passed':sum(r['tracking_pass'] for r in rr),
   'course_completed_alive':sum(r['course_completed'] and r['survived'] for r in rr),
   'failures':dict(collections.Counter(r['failure'] or ('warning' if r['warning'] else 'none') for r in rr))}
  for key in ('vx','vx_rmse','xy_rmse','yaw_rmse','height','slip_mps','joint_velocity_exceed_fraction'):
   values=[r[key] for r in rr if r.get(key) is not None];result[key]=statistics.mean(values) if values else None
  return result
 summary={'complete':True,'cases':len(rows),'training_updates':0,
  'branches':{b:stats([r for r in rows if r['branch']==b]) for b in ('teacher','student')},
  'terrains':{name:{b:stats([r for r in rows if r['terrain']==name and r['branch']==b]) for b in ('teacher','student')} for name in sorted(set(r['terrain'] for r in rows))}}
 (out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'all_results.json').write_text(json.dumps(rows,indent=2))
 with (out/'all_cases.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps(summary['branches']))
if __name__=='__main__':main()
