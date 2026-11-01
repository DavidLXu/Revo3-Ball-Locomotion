"""Run unchanged native evaluations by profile, then merge their episode records."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('checkpoint');p.add_argument('--speed',type=float)
    p.add_argument('--seconds',type=float,default=15.);p.add_argument('--episodes',type=int,default=3)
    p.add_argument('--seed',type=int,default=20000)
    p.add_argument('--profiles',default='stop,forward,backward,left,right,diagonal,switch')
    p.add_argument('--stochastic',action='store_true');p.add_argument('--noise',type=float,default=1.)
    p.add_argument('--physics-timestep',type=float)
    p.add_argument('--target-slew-limit',type=float,help='Explicit target-slew controller diagnostic.')
    p.add_argument('--pd-kv',type=float,help='Explicit controller diagnostic; uses a separate worker and labels the result.')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--output',required=True)
    a=p.parse_args();assert 1<=a.workers<=8
    assert a.pd_kv is None or a.target_slew_limit is None, 'Test one controller intervention at a time'
    names=a.profiles.split(',')
    assert len(set(names))==len(names)
    assert all(n in {'stop','forward','backward','left','right','diagonal',
        'diagonal_forward_right','diagonal_backward_left','diagonal_backward_right','switch','rapid_switch'} for n in names)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    parts=out.with_suffix('.parts');parts.mkdir(exist_ok=True)
    script=Path(__file__).with_name('evaluate_teleop.py' if a.pd_kv is None else 'evaluate_damping_transfer.py')

    if a.target_slew_limit is not None:script=Path(__file__).with_name('evaluate_slew_transfer.py')

    def evaluate(name):
        path=parts/f'{name}.json'
        command=[sys.executable,str(script),a.checkpoint,'--seconds',str(a.seconds),
            '--episodes',str(a.episodes),'--seed',str(a.seed),'--profiles',name,
            '--noise',str(a.noise),'--output',str(path)]
        if a.speed is not None:command+=['--speed',str(a.speed)]
        if a.stochastic:command+=['--stochastic']
        if a.physics_timestep is not None:command+=['--physics-timestep',str(a.physics_timestep)]
        if a.pd_kv is not None:command+=['--pd-kv',str(a.pd_kv)]
        if a.target_slew_limit is not None:command+=['--target-slew-limit',str(a.target_slew_limit)]
        with path.with_suffix('.log').open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{name} evaluation failed; inspect {path.with_suffix(".log")}')
        report=json.loads(path.read_text())
        assert len(report['episodes'])==a.episodes
        assert all(r['profile']==name and r['seed']==a.seed+i for i,r in enumerate(report['episodes']))
        return report

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        reports=list(pool.map(evaluate,names))
    rows=[r for report in reports for r in report['episodes']]
    report=dict(reports[0])
    for other in reports[1:]:
        assert other.get('controller_override')==report.get('controller_override')
        for key in ('hand_inertia_mode','hand_mass_kg'):
            assert other.get(key)==report.get(key),key
        for key in ['checkpoint','training_speed_limit_mps','test_speed_mps','seconds',
                    'noise','engine','deterministic_policy','physics_timestep_s','control_timestep_s']:
            assert other[key]==report[key],key
    report.update(episodes=rows,completion_rate=sum(r['completed'] for r in rows)/len(rows),
        mean_velocity_error_mps=sum(r['mean_velocity_error_mps'] for r in rows)/len(rows),
        parallel_workers=a.workers)
    temp=out.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2));temp.replace(out)
    print('REPORT',out,flush=True)


if __name__=='__main__':main()
