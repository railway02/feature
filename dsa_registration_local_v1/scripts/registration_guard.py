#!/usr/bin/env python3
"""The sole owner of one Local Reference registration RUN_ID.

An advisory flock stays open for the guard's entire lifetime.  The actual
supervisor is deliberately a child process, so a C++/ANTs child crash cannot
drop the lock or create a second supervisor tree on retry.
"""
from __future__ import annotations
import argparse, fcntl, json, os, signal, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
child=None; monitor=None; stopping=False
def complete(out: Path) -> bool:
    try: return json.loads((out/'OVERNIGHT_RUN_SUMMARY.json').read_text()).get('status') == 'COMPLETE'
    except Exception: return False
def terminate(proc):
    if proc and proc.poll() is None:
        try: os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError: pass
def main():
    global child, monitor, stopping
    ap=argparse.ArgumentParser(); ap.add_argument('--run-id',required=True); ap.add_argument('--workers',type=int,default=4); a=ap.parse_args()
    out=ROOT/'outputs'/a.run_id; out.mkdir(parents=True,exist_ok=True); lock_path=out/'.registration.lock'
    lock=lock_path.open('a+')
    try: fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print(f'LOCK_HELD run_id={a.run_id}',flush=True); return 73
    lock.seek(0); lock.truncate(); lock.write(json.dumps({'run_id':a.run_id,'guard_pid':os.getpid(),'started_utc':time.strftime('%FT%TZ',time.gmtime())})+'\n'); lock.flush()
    def on_signal(*_):
        nonlocal_lock=None
        global stopping
        stopping=True; terminate(child); terminate(monitor)
    signal.signal(signal.SIGTERM,on_signal); signal.signal(signal.SIGINT,on_signal)
    env=os.environ.copy(); env.update({'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS':'4','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'})
    monlog=(out/'logs'/'resource_monitor.log'); monlog.parent.mkdir(parents=True,exist_ok=True)
    mf=monlog.open('a'); monitor=subprocess.Popen([str(ROOT/'scripts'/'resource_monitor.sh'),a.run_id,'rigid'],stdout=mf,stderr=subprocess.STDOUT,start_new_session=True,env=env)
    print(f'GUARD_ACQUIRED run_id={a.run_id} guard_pid={os.getpid()} monitor_pid={monitor.pid}',flush=True)
    while not stopping and not complete(out):
        child=subprocess.Popen([sys.executable,str(ROOT/'scripts'/'overnight_supervisor.py'),'--run-id',a.run_id,'--workers',str(a.workers)],stdout=sys.stdout,stderr=sys.stderr,start_new_session=True,env=env)
        rc=child.wait(); child=None
        if stopping or complete(out): break
        print(f'CHILD_EXIT rc={rc}; resume_same_run_id_after_10s',flush=True)
        for _ in range(10):
            if stopping: break
            time.sleep(1)
    terminate(monitor)
    if monitor:
        try: monitor.wait(timeout=15)
        except subprocess.TimeoutExpired: os.killpg(monitor.pid,signal.SIGKILL)
    lock.seek(0); lock.truncate(); lock.write(json.dumps({'run_id':a.run_id,'guard_pid':os.getpid(),'status':'stopped_or_complete','finished_utc':time.strftime('%FT%TZ',time.gmtime())})+'\n'); lock.flush()
    return 0
if __name__=='__main__': raise SystemExit(main())
