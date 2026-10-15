"""Bounded source-rate-limited parallel collection with a rate-limit stop.

The Store still spaces requests by host. Parallelism overlaps network waits;
it does not bypass provider throttling. On 429/403 no new jobs are submitted.
"""
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import time
import threading

def bounded_map(jobs,worker,workers=4,seconds=600):
    started=time.monotonic();iterator=iter(jobs);pending={};stopped=False
    launch_lock=threading.Lock();last_launch=[0.]
    def invoke(item):
        with launch_lock:
            delay=max(0,last_launch[0]+.25-time.monotonic())
            if delay:time.sleep(delay)
            last_launch[0]=time.monotonic()
        return worker(item)
    with ThreadPoolExecutor(max_workers=max(1,min(int(workers),4))) as pool:
        def submit():
            if stopped or time.monotonic()-started>=seconds:return False
            try:item=next(iterator)
            except StopIteration:return False
            pending[pool.submit(invoke,item)]=item;return True
        for _ in range(max(1,min(int(workers),4))):submit()
        while pending:
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                item=pending.pop(future)
                result=future.result()
                error=result.get('error')
                if error and any(z in error.lower() for z in ('429','too many requests','403','forbidden')):stopped=True
                yield item,result
            for _ in range(len(done)):submit()
