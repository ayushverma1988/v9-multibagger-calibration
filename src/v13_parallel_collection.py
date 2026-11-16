"""Bounded source-rate-limited parallel collection with a rate-limit stop.

The Store still spaces requests by host. Parallelism overlaps network waits;
it does not bypass provider throttling. On 429/403 no new jobs are submitted.
"""
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import time
import threading
import re


def source_access_stop(result):
    """Respect actual 403/429 responses, never digits in a source URL.

    Current collectors retain requests/urllib exception text. Prefer a typed
    HTTP status when supplied, otherwise parse its status prefix after removing
    URLs. Explicit provider stop reasons remain supported without a status.
    """
    status = result.get('http_status') or result.get('status_code')
    if re.fullmatch(r'[4-5][0-9]{2}', str(status).strip()):
        return int(status) in (403, 429)
    error = re.sub(r'https?://\S+', '', str(result.get('error') or ''), flags=re.I)
    match = re.match(r'\s*(?:HTTP(?:/\d(?:\.\d)?)?\s+(?:Error\s+)?)?([1-5][0-9]{2})\b', error, re.I)
    if match:
        return int(match.group(1)) in (403, 429)
    return bool(re.match(r'\s*(?:error:\s*)?(?:too many requests|forbidden)\b', error, re.I)
                or re.search(r'\btoo many (?:403|429) error responses\b', error, re.I))

def bounded_map(jobs,worker,workers=4,seconds=600):
    started=time.monotonic();iterator=iter(jobs);pending={};stopped=False
    launch_lock=threading.Lock();last_launch=[0.]
    def invoke(item):
        with launch_lock:
            delay=max(0,last_launch[0]+.25-time.monotonic())
            if delay:time.sleep(delay)
            last_launch[0]=time.monotonic()
        return worker(item)
    # More in-flight requests can overlap slow replies without increasing
    # the existing one-launch-per-250ms ceiling or bypassing access stops.
    limit=max(1,min(int(workers),8))
    with ThreadPoolExecutor(max_workers=limit) as pool:
        def submit():
            if stopped or time.monotonic()-started>=seconds:return False
            try:item=next(iterator)
            except StopIteration:return False
            pending[pool.submit(invoke,item)]=item;return True
        for _ in range(limit):submit()
        while pending:
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                item=pending.pop(future)
                result=future.result()
                if source_access_stop(result):stopped=True
                yield item,result
            for _ in range(len(done)):submit()
