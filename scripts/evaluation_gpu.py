"""Choose an observed idle evaluation GPU, keeping the explicit fallback."""
import csv
import io
import subprocess
import time


def select_evaluation_gpu(fallback, allowed_indices=None):
    result = dict(selected_gpu=fallback, fallback_gpu=fallback,
                  selected_idle=False, time_unix=time.time())
    try:
        inventory = subprocess.check_output([
            'nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True, timeout=10)
        processes = subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
            '--format=csv,noheader,nounits'], text=True, timeout=10)
        occupied = {row[0].strip() for row in csv.reader(io.StringIO(processes)) if row}
        rows = []
        for row in csv.reader(io.StringIO(inventory)):
            index, uuid, memory, utilization = [part.strip() for part in row]
            rows.append(dict(index=int(index), uuid=uuid, memory_mib=int(memory),
                             utilization_percent=int(utilization), has_process=uuid in occupied))
        result['inventory'] = rows
        eligible = [r for r in rows if (allowed_indices is None or r['index'] in allowed_indices) and not r['has_process'] and
                    r['memory_mib'] < 256 and r['utilization_percent'] < 5]
        if eligible:
            chosen = min(eligible, key=lambda r: (r['memory_mib'], r['index']))
            result.update(selected_gpu=chosen['index'], selected_idle=True)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        result['query_error'] = repr(error)
    result['scope'] = 'Observation at launch, not an exclusive GPU reservation.'
    return result
