"""Native notebook run monitoring and embedded figures; no training replacement."""
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import subprocess
import threading
import time


def iniciar_monitor(directory):
    import psutil
    directory=Path(directory)
    stop=threading.Event()
    def sample():
        process=psutil.Process(os.getpid())
        previous={}
        fields=['utc','transcurrido_s','fase','modelo','epoca','cpu_procesos_pct','rss_bytes',
                'gpu_util_pct','gpu_mem_util_pct','gpu_mem_used_mib','gpu_temperatura_c']
        fields += [f'cpu_logico_{i:02d}_pct' for i in range(psutil.cpu_count())]
        begin=time.perf_counter()
        psutil.cpu_percent(percpu=True)
        with (directory/'recursos.csv').open('x',encoding='utf8',newline='') as f, \
             (directory/'monitor-gpu.log').open('x',encoding='utf8') as raw:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            while not stop.is_set():
                now=time.perf_counter();cpu=0.;rss=0
                for member in [process,*process.children(recursive=True)]:
                    try:
                        times=member.cpu_times();cumulative=times.user+times.system
                        if member.pid in previous:
                            old,stamp=previous[member.pid]
                            cpu+=(cumulative-old)/max(now-stamp,1e-6)*100
                        previous[member.pid]=(cumulative,now);rss+=member.memory_info().rss
                    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                try:state=json.loads((directory/'estado.json').read_text(encoding='utf8'))
                except (OSError,json.JSONDecodeError):state={}
                gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,utilization.memory,memory.used,temperature.gpu',
                                    '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
                raw.write(datetime.now(timezone.utc).isoformat()+' '+gpu.stdout.strip()+' '+gpu.stderr.strip()+'\n');raw.flush()
                values=gpu.stdout.strip().split(',') if gpu.returncode==0 else []
                row=dict(utc=datetime.now(timezone.utc).isoformat(),transcurrido_s=now-begin,fase=state.get('fase','inicio'),
                         modelo=state.get('modelo',''),epoca=state.get('epoca',''),cpu_procesos_pct=cpu,rss_bytes=rss)
                row.update({key:(float(values[i]) if len(values)>i else '') for i,key in enumerate(
                    ['gpu_util_pct','gpu_mem_util_pct','gpu_mem_used_mib','gpu_temperatura_c'])})
                row.update({f'cpu_logico_{i:02d}_pct':v for i,v in enumerate(psutil.cpu_percent(percpu=True))})
                writer.writerow(row);f.flush();stop.wait(max(.05,1-(time.perf_counter()-now)))
    thread=threading.Thread(target=sample,daemon=True);thread.start()
    return stop,thread


def mostrar_figura(fig):
    from IPython.display import display, Image
    import matplotlib.pyplot as plt
    out=io.BytesIO();fig.savefig(out,format='png',dpi=145,bbox_inches='tight')
    display(Image(data=out.getvalue()));plt.close(fig)
