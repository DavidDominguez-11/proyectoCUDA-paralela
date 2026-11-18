"""Capture one genuine live monitor window for an additional notebook RUN."""
import argparse
import csv
import ctypes
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import tkinter as tk
from PIL import ImageGrab


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--tipo',choices=['cpu','cuda'],required=True)
    ap.add_argument('--nombre')
    ap.add_argument('--modelo',choices=['FC-A','FC-B'],required=True)
    ap.add_argument('--sostenido',action='store_true')
    args=ap.parse_args()
    run=args.run.resolve()
    label=args.nombre or args.tipo
    if any(c not in 'abcdefghijklmnopqrstuvwxyz-' for c in label):raise ValueError('Invalid capture label')
    if (run/f'captura-{label}.png').exists():raise FileExistsError('Capture already exists')
    try:ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except (AttributeError,OSError):pass
    window=tk.Tk();window.title('Ejecución real del notebook — CPU / CUDA')
    window.withdraw()
    window.geometry('900x650+30+30');window.attributes('-topmost',True)
    canvas=tk.Canvas(window,width=900,height=650,bg='white',highlightthickness=0)
    canvas.pack(fill='both',expand=True)
    start=time.monotonic()
    def poll():
        if time.monotonic()-start>3600:window.destroy();return
        try:
            with (run/'recursos.csv').open(encoding='utf8',newline='') as f:data=list(csv.DictReader(f))
        except (OSError,csv.Error):window.after(1000,poll);return
        if not data:window.after(1000,poll);return
        row=data[-1];canvas.delete('all')
        def number(k):
            try:return float(row.get(k,0) or 0)
            except ValueError:return 0.
        def text(x,y,s,size=11):canvas.create_text(x,y,text=s,anchor='nw',font=('Segoe UI',size),fill='#173B57')
        text(22,16,'NOTEBOOK — EJECUCIÓN REAL CPU / CUDA',18)
        text(22,55,run.name,10);text(22,80,'Muestra UTC: '+row['utc'])
        text(22,105,'Fase: '+row['fase']+'  Modelo: '+row.get('modelo','')+'  Época: '+row.get('epoca',''))
        text(22,140,'CPU por procesador lógico — psutil (%)',13)
        for i in range(12):
            value=number(f'cpu_logico_{i:02d}_pct');y=174+i*22
            text(22,y,f'CPU {i:02d}',10)
            canvas.create_rectangle(95,y+2,595,y+15,fill='#edf1f7',outline='')
            canvas.create_rectangle(95,y+2,95+5*min(value,100),y+15,fill='#2367a2',outline='')
            text(610,y,f'{value:.1f}%',10)
        text(22,450,f"Proceso e hijos: CPU {number('cpu_procesos_pct'):.1f}%   RAM RSS {number('rss_bytes')/2**20:.0f} MiB")
        gpu=number('gpu_util_pct');memory=number('gpu_mem_used_mib')
        text(22,490,f'GTX 1650 — nvidia-smi: actividad {gpu:.0f}%    VRAM {memory:.0f} / 4096 MiB',13)
        canvas.create_rectangle(22,527,822,548,fill='#edf1f7',outline='')
        canvas.create_rectangle(22,527,22+8*min(gpu,100),548,fill='#d27700',outline='')
        text(22,565,'Fuente: muestra original en vivo; actividad agregada, no ocupación por SM.',10)
        text(22,590,'La ventana y la captura pertenecen a la ejecución actual del notebook.',10)
        selected=(row['fase']=='entrenamiento' and row.get('modelo')=='CNN-B' and number('cpu_procesos_pct')>100) if args.tipo=='cpu' else (row['fase']=='entrenamiento_'+args.modelo and number('epoca')>=2 and gpu>0)
        if args.sostenido:
            selected=selected and number('epoca')>=2 and (number('cpu_procesos_pct')>=250 if args.tipo=='cpu' else gpu>=85)
        if selected:
            window.deiconify();window.update_idletasks();window.update()
            def capture():
                x,y=window.winfo_rootx(),window.winfo_rooty();w,h=window.winfo_width(),window.winfo_height()
                target=run/f'captura-{label}.png'
                ImageGrab.grab(bbox=(x,y,x+w,y+h),all_screens=True).save(target)
                meta=dict(tipo='captura_original_monitor_en_vivo',captura_utc=datetime.now(timezone.utc).isoformat(),
                          run_id=run.name,muestra_original=row,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                          capturador_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
                target.with_suffix('.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf8')
                print('CAPTURA',target,flush=True);window.after(700,window.destroy)
            window.after(400,capture);return
        window.after(1000,poll)
    window.after(200,poll);window.mainloop()


if __name__=='__main__':main()
