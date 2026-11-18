"""Real on-screen monitor and screenshots of live psutil/nvidia-smi CSV samples.

Never captures the rest of the desktop. Not a synthetic utilization screenshot.
Run separately while the experiment is active; original CSV samples are retained.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import tkinter as tk
from PIL import ImageGrab
import ctypes

# Ask Windows for physical, not virtualized coordinates BEFORE creating Tk.
# Without per-monitor DPI awareness a 125% desktop crops the client rectangle.
try:
    ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    ctypes.windll.user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4))
except (AttributeError,OSError):
    pass

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--experimento",default="EXP-003")
    ap.add_argument("--max-segundos",type=int,default=1200)
    args=ap.parse_args()
    source=ROOT/"resultados/paralelo"/args.experimento
    destination=ROOT/"resultados/evidencias"/args.experimento
    destination.mkdir(parents=True,exist_ok=True)
    window=tk.Tk()
    window.title("Proyecto CUDA — monitor real del experimento")
    window.geometry("900x650+30+30")
    window.attributes("-topmost",True)
    canvas=tk.Canvas(window,width=900,height=650,bg="#ffffff",highlightthickness=0)
    canvas.pack(fill="both",expand=True)
    saved=set()
    start=time.monotonic()

    def number(row,key):
        try:return float(row.get(key,0) or 0)
        except ValueError:return 0.

    def poll():
        if len(saved)==2 or time.monotonic()-start>args.max_segundos:
            window.destroy();return
        samples=[]
        for path in source.glob("RUN-*/recursos.csv"):
            try:
                with path.open(encoding="utf8",newline="") as f:
                    rows=[r for r in csv.DictReader(f) if r.get("cpu_logico_11_pct") is not None]
                if rows:samples.append((rows[-1]["utc"],path,rows[-1]))
            except (OSError,csv.Error):pass
        if samples:
            _,path,row=max(samples,key=lambda x:x[0])
            canvas.delete("all")
            def text(x,y,value,size=12,color="#172b4d"):
                canvas.create_text(x,y,text=value,anchor="nw",font=("Segoe UI",size),fill=color)
            text(22,16,"MONITOR REAL — CPU / NVIDIA CUDA",18)
            text(22,55,path.parent.name,11)
            text(22,79,"Muestra UTC: "+row["utc"],11)
            text(22,103,"Fase: "+row["fase"]+"    Época: "+row.get("epoca",""),12)
            text(22,140,"CPU por procesador lógico — psutil (%)",13)
            for i in range(12):
                value=number(row,f"cpu_logico_{i:02d}_pct")
                y=174+i*22
                text(22,y,f"CPU {i:02d}",10)
                canvas.create_rectangle(95,y+2,595,y+15,fill="#edf1f7",outline="")
                canvas.create_rectangle(95,y+2,95+5*min(value,100),y+15,fill="#2367a2",outline="")
                text(610,y,f"{value:.1f}%",10)
            cpu=number(row,"cpu_procesos_pct")
            rss=number(row,"rss_bytes")/2**20
            text(22,450,f"Árbol del proceso: CPU {cpu:.1f}% (puede superar 100)   RAM RSS {rss:.0f} MiB",11)
            gpu=number(row,"gpu_util_pct")
            memory=number(row,"gpu_mem_used_mib")
            text(22,490,f"GTX 1650 — nvidia-smi: actividad {gpu:.0f}%    VRAM {memory:.0f} / 4096 MiB",13)
            canvas.create_rectangle(22,527,822,548,fill="#edf1f7",outline="")
            canvas.create_rectangle(22,527,22+8*min(gpu,100),548,fill="#d27700",outline="")
            text(22,565,"Fuente: CSV en vivo, psutil + nvidia-smi. No indica ocupación por SM.",10)
            text(22,588,"Los datos y la captura conservan UTC, RUN y fase para su trazabilidad.",10)
            target=None
            if row["fase"]=="preparacion_cpu" and "cpu" not in saved and cpu>10:
                target="cpu"
            elif row["fase"]=="entrenamiento_CNN-B" and "gpu" not in saved and gpu>=20:
                target="gpu"
            if target:
                window.update_idletasks();window.update()
                x,y=window.winfo_rootx(),window.winfo_rooty()
                w,h=window.winfo_width(),window.winfo_height()
                stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                output=destination/(stamp+"-"+target+".png")
                if output.exists():raise FileExistsError(output)
                ImageGrab.grab(bbox=(x,y,x+w,y+h)).save(output)
                meta=dict(tipo="captura_pantalla_monitor_real",captura_utc=datetime.now(timezone.utc).isoformat(),
                    run_id=path.parent.name,recursos_csv=path.relative_to(ROOT).as_posix(),muestra_original=row,
                    screenshot_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                    nota="Ventana real que muestra la ultima muestra CPU/GPU; solo se captura su area")
                output.with_suffix(".json").write_text(json.dumps(meta,indent=2),encoding="utf8")
                print("CAPTURA",output,flush=True)
                saved.add(target)
        window.after(500,poll)
    window.after(100,poll)
    window.mainloop()
    if len(saved)<2:raise RuntimeError("No se obtuvieron ambas fases; conservar evidencia parcial y repetir con otra identidad")


if __name__=="__main__":
    main()
