"""Close after five complete runs when adaptive branches invalidate pooled precision.

The original frozen controller is left unchanged. This observer preserves its
state, stops only this workspace's owned controller after five successful runs,
and records the scope amendment. It never edits completed run artifacts.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import psutil

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experimento",default="EXP-002")
    args=parser.parse_args()
    if not args.experimento.startswith("EXP-") or not args.experimento[4:].isdigit():
        raise ValueError("Expected EXP-NNN.")
    directory=ROOT/"resultados/secuencial"/args.experimento
    while True:
        try: series=json.loads((directory/"serie.json").read_text(encoding="utf-8"))
        except (FileNotFoundError,json.JSONDecodeError): time.sleep(1);continue
        if len(series.get("runs",[]))>=5: break
        # Poll slowly until the fifth run is near completion, minimizing interference.
        fifth=list(directory.glob("RUN-*-cpu-05/estado.json"))
        phase=""
        if fifth:
            try: phase=json.loads(fifth[0].read_text(encoding="utf-8")).get("fase","")
            except json.JSONDecodeError: pass
        time.sleep(.05 if phase in ("inferencia_grupo","completada") else 10)
    runs=series["runs"]
    if len(runs)!=5: raise ValueError("Already more than five complete runs; audit manually.")
    for name in runs:
        if json.loads((directory/name/"salida.json").read_text())["codigo_salida"]!=0:
            raise ValueError("Cannot close an unsuccessful run as complete.")
    (directory/"serie-antes-cierre.json").open("x",encoding="utf-8").write(json.dumps(series,indent=2)+"\n")
    controllers=[];unexpected_workers=[]
    for process in psutil.process_iter(["pid","cmdline"]):
        try:
            command=process.info["cmdline"] or []
            if args.experimento not in command or not any(str(a).endswith("benchmark_cpu.py") for a in command): continue
            if Path(process.cwd()).resolve()!=ROOT: continue
            if "run" in command: controllers.append(process)
            elif "worker" in command and "--repeticion" in command and int(command[command.index("--repeticion")+1])>5:
                unexpected_workers.append(process)
        except (psutil.NoSuchProcess,psutil.AccessDenied): continue
    stopped=[]
    for process in [*reversed(controllers),*unexpected_workers]:
        try: process.terminate();stopped.append(process.pid)
        except psutil.NoSuchProcess: pass
    psutil.wait_procs(controllers+unexpected_workers,timeout=10)
    # A controller may have launched the next interpreter while process_iter
    # was taking its snapshot. Rescan after it has exited to avoid orphan work.
    seen={p.pid for p in unexpected_workers}
    for _ in range(2):
        late=[]
        for process in psutil.process_iter(["pid","cmdline"]):
            try:
                command=process.info["cmdline"] or []
                if args.experimento not in command or "worker" not in command or "--repeticion" not in command: continue
                if not any(str(a).endswith("benchmark_cpu.py") for a in command): continue
                if int(command[command.index("--repeticion")+1])<=5 or Path(process.cwd()).resolve()!=ROOT: continue
                late.append(process)
                if process.pid not in seen: unexpected_workers.append(process);seen.add(process.pid)
                process.terminate()
            except (psutil.NoSuchProcess,psutil.AccessDenied): continue
        psutil.wait_procs(late,timeout=3)
        time.sleep(.2)
    series.update(estado="completada_alcance_comun_5",fin_utc=datetime.now(timezone.utc).isoformat(),
        cierre_por_revision="DEC-015: global precision rule inapplicable to mixed adaptive architectures; five common-work repetitions retained",
        criterio_precision_cumplido=False,criterio_global_aplicable=False,pids_control_detenidos=stopped,
        workers_adicionales_interrumpidos=[p.pid for p in unexpected_workers])
    (directory/"serie.json").write_text(json.dumps(series,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (directory/"cierre-alcance.md").write_text(
        "# Cierre de la serie con cinco corridas completas\n\n"
        "DEC-015; cierre después de comprobar cinco códigos de salida0. Se conserva el estado previo del controlador en serie-antes-cierre.json.\n\n"
        "La regla original de ampliar hasta9 para estrechar el IC del total suponía carga homogénea. La selección adaptativa cambió CNN-B+aug por CNN-A+aug en una corrida; se estratifican ramas y se define la referencia con14 fases comunes, sin descartar observaciones. No se declara cumplida la precisión objetivo del total mezclado.\n\n"
        "Se detuvo únicamente el controlador de este repositorio al terminar las cinco corridas para evitar extensiones no comparables; no se modificaron archivos de corridas cerradas. Las mediciones originales y su configuración permanecen intactas.\n",
        encoding="utf-8")
    print(json.dumps(series,ensure_ascii=False),flush=True)


if __name__=="__main__": main()
