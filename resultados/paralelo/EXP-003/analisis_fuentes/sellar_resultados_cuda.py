"""Close existing CUDA evidence; refuse overwrite, preserve postprocessing code."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from benchmark_cpu import sha,write_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--experimento",default="EXP-003")
    args=ap.parse_args()
    folder=ROOT/"resultados/paralelo"/args.experimento
    target=folder/"artefactos-experimento.csv"
    if target.exists():raise FileExistsError("Experiment evidence already sealed")
    series=json.loads((folder/"serie.json").read_text(encoding="utf8"))
    if series["estado"]!="completada":raise ValueError("Series incomplete")
    if len(series["corridas"])<5 or any(r["codigo"] for r in series["corridas"]):raise ValueError("Invalid measured runs")
    native=json.loads((folder/"notebook-validado.salida.json").read_text(encoding="utf8"))
    if native["codigo"]!=0 or not native["sin_nuevas_corridas"]:raise ValueError("Notebook execution not verified")
    if sha(folder/"serie.json")!=native["serie_despues_sha256"]:raise ValueError("Series changed after presentation")
    nb=json.loads((folder/"notebook-validado.ipynb").read_text(encoding="utf8"))
    errors=[o for c in nb["cells"] for o in c.get("outputs",[]) if o["output_type"]=="error"]
    if errors:raise ValueError("Notebook contains error outputs")
    if not (folder/"informe-paso-5.pdf").is_file():raise FileNotFoundError("PDF report")
    checkpoints=json.loads((folder/"validacion_checkpoints.json").read_text(encoding="utf8"))
    if checkpoints["estado"]!="aprobado":raise ValueError("Checkpoints not audited")
    count=0
    for run in series["corridas"]:
        directory=folder/run["run_id"]
        with (directory/"artefactos.csv").open(encoding="utf8",newline="") as f:
            for row in csv.DictReader(f):
                path=directory/row["ruta"]
                if path.stat().st_size!=int(row["bytes"]) or sha(path)!=row["sha256"]:raise ValueError("RUN evidence changed")
                count+=1
    analysis=folder/"analisis_fuentes"
    analysis.mkdir(exist_ok=False)
    for name in ("resumir_cuda.py","ejecutar_notebook_cuda.py","sellar_resultados_cuda.py","capturar_monitor_cuda.py","preparar_reproduccion_cuda.py"):
        with (analysis/name).open("xb") as f:f.write((ROOT/"scripts"/name).read_bytes())
    write_json(folder/"cierre.json",dict(estado="cerrado",fecha_utc=datetime.now(timezone.utc).isoformat(),
        corridas=len(series["corridas"]),archivos_run_verificados=count,
        notebook_codigo_nativo=0,celdas_codigo_ejecutadas=sum(c.get("execution_count") is not None for c in nb["cells"]),
        nuevos_entrenamientos_durante_presentacion=False,
        codigo_analisis="analisis_fuentes; codigo de medicion en cada RUN/contexto/fuentes.zip",
        alcance="CNN-B; CPU historica compatible; transferencias/validacion incluidas; no todo el notebook original"))
    roots=[folder,ROOT/"resultados/evidencias"/args.experimento]
    paths=sorted({p for root in roots for p in root.rglob("*") if p.is_file() and p!=target})
    with target.open("x",encoding="utf8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["ruta_relativa","bytes","sha256"])
        writer.writeheader()
        for path in paths:writer.writerow(dict(ruta_relativa=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=sha(path)))
    print("SELLADO",args.experimento,"archivos",len(paths),"SHA256 manifiesto",sha(target),flush=True)


if __name__=="__main__":main()
