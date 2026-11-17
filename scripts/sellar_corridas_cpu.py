"""Close records/manifests of completed CPU runs without overwriting evidence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experimento",default="EXP-002")
    args=parser.parse_args()
    if not args.experimento.startswith("EXP-") or not args.experimento[4:].isdigit():
        raise ValueError("Expected EXP-NNN.")
    experiment=ROOT/"resultados/secuencial"/args.experimento
    series=json.loads((experiment/"serie.json").read_text(encoding="utf-8"))
    for name in series["runs"]:
        run=experiment/name
        result=json.loads((run/"salida.json").read_text(encoding="utf-8"))
        if result["codigo_salida"]!=0: raise ValueError("Cannot seal a failed run as complete.")
        with (run/"tiempos.csv").open(encoding="utf-8",newline="") as handle:
            total=next(row for row in csv.DictReader(handle) if row["fase"]=="pipeline_total")
        record=run/"registro.md"
        if not record.exists():
            text=f"# {name}\n\nEstado: completada; CPU; {args.experimento}.\n\n"
            text+="Responsable: Codex por solicitud del usuario; revisión humana pendiente. REQ-03/07/12; PROC-03/04/05/06; DEC-011/012/013; CAM-003; BIT-005.\n\n"
            text+=f"Pipeline: {float(total['duracion_s']):.9f} s. Inicio medido UTC: {total['inicio_utc']}; fin: {total['fin_utc']}.\n\n"
            text+="Carga: 17,400 imágenes; splits12,180/2,610/2,610; cinco redes con15 épocas y batch128, PCA, dos bosques y evaluación/inferencia en15 fotos. Sin descarga ni reinstalación.\n\n"
            text+="El total excluye imports, construcción/warmup, hashes, checkpoints y figuras; fit incluye validación anidada. Caché del SO sin vaciar. CPU6hilos matemáticos, TF inter-op1; RF n_jobs=-1; preparación un worker.\n\n"
            text+="## Evidencias\n\n"
            for label,path in [("Comando e inicio","invocacion.json"),("Fin y salida","salida.json"),
                ("Configuración","configuracion.json"),("Contexto","contexto/contexto.json"),
                ("Fuentes exactas","contexto/fuentes.zip"),("Hardware","hardware.json"),
                ("Estados iniciales","estados-iniciales.json"),("Calentamiento","warmup.json"),
                ("Identidad de tensores","tensores.json"),("Tiempos crudos","tiempos.csv"),
                ("Épocas","epocas.csv"),("Métricas","metricas.json"),("Recursos","recursos.csv"),
                ("Salida estándar","stdout.log"),("Salida de diagnóstico","stderr.log"),
                ("Modelos locales y hashes","modelos-guardados.json")]:
                if not (run/path).is_file(): raise FileNotFoundError(run/path)
                text+=f"- [{label}]({path}).\n"
            text+="\nLa estadística de la serie se publica después de completar todas las repeticiones. Esta observación no es por sí sola una estimación del promedio. No hay resultados GPU en este registro.\n"
            record.write_text(text,encoding="utf-8")
        manifest=run/"artefactos.csv"
        if not manifest.exists():
            with manifest.open("x",encoding="utf-8",newline="") as handle:
                writer=csv.writer(handle);writer.writerow(["ruta_relativa","bytes","sha256"])
                for path in sorted(run.rglob("*")):
                    if path.is_file() and path!=manifest:
                        writer.writerow([path.relative_to(run).as_posix(),path.stat().st_size,
                                         hashlib.sha256(path.read_bytes()).hexdigest()])
            print(f"Registro cerrado: {name}")


if __name__=="__main__": main()
