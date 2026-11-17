"""Copy a frozen CPU configuration/manifests to a new ID without resampling."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origen",default="EXP-002")
    parser.add_argument("--nuevo",required=True)
    parser.add_argument("--dataset",type=Path)
    args=parser.parse_args()
    for value in (args.origen,args.nuevo):
        if not value.startswith("EXP-") or not value[4:].isdigit(): raise ValueError("Expected EXP-NNN.")
    source=ROOT/"resultados/secuencial"/args.origen
    target=ROOT/"resultados/secuencial"/args.nuevo
    destination=ROOT/"configuraciones"/(args.nuevo+".json")
    config_path=ROOT/"configuraciones"/(args.origen+".json")
    config=json.loads(config_path.read_text(encoding="utf-8"))
    if target.exists() or destination.exists(): raise FileExistsError("New experiment must not exist.")
    if sha(ROOT/"scripts/benchmark_cpu.py")!=config["codigo"]["benchmark_sha256"]:
        raise ValueError("Frozen benchmark source differs; restore the recorded version first.")
    if sha(ROOT/"LAB3-DATASCIENCE/Lab3_ASL_Deep_Learning.ipynb")!=config["codigo"]["notebook_sha256"]:
        raise ValueError("Original notebook differs.")
    series=json.loads((source/"serie.json").read_text(encoding="utf-8"))
    installed=sorted({f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()
                      if d.metadata["Name"]},key=str.casefold)
    expected=(source/series["runs"][0]/"contexto/paquetes-instalados.txt").read_text(encoding="utf-8").splitlines()
    if installed!=expected: raise ValueError("Use the recorded PCUDA environment; package inventory differs.")
    for name,key in (("manifest_fuente.csv","manifest_fuente_sha256"),
                     ("manifest_splits.csv","manifest_splits_sha256"),("label_mapping.json","label_mapping_sha256")):
        if sha(source/name)!=config["datos"][key]: raise ValueError(f"Changed manifest: {name}")
    dataset=(args.dataset or Path(config["datos"]["ruta"])).resolve()
    if not dataset.is_dir(): raise FileNotFoundError(dataset)
    target.mkdir(parents=True)
    for name in ("manifest_fuente.csv","manifest_splits.csv","label_mapping.json","validacion_datos.json"):
        shutil.copyfile(source/name,target/name)
    config["experimento_id"]=args.nuevo
    config["creado_utc"]=datetime.now(timezone.utc).isoformat()
    config["datos"]["ruta"]=str(dataset)
    config["reproduccion_de"]=dict(experimento=args.origen,configuracion_sha256=sha(config_path),
        alcance="DEC-015: five runs, fourteen common phases; no resampling; same DAT identities")
    config["medicion"]["criterio"]="DEC-015: close after five successful runs using cerrar_serie_cpu.py; classify adaptive branches."
    config["comandos"]={"directorio":str(ROOT),
        "cpu":f"PCUDA\\Scripts\\python.exe -X utf8 scripts/benchmark_cpu.py run --experimento {args.nuevo}",
        "cierre":f"PCUDA\\Scripts\\python.exe -X utf8 scripts/cerrar_serie_cpu.py --experimento {args.nuevo}"}
    text=json.dumps(config,ensure_ascii=False,indent=2)+"\n"
    destination.write_text(text,encoding="utf-8")
    (target/"configuracion.json").write_text(text,encoding="utf-8")
    print(f"Prepared {args.nuevo} from frozen manifests; no training, resampling or download. Run benchmark and closure observer in two terminals.")


if __name__=="__main__": main()
