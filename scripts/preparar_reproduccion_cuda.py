"""New experiment identity using preserved data/state, never overwrite EXP-003."""
import argparse
import copy
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from benchmark_cpu import sha,write_json
from benchmark_cuda import locations


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--experimento",required=True,help="A NEW EXP-NNN, not EXP-003")
    parser.add_argument("--dataset",required=True,type=Path,help="Existing root with the frozen relative image paths")
    args=parser.parse_args()
    conf,folder=locations(args.experimento)
    if conf.exists() or folder.exists():raise FileExistsError("Choose a new identity")
    source=ROOT/"configuraciones/EXP-003.json"
    config=json.loads(source.read_text(encoding="utf8"))
    dataset=args.dataset.resolve(strict=True)
    if not dataset.is_dir():raise NotADirectoryError(dataset)
    for relative,digest in config["codigo"]["fuentes"].items():
        if sha(ROOT/relative)!=digest:raise ValueError("Restore frozen sources before reproduction: "+relative)
    cpu_config=json.loads((ROOT/"configuraciones/EXP-002.json").read_text(encoding="utf8"))
    for relative,digest in {"scripts/benchmark_cpu.py":cpu_config["codigo"]["benchmark_sha256"],
                           "LAB3-DATASCIENCE/Lab3_ASL_Deep_Learning.ipynb":cpu_config["codigo"]["notebook_sha256"]}.items():
        if sha(ROOT/relative)!=digest:raise ValueError("Original/transitive source differs")
        config["codigo"]["fuentes"][relative]=digest
    for name,version in config["entorno"]["paquetes"].items():
        if importlib.metadata.version(name)!=version:raise ValueError("Environment differs: "+name)
    initial=ROOT/"data/EXP-002/RUN-20261117T171942Z-cpu-01/CNN-B.estado-inicial.npz"
    initial_info=json.loads((ROOT/"resultados/secuencial/EXP-002/RUN-20261117T171942Z-cpu-01/estados-iniciales.json").read_text(encoding="utf8"))["CNN-B"]
    if sha(initial)!=initial_info["archivo_sha256"]:raise ValueError("Initial state differs")
    config=copy.deepcopy(config)
    config.update(experimento_id=args.experimento,creado_utc=datetime.now(timezone.utc).isoformat(),
        origen_reproduccion=dict(experimento="EXP-003",configuracion_sha256=sha(source),
                                generador_sha256=sha(Path(__file__))))
    config["datos"]["ruta"]=str(dataset)
    config["codigo"]["fuentes"]["scripts/preparar_reproduccion_cuda.py"]=sha(Path(__file__))
    folder.mkdir(parents=True,exist_ok=False)
    write_json(conf,config)
    (folder/"configuracion.json").write_bytes(conf.read_bytes())
    print("Prepared, not executed:",args.experimento,sha(conf))
    print("Run with PCUDA: scripts/benchmark_cuda.py run --experimento",args.experimento)


if __name__=="__main__":main()
