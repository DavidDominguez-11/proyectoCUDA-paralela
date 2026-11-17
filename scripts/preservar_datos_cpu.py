"""Preserve the exact selected inputs locally, after timing; never download."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    result=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""): result.update(block)
    return result.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experimento",default="EXP-002")
    args=parser.parse_args()
    if not args.experimento.startswith("EXP-") or not args.experimento[4:].isdigit():
        raise ValueError("Expected EXP-NNN.")
    experiment=ROOT/"resultados/secuencial"/args.experimento
    if not json.loads((experiment/"serie.json").read_text(encoding="utf-8"))["estado"].startswith("completada"):
        raise ValueError("Archive only after the measured series, to avoid I/O interference.")
    config=json.loads((ROOT/"configuraciones"/(args.experimento+".json")).read_text(encoding="utf-8"))
    manifest=experiment/"manifest_fuente.csv"
    if sha(manifest)!=config["datos"]["manifest_fuente_sha256"]:
        raise ValueError("Source manifest changed.")
    with manifest.open(encoding="utf-8",newline="") as handle: rows=list(csv.DictReader(handle))
    target=ROOT/"data"/args.experimento/"fuente-seleccionada.zip"
    target.parent.mkdir(parents=True,exist_ok=True)
    source_root=Path(config["datos"]["ruta"]).resolve()
    with zipfile.ZipFile(target,"x",compression=zipfile.ZIP_STORED) as archive:
        for row in sorted(rows,key=lambda item:item["ruta_relativa"]):
            relative=Path(row["ruta_relativa"])
            source=(source_root/relative).resolve()
            if relative.is_absolute() or not source.is_relative_to(source_root):
                raise ValueError("Unsafe source path.")
            value=source.read_bytes()
            if hashlib.sha256(value).hexdigest()!=row["sha256"]:
                raise ValueError(f"Source changed: {relative}")
            entry=zipfile.ZipInfo(relative.as_posix(),date_time=(1980,1,1,0,0,0))
            entry.create_system=3
            entry.external_attr=0o100644<<16
            archive.writestr(entry,value)
    with zipfile.ZipFile(target) as archive:
        if len(archive.namelist())!=len(rows) or archive.testzip() is not None:
            raise ValueError("Archive verification failed.")
    metadata=dict(fecha_utc=datetime.now(timezone.utc).isoformat(),archivo=target.relative_to(ROOT).as_posix(),
        archivos=len(rows),bytes=target.stat().st_size,sha256=sha(target),
        manifest_fuente_sha256=sha(manifest),metodo="ZIP_STORED; sorted paths; fixed entry metadata; byte-identical inputs",
        alcance="Only 17400 selected dataset images. Team photos already preserved in each context source ZIP.")
    (experiment/"preservacion-datos.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    text=f"# Preservación local de datos de {args.experimento}\n\n"
    text+=f"Fecha UTC: {metadata['fecha_utc']}. Archivo: `{metadata['archivo']}`.\n\n"
    text+=f"Imágenes: {len(rows)}; bytes: {metadata['bytes']}; SHA-256: `{metadata['sha256']}`.\n\n"
    text+="Se verificó cada entrada contra manifest_fuente.csv y la integridad del ZIP. Copia local sin descarga ni modificación del caché, creada después de medir. Entradas ordenadas y metadatos ZIP fijos para evitar timestamps variables en el archivo.\n\n"
    text+="El ZIP conserva únicamente las imágenes seleccionadas, con rutas relativas originales; no sustituye al inventario de87,000 archivos del proveedor. Las15 fotos propias están en las fuentes de cada contexto.\n\n"
    text+="Comando desde la raíz (rechaza sobrescribir el ZIP):\n\n```powershell\n"
    text+=f".\\PCUDA\\Scripts\\python.exe -X utf8 scripts/preservar_datos_cpu.py --experimento {args.experimento}\n```\n\n"
    text+="Para recuperar entradas sin red: verificar SHA-256, extraer el ZIP a una carpeta nueva y reutilizar manifest_splits.csv/label_mapping.json; no volver a muestrear. Registrar una configuración nueva si cambia la ruta del dataset. No ejecutar prepare sobre el subconjunto como si contuviera las87,000 imágenes.\n\n"
    text+="Los binarios de data/ están ignorados por Git. Conservarlos con el paquete local de reproducción o una copia de respaldo; publicar sus manifiestos/hashes. No se certifica recuperación en otra máquina ni reconstrucción de PCUDA desde cero.\n"
    (experiment/"preservacion-datos.md").write_text(text,encoding="utf-8")
    print(json.dumps(metadata,ensure_ascii=False))


if __name__=="__main__": main()
