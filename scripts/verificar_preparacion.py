"""Verify documentation and static capture; never run the model or GPU code."""

from __future__ import annotations

import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from urllib.parse import unquote
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify() -> list[str]:
    results = []
    documents = [ROOT / "README.md", ROOT / "AGENTS.md",
                 *sorted((ROOT / "configuraciones").rglob("*.md")),
                 *sorted((ROOT / "docs/trazabilidad").rglob("*.md")),
                 *sorted((ROOT / "resultados").rglob("*.md"))]
    links_count = 0
    for doc in documents:
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", doc.read_text(encoding="utf-8")):
            if "://" in target or target.startswith(("#", "mailto:")):
                continue
            target = unquote(target.split("#", 1)[0])
            if not (doc.parent / target).exists():
                raise ValueError(f"Broken link: {doc.relative_to(ROOT)} -> {target}")
            links_count += 1
    results.append(f"Enlaces locales: {links_count} validos en {len(documents)} Markdown.")

    config = json.loads((ROOT / "configuraciones/experimento.plantilla.json").read_text(encoding="utf-8"))
    if config.get("es_plantilla") is not True or config["experimento_id"] is not None:
        raise ValueError("Configuration must remain an unfilled template.")
    results.append("JSON valido y marcado como plantilla pendiente, sin experimento ficticio.")

    count = 0
    with (ROOT / "resultados/inventario-inicial.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            data = (ROOT / row["ruta_relativa"]).read_bytes()
            if sha(data) != row["sha256"] or len(data) != int(row["bytes"]):
                raise ValueError(f"Changed baseline input: {row['ruta_relativa']}")
            count += 1
    results.append(f"Inventario inicial: {count} archivos con bytes y SHA-256 coincidentes.")

    csv_count = 0
    for path in sorted((ROOT / "resultados").rglob("*.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        if not rows or len(set(rows[0])) != len(rows[0]):
            raise ValueError(f"Empty or duplicate CSV header: {path}")
        if any(len(row) != len(rows[0]) for row in rows[1:]):
            raise ValueError(f"Inconsistent CSV width: {path}")
        csv_count += 1
    for name in ("secuencial/tiempos.csv", "paralelo/tiempos.csv",
                 "comparacion/resumen.csv", "comparacion/speedup.csv"):
        with (ROOT / "resultados" / name).open(encoding="utf-8", newline="") as handle:
            if len(list(csv.reader(handle))) != 1:
                raise ValueError(f"Preparation must not invent benchmark data: {name}")
    results.append(f"CSV: {csv_count} esquemas consistentes; tablas experimentales sin observaciones.")

    for path in sorted((ROOT / "scripts").glob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    results.append("Scripts Python: sintaxis valida por analisis AST, sin importar el modelo.")

    with tempfile.TemporaryDirectory(prefix="cuda-trazabilidad-") as temporary:
        destination = Path(temporary) / "contexto"
        command = [sys.executable, "-X", "utf8", str(ROOT / "scripts/capturar_contexto.py"),
                   "--destino", str(destination)]
        first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                               encoding="utf-8", timeout=120)
        if first.returncode:
            raise ValueError(f"Capture failed: {first.stderr}")
        context = json.loads((destination / "contexto.json").read_text(encoding="utf-8"))
        if context["tipo_evidencia"] != "captura_estatica_sin_ejecucion_del_modelo":
            raise ValueError("Incorrect evidence type.")
        with zipfile.ZipFile(destination / "fuentes.zip") as archive:
            if archive.testzip() is not None:
                raise ValueError("Corrupt source ZIP.")
            with (destination / "fuentes.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if set(archive.namelist()) != {row["ruta_relativa"] for row in rows}:
                raise ValueError("Source archive and manifest differ.")
            for row in rows:
                data = archive.read(row["ruta_relativa"])
                if sha(data) != row["sha256"] or len(data) != int(row["bytes"]):
                    raise ValueError("Source hash mismatch.")
        with (destination / "artefactos.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                data = (destination / row["archivo"]).read_bytes()
                if sha(data) != row["sha256"] or len(data) != int(row["bytes"]):
                    raise ValueError("Context artifact hash mismatch.")
        before = {path.name: sha(path.read_bytes()) for path in destination.iterdir()}
        second = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=30)
        after = {path.name: sha(path.read_bytes()) for path in destination.iterdir()}
        if second.returncode != 1 or before != after:
            raise ValueError("Capture overwrote an existing destination.")
        results.append(f"Captura temporal: {len(rows)} fuentes archivadas y hashes verificados; "
                       "contexto/paquetes registrados; destino existente rechazado sin cambios.")
    return results


def main() -> int:
    try:
        results = verify()
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f"Verificacion fallida: {exc}", file=sys.stderr)
        return 1
    report = ROOT / "resultados/preparacion/verificacion-preparacion.md"
    text = "# Verificacion de la preparacion\n\n"
    text += f"Fecha UTC: {datetime.now(timezone.utc).isoformat()}.\n\n"
    text += "Estado: comprobaciones aprobadas. Tipo: evidencia de preparacion.\n\n"
    text += "Comando desde la raiz del repositorio:\n\n```powershell\n"
    text += "python -X utf8 scripts/verificar_preparacion.py\n```\n\n"
    text += f"Interprete: `{sys.executable}`. Python: {sys.version.split()[0]}.\n\n"
    text += "\n".join(f"- {item}" for item in results)
    text += "\n\nNo se ejecuto el notebook, entrenamiento, benchmark, descarga de datos "
    text += "ni comprobacion de hardware GPU. La captura de prueba se elimino al cerrar "
    text += "el directorio temporal; se conserva este resultado y el script para repetirla.\n"
    text += "\nEste informe de preparacion se actualiza al repetir el comando. Los RUN "
    text += "experimentales se conservan por separado sin sobrescribir.\n"
    report.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
