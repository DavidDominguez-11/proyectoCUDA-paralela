"""Capture reproducibility context without importing or executing the project.

Run with the Python interpreter whose installed distributions should be recorded.
Creates a new destination only. Does not probe hardware or benchmark anything.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = {"docs", "LAB3-DATASCIENCE", "scripts", "configuraciones",
                "src", "tests", "notebooks", "cuda"}
EXTENSIONS = {".py", ".ipynb", ".md", ".pdf", ".json", ".txt", ".csv",
              ".jpeg", ".jpg", ".png", ".cu", ".cuh", ".cpp", ".h", ".toml",
              ".yaml", ".yml", ".ps1", ".sh", ".lock"}
EXCLUDED_PARTS = {"data", "datasets", "__pycache__", ".ipynb_checkpoints",
                  ".venv", "node_modules", "PCUDA", "build", "dist"}
ENV_NAMES = ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS",
             "TF_DETERMINISTIC_OPS", "CUBLAS_WORKSPACE_CONFIG")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, check=True, timeout=30,
    )
    return result.stdout.decode("utf-8", errors="strict")


def is_source(name: str) -> bool:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe source path: {name}")
    if any(part in EXCLUDED_PARTS for part in path.parts):
        return False
    if path.name in {".gitignore", "README.md", "AGENTS.md"} and len(path.parts) == 1:
        return True
    return ((path.parts[0] in SOURCE_ROOTS or len(path.parts) == 1) and
            (path.suffix.lower() in EXTENSIONS or path.name == ".gitignore") and
            not path.name.startswith(".env") and
            "secret" not in path.name.lower() and
            "credential" not in path.name.lower())


def capture(destination: Path) -> None:
    destination = destination.resolve()
    if destination == ROOT or (destination.is_relative_to(ROOT) and
                              not destination.is_relative_to(ROOT / "resultados")):
        raise ValueError("Inside the repository, destination must be under resultados/.")
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")

    # Read all source bytes before creating output; the archive is self-contained.
    commit = git("rev-parse", "HEAD").strip()
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    names = sorted(set(filter(None, git("ls-files", "--cached", "--others",
                                       "--exclude-standard", "-z").split("\0"))))
    sources = {}
    for name in names:
        if not is_source(name):
            continue
        path = ROOT / name
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f"Symlink or external source not supported: {name}")
        if path.is_file():
            sources[Path(name).as_posix()] = path.read_bytes()

    packages = sorted({
        f"{dist.metadata['Name']}=={dist.version}"
        for dist in importlib.metadata.distributions() if dist.metadata["Name"]
    }, key=str.casefold)
    timestamp = datetime.now(timezone.utc).isoformat()
    context = {
        "capturado_utc": timestamp,
        "tipo_evidencia": "captura_estatica_sin_ejecucion_del_modelo",
        "commit": commit,
        "arbol_limpio": not status.strip(),
        "directorio_repositorio": str(ROOT),
        "directorio_invocacion": str(Path.cwd()),
        "comando_captura": sys.argv,
        "python": {"ejecutable": sys.executable, "version": sys.version},
        "sistema": {"plataforma": platform.platform(), "maquina": platform.machine(),
                    "procesadores_logicos": os.cpu_count()},
        "variables_permitidas": {name: os.environ.get(name) for name in ENV_NAMES},
        "pendiente_validar_en_ejecucion": ["CPU y RAM efectivos", "GPU y VRAM",
            "driver, CUDA y cuDNN efectivos", "dispositivos usados por fase",
            "entorno reconstruido desde lock", "comando del experimento y resultados"],
        "alcance_fuentes": "archivos Git y nuevos no ignorados en roots/extensiones permitidos",
        "directorios_fuente_permitidos": sorted(SOURCE_ROOTS),
        "extensiones_fuente_permitidas": sorted(EXTENSIONS),
        "exclusiones": "resultados, datos grandes, entornos, secretos por nombre; revisar selección antes de compartir",
        "fuentes_count": len(sources),
        "nota_paquetes": "Inventario instalado del interprete; no sustituye lock, wheels ni prueba de reinstalacion.",
    }

    destination.mkdir(parents=True, exist_ok=False)
    (destination / "contexto.json").write_text(
        json.dumps(context, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (destination / "paquetes-instalados.txt").write_text(
        "\n".join(packages) + "\n", encoding="utf-8")
    (destination / "git-status.txt").write_text(status, encoding="utf-8")
    (destination / "comandos-contexto.txt").write_text(
        "git rev-parse HEAD\ngit status --porcelain=v1 --untracked-files=all\n"
        "git ls-files --cached --others --exclude-standard -z\n"
        "Paquetes: importlib.metadata.distributions() del interprete registrado.\n",
        encoding="utf-8")
    with zipfile.ZipFile(destination / "fuentes.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sources.items():
            archive.writestr(name, data)
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(["ruta_relativa", "bytes", "sha256"])
    for name, data in sources.items():
        writer.writerow([name, len(data), digest(data)])
    (destination / "fuentes.csv").write_text(buffer.getvalue(), encoding="utf-8", newline="")
    with (destination / "artefactos.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["archivo", "bytes", "sha256"])
        for path in sorted(destination.iterdir()):
            if path.name != "artefactos.csv":
                data = path.read_bytes()
                writer.writerow([path.name, len(data), digest(data)])
    print(f"Contexto estatico guardado en {destination}; {len(sources)} fuentes. Sin benchmark.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destino", required=True, type=Path)
    args = parser.parse_args()
    try:
        capture(args.destino)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"No se completo la captura: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
