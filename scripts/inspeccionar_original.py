"""Extract source and stored output evidence; never execute notebook cells."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destino", required=True, type=Path)
    args = parser.parse_args()
    notebook_path = ROOT / "LAB3-DATASCIENCE/Lab3_ASL_Deep_Learning.ipynb"
    pdf_path = ROOT / "docs/Proyecto 3 CUDA.pdf"
    notebook_bytes = notebook_path.read_bytes()
    notebook = json.loads(notebook_bytes.decode("utf-8"))
    code_cells = [(i, cell) for i, cell in enumerate(notebook["cells"])
                  if cell["cell_type"] == "code"]
    lines = ["# Evidencia de lectura estática del original", "",
             f"Fecha UTC: {datetime.now(timezone.utc).isoformat()}.", "",
             "Tipo: fuentes y salidas históricas extraídas; sin ejecutar celdas.", "",
             f"Notebook SHA-256: `{hashlib.sha256(notebook_bytes).hexdigest()}`.", "",
             f"PDF SHA-256: `{hashlib.sha256(pdf_path.read_bytes()).hexdigest()}`.", "",
             f"Celdas: {len(notebook['cells'])}; de código: {len(code_cells)}.", "",
             "Índices de celda desde cero. El contador guardado no identifica una corrida nueva.", "",
             "Comando desde la raíz (usar otro destino para repetir sin sobrescribir):", "",
             "```powershell",
             "python -X utf8 scripts/inspeccionar_original.py --destino resultados/analisis-original/revision-estatica.md",
             "```", "", "## Épocas visibles en salidas guardadas", "",
             "| Celda | Modelo | Etiquetas Epoch encontradas | Última época visible | Máximo anunciado |",
             "|---|---|---:|---:|---:|"]
    models = {32: "CNN-A", 34: "CNN-B", 41: "FC-A", 42: "FC-B", 52: "CNN seleccionada + augmentation"}
    for index, name in models.items():
        output = "".join("".join(item.get("text", []))
                         for item in notebook["cells"][index].get("outputs", []))
        epochs = re.findall(r"Epoch (\d+)/(\d+)", output)
        last, maximum = epochs[-1] if epochs else ("no_visible", "no_visible")
        lines.append(f"| {index} | {name} | {len(epochs)} | {last} | {maximum} |")
    lines += ["", "Estas etiquetas pertenecen a logs persistidos. No prueban una ejecución actual ni tiempos comparables.",
              "", "## Código completo de las celdas originales"]
    for index, cell in code_cells:
        source = "".join(cell["source"])
        lines += ["", f"### Celda {index}", "",
                  f"SHA-256 del texto UTF-8 de la celda: `{hashlib.sha256(source.encode('utf-8')).hexdigest()}`.",
                  "", "```python", source.rstrip(), "```"]
    lines += ["", "## Salidas históricas seleccionadas"]
    for index in (6, 23, 24, 36, 43, 47, 53, 56, 62, 64):
        lines += ["", f"### Salida de celda {index}", "", "```text"]
        for item in notebook["cells"][index].get("outputs", []):
            lines.append("".join(item.get("text", item.get("data", {}).get("text/plain", []))).rstrip())
        lines.append("```")
    args.destino.parent.mkdir(parents=True, exist_ok=True)
    with args.destino.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"Evidencia estática: {args.destino}; {len(code_cells)} celdas de código. Sin ejecución del modelo.")


if __name__ == "__main__":
    main()
