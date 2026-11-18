"""Create a COPY with guarded original cells and reproducible CUDA workflow."""
import copy
import hashlib
import json
from pathlib import Path
import textwrap
import uuid

ROOT=Path(__file__).resolve().parents[1]


def build(refresh=False):
    source=ROOT/"LAB3-DATASCIENCE/Lab3_ASL_Deep_Learning.ipynb"
    destination=ROOT/"notebooks/Lab3_ASL_CUDA.ipynb"
    original=json.loads(source.read_text(encoding="utf8"))
    notebook=copy.deepcopy(original)
    cells=[]
    def md(text):
        cells.append(dict(cell_type="markdown",metadata={},source=text,id=uuid.uuid4().hex[:12]))
    def code(text):
        cells.append(dict(cell_type="code",metadata={},source=text,outputs=[],execution_count=None,id=uuid.uuid4().hex[:12]))
    md("""# Proyecto 3 — ASL: copia original y entrenamiento CUDA Python

**Paso 5.** El original no se modifica. Esta copia conserva sus celdas fuente
como antecedente, con ejecución CPU completa desactivada para evitar repetir
las tres horas de trabajo. Las salidas originales se retiran de esta copia para
no confundir antecedentes con ejecución nueva.

La implementación CUDA está en `src/asl_cuda/`, distribuida en kernels Python
explícitos y SGEMM cuBLAS sobre buffers CUDA. Preparación y evaluación permanecen
en CPU. Los RUN reales contienen datos/entorno/hashes/logs/eventos/recursos.

Fuente de verdad: `docs/Proyecto 3 CUDA.pdf`. Ver `docs/trazabilidad/implementacion-cuda.md`.
""")
    code('''from pathlib import Path
import os, sys, json
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "docs/Proyecto 3 CUDA.pdf").is_file())
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
EJECUTAR_ORIGINAL = False
EJECUTAR_GPU = os.environ.get("ASL_CUDA_RUN", "0") == "1"
EXPERIMENTO = "EXP-003"
print("Intérprete:", sys.executable)
print("CPU original habilitada:", EJECUTAR_ORIGINAL, "GPU habilitada:", EJECUTAR_GPU)
''')
    md("## A. Código original copiado (referencia, ejecución opcional)\n\nLos textos siguientes pertenecen al laboratorio original. Sus conclusiones no son resultados CUDA nuevos.")
    for i,cell in enumerate(original["cells"]):
        copied=copy.deepcopy(cell)
        copied["id"]=uuid.uuid4().hex[:12]
        copied.setdefault("metadata",{})["origen_celda"]=i
        src="".join(cell.get("source",[]))
        copied["metadata"]["origen_source_sha256"]=hashlib.sha256(src.encode()).hexdigest()
        if cell["cell_type"]=="code":
            copied["source"]="if EJECUTAR_ORIGINAL:\n"+textwrap.indent(src or "pass","    ")
            copied["outputs"],copied["execution_count"]=[],None
        cells.append(copied)
    md("""## B. Implementación CUDA y admisión del entorno

Los kernels K01–K19 y las revisiones de rendimiento se documentan con su código.
Se mantienen FP32, CNN-B base, pesos iniciales, 17,400 imágenes y splits congelados,
batch128 y15épocas (1440actualizaciones). No se vuelve a descargar ni muestrear.

Los productos grandes usan cuBLAS SGEMM; sus grids internos los administra NVIDIA.
Los grids de im2col/col2im, pooling, BN, Dropout, softmax y Adam son explícitos.
La preparación CPU usa6workers y las transferencias se contabilizan.
""")
    code('''from asl_cuda.runtime import device_info, cuda
from asl_cuda import kernels
print(device_info())
print("Kernels Python:", ROOT / "src/asl_cuda/kernels.py")
print("Referencia CPU: resultados/secuencial/EXP-002; no se ejecuta otra campaña CPU")
cuda.close()  # La preinspección no retiene un segundo contexto durante los workers.
''')
    md("""## C. Corridas reales, sin sobrescribir evidencias

La ejecución automática establece `ASL_CUDA_RUN=1`. Cada corrida es un proceso
nuevo en PCUDA. Si la serie ya terminó, se conservan sus observaciones y se
reutilizan para mostrar resultados; no se repiten entrenamientos por abrir el
notebook. Una serie incompleta exige diagnóstico explícito.
""")
    code('''from benchmark_cuda import prepare, run_series, locations
config_path, results_path = locations(EXPERIMENTO)
if EJECUTAR_GPU:
    if not config_path.exists():
        prepare(EXPERIMENTO)
    serie = run_series(EXPERIMENTO)
    print(json.dumps(serie, indent=2, ensure_ascii=False))
else:
    print("Mediciones desactivadas. Configure ASL_CUDA_RUN=1 para crear o consultar la serie.")
''')
    md("## D. Resultados, transferencias y comparación\n\nSe muestran datos reales de los RUN. La validación CPU está anidada en entrenamiento; no se suma dos veces. Los eventos no sustituyen tiempo de pared sincronizado.")
    code('''from IPython.display import display, Markdown, Image
if (results_path / "serie.json").is_file():
    from resumir_cuda import generar
    generar(EXPERIMENTO)
    display(Markdown((results_path / "resultados.md").read_text(encoding="utf8")))
    for figura in sorted((results_path / "graficas").glob("*.png")):
        display(Image(filename=str(figura)))
else:
    print("Todavía no existe una serie. No se inventan resultados.")
''')
    md("""## E. Fuentes de la implementación para revisión

Los siguientes bloques presentan el código que importa el notebook. La fuente
ejecutable única permanece en los módulos; no se mantienen dos implementaciones
independientes. Sus hashes forman parte de la configuración congelada.
""")
    for path in sorted((ROOT/"src/asl_cuda").glob("*.py")):
        md("### "+path.name+"\n\n```python\n"+path.read_text(encoding="utf8")+"\n```\n")
    notebook.update(cells=cells,nbformat=4,nbformat_minor=5)
    notebook["metadata"]=dict(kernelspec=dict(display_name="PCUDA",language="python",name="pcuda"),
        language_info=dict(name="python",version="3.14.3"),
        trazabilidad=dict(original=str(source.relative_to(ROOT)),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                          cambio="CAM-006",exp="EXP-003"))
    destination.parent.mkdir(exist_ok=True)
    if refresh and (ROOT/"configuraciones/EXP-003.json").exists():
        raise RuntimeError("Frozen experiment exists: notebook source cannot be refreshed")
    with destination.open("w" if refresh else "x",encoding="utf8") as f:
        json.dump(notebook,f,ensure_ascii=False,indent=1)
    print(destination)


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--actualizar",action="store_true")
    build(parser.parse_args().actualizar)
