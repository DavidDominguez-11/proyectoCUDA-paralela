"""Build an evidence-only notebook showing the real final GPU RUN outputs.

Preserves the frozen measurement notebook. No training or GPU initialization.
"""
import copy
import hashlib
import json
from pathlib import Path
import uuid

ROOT=Path(__file__).resolve().parents[1]


def build():
    experiment=ROOT/"resultados/paralelo/EXP-003"
    series=json.loads((experiment/"serie.json").read_text(encoding="utf8"))
    if series["estado"]!="completada":raise ValueError("Measured series must be completed")
    run=series["corridas"][-1]["run_id"]
    source=ROOT/"notebooks/Lab3_ASL_CUDA.ipynb"
    notebook=json.loads(source.read_text(encoding="utf8"))
    notebook=copy.deepcopy(notebook)
    for cell in notebook["cells"]:
        if cell["cell_type"]=="code":
            cell["outputs"],cell["execution_count"]=[],None
            text="".join(cell["source"])
            if 'EJECUTAR_GPU = os.environ.get' in text:
                text=text.replace('EJECUTAR_GPU = os.environ.get("ASL_CUDA_RUN", "0") == "1"',
                    'EJECUTAR_GPU = False  # Esta vista solo recupera evidencia real; no entrena.')
            if 'from asl_cuda.runtime import device_info' in text:
                text='''# Hardware REAL registrado durante la corrida, no una consulta nueva.
hardware_registrado = json.loads((ULTIMO_RUN / "hardware.json").read_text(encoding="utf8"))
print(json.dumps(hardware_registrado, indent=2, ensure_ascii=False))
'''
            cell["source"]=text
    extra=[]
    def md(s):extra.append(dict(cell_type="markdown",metadata={},source=s,id=uuid.uuid4().hex[:12]))
    def code(s):extra.append(dict(cell_type="code",metadata={"tipo_evidencia":"salidas_reales_recuperadas","run_id":run},
                                source=s,outputs=[],execution_count=None,id=uuid.uuid4().hex[:12]))
    md(f'''# Salidas reales de la última corrida CUDA

**RUN:** `{run}` · **EXP-003**. El entrenamiento ya fue ejecutado y sus artefactos
están sellados. Las celdas siguientes **leen y muestran las salidas originales**:
no simulan un entrenamiento ni vuelven a entrenar. Se conservan abajo el código
original protegido y la implementación CUDA documentada.
''')
    code(f'''import csv, hashlib
import pandas as pd
from IPython.display import display, Image
ULTIMO_RUN = ROOT / "resultados/paralelo/EXP-003/{run}"
salida_real = json.loads((ULTIMO_RUN / "salida.json").read_text(encoding="utf8"))
assert salida_real["codigo"] == 0
print("RUN real:", ULTIMO_RUN.name)
print("Código de salida del entrenamiento:", salida_real["codigo"])
print("No se ejecuta fit ni se crean corridas nuevas en esta visualización.")
''')
    md("## 1. Salida original de las 15 épocas\n\nTexto recuperado literalmente de `stdout.log` del RUN indicado.")
    code('''log_real = (ULTIMO_RUN / "stdout.log").read_text(encoding="utf8")
lineas_epoca = [linea for linea in log_real.splitlines() if linea.startswith("EPOCH ")]
assert len(lineas_epoca) == 15
for linea in lineas_epoca:
    print(linea)
''')
    md("## 2. Resultados por época\n\nDatos reales de `epocas.csv`. El tiempo de época incluye su validación CPU.")
    code('''epocas_reales = pd.read_csv(ULTIMO_RUN / "epocas.csv")
assert len(epocas_reales) == 15 and int(epocas_reales.iloc[-1]["pasos_acumulados"]) == 1440
display(epocas_reales)
''')
    md("## 3. Métricas finales de prueba y validación")
    code('''metricas_reales = json.loads((ULTIMO_RUN / "metricas.json").read_text(encoding="utf8"))
print("Épocas:", metricas_reales["epocas"])
print("Actualizaciones reales:", metricas_reales["pasos"])
print(f"Test accuracy: {metricas_reales['test']['accuracy']:.8%}")
print(f"Test loss: {metricas_reales['test']['loss']:.10f}")
print("Última validación CPU:")
print(json.dumps(metricas_reales["val_final"], indent=2, ensure_ascii=False))
''')
    md("## 4. Tiempos reales, cómputo y transferencias\n\nSon mediciones de la última corrida. Los intervalos de eventos y la validación están anidados en el entrenamiento: no deben sumarse otra vez al total.")
    code('''tiempos_reales = pd.read_csv(ULTIMO_RUN / "tiempos.csv")
display(tiempos_reales[["fase", "tipo_medicion", "duracion_s", "epocas", "pasos", "motivo"]])
tiempo_train = float(tiempos_reales.loc[tiempos_reales.fase == "entrenamiento_CNN-B", "duracion_s"].iloc[0])
print(f"Entrenamiento completo de esta corrida: {tiempo_train:.6f} s ({tiempo_train/60:.4f} min)")
print("Incluye H2D/D2H, staging y validación CPU; no es tiempo exclusivo de kernels.")
''')
    md("## 5. Curvas de la última corrida, obtenidas de sus datos originales")
    code('''import io
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(epocas_reales.epoca, epocas_reales.train_accuracy, label="Entrenamiento")
axes[0].plot(epocas_reales.epoca, epocas_reales.val_accuracy, label="Validación CPU")
axes[0].set(xlabel="Época", ylabel="Accuracy", ylim=(0, 1.02))
axes[1].plot(epocas_reales.epoca, epocas_reales.train_loss, label="Entrenamiento")
axes[1].plot(epocas_reales.epoca, epocas_reales.val_loss, label="Validación CPU")
axes[1].set(xlabel="Época", ylabel="Pérdida crossentropy")
for ax in axes:
    ax.legend(); ax.grid(alpha=.15)
fig.suptitle(ULTIMO_RUN.name)
fig.tight_layout()
buffer = io.BytesIO()
fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
display(Image(data=buffer.getvalue()))
plt.close(fig)
''')
    md("## 6. Hardware, estado guardado y procedencia verificable")
    code('''hardware_real = json.loads((ULTIMO_RUN / "hardware.json").read_text(encoding="utf8"))
print("GPU:", hardware_real["device"]["name"])
print("Compute capability:", hardware_real["device"]["compute_capability"])
print("cuBLAS:", hardware_real["cublas_version"])
print("Buffers propios (MiB):", hardware_real["buffers_bytes"] / 2**20)
modelo_real = json.loads((ULTIMO_RUN / "modelo-guardado.json").read_text(encoding="utf8"))
print(json.dumps(modelo_real, indent=2, ensure_ascii=False))
proveniencia = pd.DataFrame([{"archivo": nombre, "sha256": hashlib.sha256((ULTIMO_RUN/nombre).read_bytes()).hexdigest()}
    for nombre in ["stdout.log", "epocas.csv", "metricas.json", "tiempos.csv", "hardware.json", "modelo-guardado.json"]])
display(proveniencia)
''')
    # Bootstrap first; immediately show real outputs before the lengthy reference.
    notebook["cells"]=[extra[0],notebook["cells"][1],*extra[1:],*notebook["cells"][2:]]
    notebook["metadata"]["salidas_reales"]=dict(run_id=run,experimento="EXP-003",
        fuente_congelada_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),solo_consulta=True)
    folder=ROOT/"resultados/notebook-salidas"
    folder.mkdir(exist_ok=False)
    template=folder/"plantilla.ipynb"
    with template.open("x",encoding="utf8") as f:json.dump(notebook,f,indent=1,ensure_ascii=False)
    print(template)


if __name__=="__main__":build()
