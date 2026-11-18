"""Two separate, genuinely executable notebooks from the measured worker ASTs."""
import ast
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import nbformat as nb

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'resultados/preparacion/notebooks-entrega'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    global BASE
    ap=argparse.ArgumentParser()
    ap.add_argument('--revision',default='notebooks-entrega')
    ap.add_argument('--cpu-exp',default='EXP-004')
    ap.add_argument('--cuda-exp',default='EXP-005')
    args=ap.parse_args()
    BASE=ROOT/'resultados/preparacion'/args.revision
    BASE.mkdir(parents=True,exist_ok=False)
    delivery=ROOT/'entregables/notebooks'
    delivery.mkdir(exist_ok=True)
    sources={name:(ROOT/'scripts'/f'benchmark_{name}.py').read_text(encoding='utf8') for name in ('cpu','cuda')}
    now=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    original=json.loads((ROOT/'LAB3-DATASCIENCE/Lab3_ASL_Deep_Learning.ipynb').read_text(encoding='utf8'))
    contracts=[]
    for kind,new,old,branch,title in [('cpu',args.cpu_exp,'EXP-002','secuencial','ASL_Secuencial'),
                                       ('cuda',args.cuda_exp,'EXP-003','paralelo','ASL_CUDA')]:
        conf=ROOT/'configuraciones'/f'{new}.json'
        folder=ROOT/'resultados'/branch/new
        if conf.exists() or folder.exists():raise FileExistsError(new)
        folder.mkdir(parents=True)
        config=json.loads((ROOT/'configuraciones'/f'{old}.json').read_text(encoding='utf8'))
        config.update(experimento_id=new,creado_utc=datetime.now(timezone.utc).isoformat(),
                      finalidad='Una ejecucion adicional directamente en notebook; no modifica medias EXP-002/003',
                      responsable='Codex por solicitud del usuario; revision docente de codigo y salidas nativas',
                      origen=dict(experimento=old,configuracion_sha256=sha(ROOT/'configuraciones'/f'{old}.json')))
        if kind=='cuda':
            # A user-opened historical notebook may now contain extra outputs.
            # Preserve it; freeze its current identity only in the NEW experiment.
            config['codigo']['fuentes']={p:v for p,v in config['codigo']['fuentes'].items() if not p.endswith('.ipynb')}
        config['medicion']['repeticiones_notebook']=1
        if kind=='cpu':
            for name in ('manifest_fuente.csv','manifest_splits.csv','label_mapping.json','validacion_datos.json'):
                shutil.copyfile(ROOT/'resultados/secuencial/EXP-002'/name,folder/name)
        run_id=f'RUN-{now}-{kind}-notebook-01'
        run=folder/run_id
        run.mkdir()
        cells=[]
        def md(text):cells.append(nb.v4.new_markdown_cell(text))
        def code(text,**meta):cells.append(nb.v4.new_code_cell(text,metadata=meta))
        md(f'''# {'Solución secuencial CPU' if kind=='cpu' else 'Solución optimizada CUDA Python'} — ASL

David Domínguez · 23712 / Jonathan Diaz · 23873 / Luis Padilla · 23663

**Ejecución nueva directa en este notebook:** `{new}` / `{run_id}`.
Cada salida corresponde al código de la celda ejecutada. No se simula `fit` ni se
sustituye por leer un historial. La corrida es adicional a las cinco mediciones
del informe: sus promedios permanecen intactos. Al final se incluyen los registros
históricos completos identificados como tales.

Kernel: **PCUDA**. Preparación de archivos/imágenes en CPU.
{'Entrenamiento, validación e inferencia en CPU; librerías con paralelismo interno.' if kind=='cpu' else 'Matemática del entrenamiento en GPU; validación y prueba en CPU.'}
Datos congelados:17,400 imágenes,RGB64×64,float32/255;train12180,val2610,test2610.
Se ejecutan15épocas completas por red,batch128,último20,sin EarlyStopping.

Las celdas operativas se extrajeron del AST del controlador medido y se separaron
por fase. Esta ejecución verifica su recorrido en notebook. Para repetir se debe
preparar otro EXP/RUN; este destino no se sobrescribe después de cerrarlo.
''')
        code(f'''from pathlib import Path
import ast, csv, hashlib, json, os, random, subprocess, sys, time
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "docs/Proyecto 3 CUDA.pdf").is_file())
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
EXPERIMENT = ROOT / "resultados/{branch}/{new}"
CONFIG = ROOT / "configuraciones/{new}.json"
NOTEBOOK = ROOT / "LAB3-DATASCIENCE/Lab3_ASL_Deep_Learning.ipynb"
directory = EXPERIMENT / "{run_id}"
repetition = 1
experiment = "{new}"
SEED, IMG_SIZE, NUM_CLASSES = 42, 64, 29
__file__ = str(ROOT / "scripts/benchmark_{kind}.py")
assert not (directory / "salida.json").exists(), "Corrida cerrada: preparar otro EXP/RUN"
from IPython.display import display, Image as NotebookImage, Markdown
from PIL import Image
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
print("Ejecución real:", experiment, directory.name)
print("Intérprete:", sys.executable)
''')
        # All measurement helpers are real executable definitions, not screenshots.
        cpu_tree=ast.parse(sources['cpu'])
        module_imports=[n for n in ast.parse(sources[kind]).body if isinstance(n,(ast.Import,ast.ImportFrom))]
        code('\n'.join(ast.unparse(n) for n in module_imports))
        helpers=[node for node in cpu_tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))
                 and node.name in ('utc','sha','write_json','write_csv','original_namespace','Recorder')]
        field=next(n for n in cpu_tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FIELDS' for t in n.targets))
        md('## 1. Configuración, registro y funciones de medición\n\nLas funciones siguientes registran tiempos y estado real; no entrenan al definirse.')
        code(ast.unparse(field)+'\n\n'+'\n\n'.join(ast.unparse(n) for n in helpers))
        code('''config = json.loads(CONFIG.read_text(encoding="utf8"))
print(json.dumps(config, indent=2, ensure_ascii=False))
from registro_notebook import iniciar_monitor
monitor_stop, monitor_thread = iniciar_monitor(directory)
''')
        # Show the actual original model/data definitions; executing defs is safe.
        md('## 2. Datos y funciones originales utilizadas\n\nCódigo original conservado: carga de imágenes y arquitectura. Las tablas muestran todos los conteos de clases y particiones.')
        definitions=[]
        for cell in original['cells']:
            if cell['cell_type']=='code':
                for node in ast.parse(''.join(cell['source'])).body:
                    if isinstance(node,ast.FunctionDef) and node.name in (('build_sampled_file_list','load_images','build_cnn_a','build_cnn_b','build_fc') if kind=='cpu' else ('build_cnn_b',)):
                        definitions.append(ast.unparse(node))
        code('\n\n'.join(definitions))
        code('''datos_manifest = pd.read_csv(ROOT / "resultados/secuencial/EXP-002/manifest_splits.csv", keep_default_na=False)
display(pd.crosstab(datos_manifest.clase, datos_manifest.split, margins=True))
display(datos_manifest.head(12))
print("17,400 rutas/identidades completas en el manifiesto congelado; no se vuelve a muestrear.")
''')
        if kind=='cuda':
            md('## 3. CUDA Python: configuración y código completo de los módulos\n\nCada celda `%%modulo` ejecuta la fuente Python mostrada en un módulo de memoria. No escribe ni altera los archivos del proyecto. Los kernels y la red usados abajo son estos mismos módulos.')
            code('''import types
from IPython.core.magic import register_cell_magic
pkg = types.ModuleType("asl_cuda")
pkg.__path__ = [str(ROOT / "src/asl_cuda")]
sys.modules["asl_cuda"] = pkg
@register_cell_magic
def modulo(nombre, codigo):
    module = types.ModuleType(nombre)
    module.__file__ = str(ROOT / "src" / (nombre.replace(".", "/") + ".py"))
    module.__package__ = nombre.rsplit(".", 1)[0]
    sys.modules[nombre] = module
    setattr(sys.modules["asl_cuda"], nombre.rsplit(".", 1)[-1], module)
    exec(compile(codigo, module.__file__, "exec"), module.__dict__)
    print("Módulo ejecutado:", nombre, "SHA-256:", hashlib.sha256(codigo.encode()).hexdigest())
''')
            for name in ('runtime','kernels','blas','model'):
                path=ROOT/'src/asl_cuda'/f'{name}.py'
                code(f'%%modulo asl_cuda.{name}\n'+path.read_text(encoding='utf8'))
            tree=ast.parse(sources['cuda'])
            local=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('locations','load_manifest','load_images')]
            code('\n\n'.join(ast.unparse(n) for n in local))
        tree=ast.parse(sources[kind])
        worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
        before=[];body=[]
        for node in worker.body:
            if isinstance(node,ast.With) and 'threadpool_limits' in ast.unparse(node.items[0].context_expr):
                body.extend(node.body)
            elif body:
                body.append(node)
            else:
                before.append(node)
        md('## Configuración efectiva del dispositivo y verificación de entradas')
        code('\n\n'.join(ast.unparse(n) for n in before))
        code('''display(Markdown("### Hardware reconocido durante esta ejecución"))
print((directory / "hardware.json").read_text(encoding="utf8"))
print("No es una consulta histórica: el registro acaba de generarse en esta ejecución.")
''') if kind=='cuda' else None  # CPU hardware is emitted inside before as well.
        if kind=='cpu':code('print((directory / "hardware.json").read_text(encoding="utf8"))')
        # Restore context manager across notebook cells; original numerical ops unchanged.
        code('''threadpool_guard = threadpool_limits(limits=6)
threadpool_guard.__enter__()
print("Límite de bibliotecas matemáticas: 6 hilos")
''')
        def index_matching(text,start=0):
            return next(i for i in range(start,len(body)) if text in ast.unparse(body[i]))
        if kind=='cpu':
            markers=[(0,'Calentamiento y restauración de estados'),
                (index_matching('metrics, training_history'),'Preparación de datos en CPU'),
                (index_matching('class Epochs'),'Callbacks y entrenamiento/evaluación originales'),
                (index_matching("fit_evaluate('CNN-A')"),'Entrenamiento y evaluación CNN-A'),
                (index_matching("fit_evaluate('CNN-B')"),'Entrenamiento y evaluación CNN-B'),
                (index_matching('best_cnn ='),'Diagnóstico de la CNN seleccionada'),
                (index_matching("fit_evaluate('FC-A')"),'Entrenamiento y evaluación FC-A'),
                (index_matching("fit_evaluate('FC-B')"),'Entrenamiento y evaluación FC-B'),
                (index_matching('pca = PCA'),'PCA: ajuste únicamente con train'),
                (index_matching('forests ='),'Random Forest: ajuste y validación'),
                (index_matching('aug_name ='),'Entrenamiento y evaluación con aumento'),
                (index_matching('registry ='),'Selección e inferencia sobre fotografías propias'),
                (index_matching('fingerprints ='),'Identidad de tensores y persistencia de resultados')]
        else:
            markers=[(0,'Warmup y restauración completa'),
                (index_matching("state('preparacion_cpu'"),'FASE CPU: lectura y preparación de imágenes'),
                (index_matching('audit_start ='),'Verificación de tensores frente a CPU'),
                (index_matching("state('entrenamiento_CNN-B'"),'FASE GPU: entrenamiento CNN-B, 15 épocas'),
                (index_matching("state('evaluacion_final_cpu'"),'FASE CPU: evaluación final tras D2H'),
                (index_matching("state('guardado_fuera_de_timing'"),'Persistencia del modelo y configuración de kernels')]
        markers.sort()
        for j,(start,heading) in enumerate(markers):
            end=markers[j+1][0] if j+1<len(markers) else len(body)
            md('## '+heading)
            code('\n\n'.join(ast.unparse(n) for n in body[start:end]),fase_operativa=heading)
            if 'Entrenamiento y evaluación ' in heading:
                model_name=heading.removeprefix('Entrenamiento y evaluación ')
                if model_name=='con aumento':model_expr='aug_name'
                else:model_expr=repr(model_name)
                code(f'''modelo_tabla = {model_expr}
display(pd.DataFrame(training_history[modelo_tabla], index=range(1,16)).rename_axis("epoca"))
print("Prueba:", metrics[modelo_tabla])
print("Arquitectura real:")
models[modelo_tabla].summary()
hist = pd.DataFrame(training_history[modelo_tabla])
fig, ax = plt.subplots(1,2,figsize=(11,3.8))
for name in ("accuracy","val_accuracy"): ax[0].plot(range(1,len(hist)+1),hist[name],label=name)
for name in ("loss","val_loss"): ax[1].plot(range(1,len(hist)+1),hist[name],label=name)
for a in ax: a.set_xlabel("Época"); a.legend(); a.grid(alpha=.15)
ax[0].set_ylabel("Accuracy"); ax[1].set_ylabel("Pérdida")
fig.suptitle(modelo_tabla); fig.tight_layout()
from registro_notebook import mostrar_figura
mostrar_figura(fig)
''')
            if heading=='Preparación de datos en CPU' or 'Verificación de tensores' in heading:
                code('''for etiqueta, (imagenes, clases_y) in inputs.items():
    print(etiqueta, "X", imagenes.shape, imagenes.dtype, "y", clases_y.shape, clases_y.dtype,
          "rango", float(imagenes.min()), float(imagenes.max()))
fig, axes = plt.subplots(2,6,figsize=(12,4))
for i,ax in enumerate(axes.flat):
    ax.imshow(x[i]); ax.set_title(str(y[i])); ax.axis("off")
fig.suptitle("Muestras reales tras preparación CPU")
from registro_notebook import mostrar_figura
mostrar_figura(fig)
''')
        md('## Resultados completos de esta ejecución')
        code('''print((directory / "metricas.json").read_text(encoding="utf8"))
display(pd.read_csv(directory / "tiempos.csv"))
display(pd.read_csv(directory / "epocas.csv"))
''')
        if kind=='cpu':
            code('''confusion_df = pd.read_csv(directory / "confusion.csv", index_col=0)
display(confusion_df)
fig, ax = plt.subplots(figsize=(10,8)); im=ax.imshow(confusion_df.values,cmap="Blues")
ax.set(xticks=range(29),yticks=range(29),xticklabels=confusion_df.columns,yticklabels=confusion_df.index,
       xlabel="Predicción",ylabel="Clase real",title="Matriz de confusión de "+best_cnn)
ax.tick_params(axis="x",labelrotation=90);fig.colorbar(im,ax=ax);fig.tight_layout();mostrar_figura(fig)
propias = pd.read_csv(directory / "predicciones_grupo.csv")
display(propias)
print("Modelo final:",final,"; PCA varianza:",pca.explained_variance_ratio_.sum())
print((directory / "modelos-guardados.json").read_text(encoding="utf8"))
''')
        else:
            code('''reference.summary()
epocas = pd.read_csv(directory / "epocas.csv")
fig, axes = plt.subplots(1,2,figsize=(11,4))
for name in ("train_accuracy","val_accuracy"): axes[0].plot(epocas.epoca,epocas[name],label=name)
for name in ("train_loss","val_loss"): axes[1].plot(epocas.epoca,epocas[name],label=name)
for ax in axes: ax.set_xlabel("Época");ax.grid(alpha=.15);ax.legend()
axes[0].set_ylabel("Accuracy");axes[1].set_ylabel("Pérdida");fig.tight_layout();mostrar_figura(fig)
for nombre in ("transferencias.json","lanzamientos.json","modelo-guardado.json","ordenes.json"):
    print("\\n",nombre);print((directory/nombre).read_text(encoding="utf8"))
''')
        md('## Recursos utilizados, logs y cierre de la ejecución nueva')
        code('''monitor_stop.set(); monitor_thread.join(timeout=15)
threadpool_guard.__exit__(None,None,None)
recursos_actuales = pd.read_csv(directory / "recursos.csv")
display(recursos_actuales)
columnas_cpu=[c for c in recursos_actuales if c.startswith("cpu_logico_")]
fig, axes = plt.subplots(2,1,figsize=(12,6))
for col in columnas_cpu: axes[0].plot(recursos_actuales.transcurrido_s,recursos_actuales[col],label=col)
axes[0].set(ylabel="CPU por lógico (%)",title="Uso real registrado durante esta ejecución")
axes[0].legend(ncol=4,fontsize=7)
axes[1].plot(recursos_actuales.transcurrido_s,recursos_actuales.gpu_util_pct,label="Actividad GPU (%)")
axes[1].plot(recursos_actuales.transcurrido_s,recursos_actuales.gpu_mem_used_mib/4096*100,label="VRAM ocupada (%)")
axes[1].set(xlabel="Tiempo de proceso (s)",ylabel="Porcentaje");axes[1].legend();fig.tight_layout();mostrar_figura(fig)
for captura in sorted(directory.glob("captura-*.png")):
    display(NotebookImage(filename=str(captura)))
    print(captura.name, (captura.with_suffix(".json")).read_text(encoding="utf8"))
write_json(directory / "salida.json",dict(codigo=0,fin_utc=utc(),tipo="ejecucion_directa_notebook"))
print("EJECUCIÓN NUEVA COMPLETA:", directory.name)
''')
        md('## Cinco mediciones históricas del informe — registros íntegros\n\nEsta sección es histórica y está separada de la corrida nueva anterior. Se muestran stdout/stderr completos,épocas,métricas y tiempos de cada RUN. No se vuelven a entrenar ni se mezclan con el promedio original.')
        oldfolder=f'resultados/{branch}/{old}'
        oldseries=json.loads((ROOT/oldfolder/'serie.json').read_text(encoding='utf8'))
        runs=oldseries['runs'] if kind=='cpu' else [r['run_id'] for r in oldseries['corridas']]
        for historical in runs:
            md('### Registro histórico '+historical)
            code(f'''historico = ROOT / "{oldfolder}/{historical}"
print("HISTÓRICO:",historico.name)
print((historico/"stdout.log").read_text(encoding="utf8"))
print("STDERR ORIGINAL:")
print((historico/"stderr.log").read_text(encoding="utf8"))
display(pd.read_csv(historico/"epocas.csv"))
display(pd.read_csv(historico/"tiempos.csv"))
print((historico/"metricas.json").read_text(encoding="utf8"))
''',scrolled=True)
        md('## Comparación histórica del informe (no incluye la corrida nueva)')
        code('''display(pd.read_csv(ROOT/"resultados/paralelo/EXP-003/comparacion_cnn_b.csv"))
display(pd.read_csv(ROOT/"resultados/paralelo/EXP-003/speedup.csv"))
print("Solo CNN-B base comparable. Preparación1→6workers es mejoraCPU;referencia histórica,RNG distinto y sin afirmación de saturaciónSM/VRAM.")
''')
        notebook=nb.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(name='pcuda',language='python',display_name='PCUDA'),
            language_info=dict(name='python',version='3.14.3'),ejecucion_real=dict(experimento=new,run_id=run_id,original=old)))
        template=BASE/f'{title}.plantilla.ipynb'
        nb.write(notebook,template)
        config['codigo']['notebook_entrega_plantilla']=dict(ruta=template.relative_to(ROOT).as_posix(),sha256=sha(template))
        if kind=='cuda':config['codigo']['fuentes'][template.relative_to(ROOT).as_posix()]=sha(template)
        config['codigo']['presentacion_por_fases']='AST del worker; threadpool_limits mantenido entre celdas; resumen/graficas fuera de intervalos medidos'
        conf.write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf8')
        (folder/'configuracion.json').write_bytes(conf.read_bytes())
        (run/'configuracion.json').write_bytes(conf.read_bytes())
        contracts.append(dict(tipo=kind,experimento=new,run_id=run_id,plantilla=str(template),
                              salida=str(delivery/f'{title}.ipynb'),run=str(run),configuracion_sha256=sha(conf)))
    (BASE/'ejecuciones.json').write_text(json.dumps(contracts,indent=2),encoding='utf8')
    print(json.dumps(contracts,indent=2))


if __name__=='__main__':main()
