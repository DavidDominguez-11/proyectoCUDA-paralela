"""CPU benchmark of original notebook functions. No downloads or installations.

prepare freezes data/configuration; run launches isolated measured processes;
worker is the internal execution entrypoint. Summaries are generated separately.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "LAB3-DATASCIENCE/Lab3_ASL_Deep_Learning.ipynb"
EXPERIMENT = ROOT / "resultados/secuencial/EXP-002"
CONFIG = ROOT / "configuraciones/EXP-002.json"
FIELDS = ("experimento_id,run_id,dispositivo,repeticion,fase,tipo_medicion,duracion_s,"
          "inicio_utc,fin_utc,epocas,pasos,muestras,batch_size,incluido_resumen,estado,"
          "configuracion_sha256,datos_sha256,evidencia,motivo").split(",")


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows, fields=None):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def original_namespace():
    """Compile only named definitions, preserving their original source bodies."""
    import numpy as np
    import pandas as pd
    from PIL import Image
    namespace = dict(np=np, pd=pd, Image=Image, random=random, SEED=42, IMG_SIZE=64)
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    names = {"build_sampled_file_list", "load_images", "build_cnn_a", "build_cnn_b", "build_fc"}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                exec(compile(ast.Module(body=[node], type_ignores=[]), f"original/celda-{index}", "exec"), namespace)
    namespace["_notebook"] = notebook
    return namespace


def prepare(dataset):
    import numpy as np
    import pandas as pd
    from PIL import Image
    import psutil
    from sklearn.model_selection import train_test_split
    if CONFIG.exists() or EXPERIMENT.exists():
        raise FileExistsError("Experiment already exists; preserve it and use a new experiment for changes.")
    train = dataset / "asl_alphabet_train/asl_alphabet_train"
    official = dataset / "asl_alphabet_test/asl_alphabet_test"
    if not train.is_dir() or not official.is_dir():
        raise FileNotFoundError("Expected cached ASL train/test directories are missing.")
    namespace = original_namespace()
    classes = sorted(p.name for p in train.iterdir() if p.is_dir())
    counts = {name: len(list((train / name).glob("*.jpg"))) for name in classes}
    if len(classes) != 29 or sum(counts.values()) != 87000:
        raise ValueError(f"Unexpected dataset: {counts}")
    sampled = namespace["build_sampled_file_list"](train, classes, 600)
    df_train, temp = train_test_split(sampled, test_size=.30, stratify=sampled.label, random_state=42)
    df_val, df_test = train_test_split(temp, test_size=.50, stratify=temp.label, random_state=42)
    EXPERIMENT.mkdir(parents=True)
    source_rows, splits, hashes = [], [], {}
    for split, frame in (("train", df_train), ("val", df_val), ("test", df_test)):
        for order, item in enumerate(frame.itertuples()):
            path = Path(item.filepath)
            value = sha(path)
            with Image.open(path) as image:
                width, height = image.size
                mode = image.mode
                image.verify()
            relative = path.relative_to(dataset).as_posix()
            source_rows.append(dict(dataset_id="DAT-01-v1", ruta_relativa=relative,
                bytes=path.stat().st_size, sha256=value, clase=item.label,
                ancho=width, alto=height, modo=mode, estado="verificado", motivo=""))
            splits.append(dict(dataset_id=f"DAT-03-{EXPERIMENT.name}", ruta_relativa=relative, sha256=value,
                clase=item.label, label_idx=classes.index(item.label), split=split, orden=order))
            hashes.setdefault(value, []).append(dict(split=split, ruta=relative))
    write_csv(EXPERIMENT / "manifest_fuente.csv", source_rows)
    write_csv(EXPERIMENT / "manifest_splits.csv", splits)
    write_json(EXPERIMENT / "label_mapping.json", dict(enumerate(classes)))
    duplicates = [items for items in hashes.values() if len(items) > 1]
    write_json(EXPERIMENT / "validacion_datos.json", dict(
        fecha_utc=utc(), counts=counts, selected=len(splits),
        official_images=len(list(official.glob("*.jpg"))),
        selected_bytes=sum(row["bytes"] for row in source_rows),
        splits={k: sum(row["split"] == k for row in splits) for k in ("train", "val", "test")},
        duplicate_groups=duplicates,
        cross_split_duplicate_groups=sum(len({x["split"] for x in group}) > 1 for group in duplicates),
        duplicates_policy="Preserve original sampled data; disclose duplicates without silently changing splits."))
    import winreg
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
        cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    config = dict(
        es_plantilla=False, experimento_id=EXPERIMENT.name, creado_utc=utc(),
        pregunta="Tiempo CPU de las operaciones de procesamiento del laboratorio original completo, con carga fija.",
        responsable="Codex, por solicitud del usuario; revision humana pendiente",
        requisitos=["REQ-03", "REQ-07", "REQ-12"], decisiones=["DEC-011", "DEC-012", "DEC-013"],
        codigo=dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    notebook_sha256=sha(NOTEBOOK), benchmark_sha256=sha(__file__)),
        datos=dict(dataset_id="DAT-01-v1", version=1, ruta=str(dataset),
            manifest_fuente_sha256=sha(EXPERIMENT / "manifest_fuente.csv"),
            manifest_splits_sha256=sha(EXPERIMENT / "manifest_splits.csv"),
            label_mapping_sha256=sha(EXPERIMENT / "label_mapping.json"),
            samples_per_class=600, splits=dict(train=12180, val=2610, test=2610),
            transformaciones=dict(rgb=True, tamano=[64,64], dtype="float32", etiquetas="int64",
                normalizacion="/255.0", resize="Pillow RGB default BICUBIC; original unchanged",
                exif="no transpose; original behavior"),
            augmentation="Original layers inside CPU model; selected CNN architecture; not CUDA design"),
        modelos=["CNN-A", "CNN-B", "FC-A", "FC-B", "mejor CNN + augmentation", "PCA+RF-A", "PCA+RF-B"],
        carga=dict(epocas_fijas=15, pasos_por_epoca=96, batch_size=128,
                   descartar_ultimo_batch=False, early_stopping=False,
                   validacion="2610 samples each epoch; nested timing retained"),
        aleatoriedad=dict(seed=42, semillas_modelos={"CNN-A":42,"CNN-B":43,"FC-A":44,"FC-B":45,
                        "CNN-A+aug":46,"CNN-B+aug":47}, shuffle=True, determinismo=True),
        precision=dict(dtype="float32", mixed_precision=False, xla=False, oneDNN=True, tf32="no_aplica_CPU"),
        cpu=dict(modelo=cpu_name, nucleos_fisicos=psutil.cpu_count(logical=False),
                 nucleos_logicos=psutil.cpu_count(), ram_bytes=psutil.virtual_memory().total,
                 hilos_matematicos=6, intra_op=6, inter_op=1, workers_preparacion=1,
                 afinidad="sin restriccion adicional", gpu_visible=False),
        medicion=dict(repeticiones_iniciales=5, extensiones=[7,9],
            criterio="At n=5 and n=7 extend if Student-t 95% relative half-width of pipeline_total exceeds 5%; cap n=9, disclose if unmet.",
            warmup="Each candidate model: fit 148 train +128 val, 1 epoch, batch128; restore ALL model/optimizer variables and seeds.",
            procesos_nuevos=True, temporizador="time.perf_counter_ns; CPU synchronous results",
            cache="OS page cache not flushed; manifests pre-read all selected files; no cold-disk claim",
            exclusiones="No timing outlier removal; failed processes retained and reported",
            total="Continuous wall time from measured preparation to completion of group inference; excludes imports, model construction/warmup, hashes, checkpoint writes and plots",
            nested="Validation callback durations are nested inside each fit; do not add twice",
            monitorizacion="psutil complete launcher/worker descendant tree at 1-second intervals; summed CPU and RSS working sets (shared memory may be counted twice)"),
        validacion=dict(shapes=True, finite=True, input_hashes=True, initial_state_hashes=True,
                        accuracy_tolerance_across_repetitions=1e-6, atol_futuro_gpu=1e-5, rtol_futuro_gpu=1e-4),
        comandos=dict(directorio=str(ROOT),
            preparar=f'.\\PCUDA\\Scripts\\python.exe -X utf8 scripts/benchmark_cpu.py prepare --experimento {EXPERIMENT.name} --dataset "{dataset}"',
            cpu=f".\\PCUDA\\Scripts\\python.exe -X utf8 scripts/benchmark_cpu.py run --experimento {EXPERIMENT.name}",
            analisis=f".\\PCUDA\\Scripts\\python.exe -X utf8 scripts/resumir_cpu.py --experimento {EXPERIMENT.name}"))
    write_json(CONFIG, config)
    (EXPERIMENT / "configuracion.json").write_bytes(CONFIG.read_bytes())
    print(json.dumps(dict(estado="preparado", count=len(splits), duplicates=len(duplicates),
                         cpu=cpu_name, config_sha256=sha(CONFIG)), ensure_ascii=False), flush=True)


class Recorder:
    def __init__(self, directory, config, repetition):
        self.directory, self.config, self.repetition = directory, config, repetition
        self.rows = []
        self.epochs = []

    def state(self, phase, **more):
        write_json(self.directory / "estado.json", dict(utc=utc(), fase=phase, **more))
        print(f"[{utc()}] {phase} {more}", flush=True)

    def measure(self, phase, function, **load):
        self.state(phase)
        start_utc, start = utc(), time.perf_counter_ns()
        value = function()
        end, end_utc = time.perf_counter_ns(), utc()
        self.add(phase, (end-start)/1e9, start_utc, end_utc, **load)
        return value

    def add(self, phase, seconds, start, end, **load):
        row = dict.fromkeys(FIELDS, "")
        row.update(experimento_id=self.config["experimento_id"], run_id=self.directory.name, dispositivo="cpu",
            repeticion=self.repetition, fase=phase, tipo_medicion="pared", duracion_s=seconds,
            inicio_utc=start, fin_utc=end, incluido_resumen="true", estado="completada",
            configuracion_sha256=sha(CONFIG), datos_sha256=self.config["datos"]["manifest_splits_sha256"],
            evidencia=(self.directory / "tiempos.csv").relative_to(ROOT).as_posix())
        row.update(load)
        self.rows.append(row)
        write_csv(self.directory / "tiempos.csv", self.rows, FIELDS)
        print(f"RESULTADO {phase}: {seconds:.6f} s", flush=True)


def worker(directory, repetition):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if sha(NOTEBOOK) != config["codigo"]["notebook_sha256"] or sha(__file__) != config["codigo"]["benchmark_sha256"]:
        raise ValueError("Source identity differs from frozen experiment.")
    threads = config["cpu"]["hilos_matematicos"]
    for key, value in {"CUDA_VISIBLE_DEVICES":"-1", "OMP_NUM_THREADS":str(threads),
                       "MKL_NUM_THREADS":str(threads), "OPENBLAS_NUM_THREADS":str(threads),
                       "TF_NUM_INTRAOP_THREADS":str(threads), "TF_NUM_INTEROP_THREADS":"1",
                       "TF_DETERMINISTIC_OPS":"1", "TF_ENABLE_ONEDNN_OPTS":"1"}.items():
        os.environ[key] = value
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix
    from threadpoolctl import threadpool_limits, threadpool_info
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    tf.keras.mixed_precision.set_global_policy("float32")
    if tf.config.get_visible_devices("GPU"):
        raise ValueError("GPU visible in CPU benchmark.")
    namespace = original_namespace()
    namespace.update(keras=tf.keras, layers=tf.keras.layers, NUM_CLASSES=29)
    classes = list(json.loads((EXPERIMENT / "label_mapping.json").read_text(encoding="utf-8")).values())
    namespace["label_to_idx"] = {name:i for i,name in enumerate(classes)}
    data_root = Path(config["datos"]["ruta"])
    if sha(EXPERIMENT / "manifest_splits.csv") != config["datos"]["manifest_splits_sha256"]:
        raise ValueError("Split manifest changed.")
    manifest = pd.read_csv(EXPERIMENT / "manifest_splits.csv", keep_default_na=False)
    frames = {}
    for split in ("train", "val", "test"):
        frame = manifest[manifest.split == split].sort_values("orden").copy()
        frame["filepath"] = frame.ruta_relativa.map(lambda s:str(data_root / s))
        frame["label"] = frame.clase
        frames[split] = frame
    # Hash each selected source outside the measured interval to validate inputs.
    for item in manifest.itertuples():
        if sha(data_root / item.ruta_relativa) != item.sha256:
            raise ValueError(f"Input changed: {item.ruta_relativa}")
    record = Recorder(directory, config, repetition)
    (directory / "configuracion.json").write_bytes(CONFIG.read_bytes())
    write_json(directory / "hardware.json", dict(cpu=config["cpu"], tensorflow=tf.__version__,
        visibles=[str(d) for d in tf.config.get_visible_devices()],
        logical=[str(d) for d in tf.config.list_logical_devices()],
        test_device=tf.constant([1.]).device, build=tf.sysconfig.get_build_info(),
        threadpools=threadpool_info(), variables={k:os.environ.get(k) for k in (
            "CUDA_VISIBLE_DEVICES","OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
            "TF_NUM_INTRAOP_THREADS","TF_NUM_INTEROP_THREADS","TF_DETERMINISTIC_OPS","TF_ENABLE_ONEDNN_OPTS")}))
    models, warmup, initial = {}, {}, {}
    warm_x, warm_y = namespace["load_images"](frames["train"].head(148))
    warm_vx, warm_vy = namespace["load_images"](frames["val"].head(128))
    data_directory = ROOT / "data" / EXPERIMENT.name / directory.name
    data_directory.mkdir(parents=True, exist_ok=False)

    def augmented(name):
        namespace["best_cnn_name"] = name
        tree = ast.parse("".join(namespace["_notebook"]["cells"][52]["source"]))
        nodes = [node for node in tree.body if (
            isinstance(node, ast.Assign) and all(not isinstance(t, ast.Name) or t.id != "history_aug" for t in node.targets)
        ) or (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
              and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "compile")]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "original/celda-52-configuracion", "exec"), namespace)
        return namespace["cnn_aug"]

    # Every possible candidate is compiled/warmed before starting the total timer.
    with threadpool_limits(limits=threads):
        for name, seed in config["aleatoriedad"]["semillas_modelos"].items():
            record.state("inicializacion_y_warmup", modelo=name)
            tf.keras.utils.set_random_seed(seed)
            start = time.perf_counter()
            if name == "CNN-A": model = namespace["build_cnn_a"]()
            elif name == "CNN-B": model = namespace["build_cnn_b"]()
            elif name == "FC-A": model = namespace["build_fc"]([128,64], dropout=.3, name="FC_A_pequena")
            elif name == "FC-B": model = namespace["build_fc"]([512,256,128], dropout=.4, name="FC_B_grande")
            else: model = augmented(name.split("+")[0])
            model.jit_compile = False
            model.optimizer.build(model.trainable_variables)
            variables = list(model.variables) + list(model.optimizer.variables)
            saved = [v.numpy().copy() for v in variables]
            h = hashlib.sha256()
            for array in saved:
                h.update(str((array.shape,str(array.dtype))).encode())
                h.update(array.tobytes())
            fingerprint = h.hexdigest()
            state_path = data_directory / f"{name}.estado-inicial.npz"
            np.savez(state_path, *saved)
            model.fit(warm_x, warm_y, validation_data=(warm_vx,warm_vy), epochs=1,
                      batch_size=128, shuffle=False, verbose=0)
            if len(variables) != len(list(model.variables) + list(model.optimizer.variables)):
                raise ValueError("Warmup created unexpected state; cannot restore exactly.")
            for variable, value in zip(variables,saved): variable.assign(value)
            if any(not np.array_equal(v.numpy(), a) for v,a in zip(variables,saved)):
                raise ValueError("Warmup state restoration failed.")
            initial[name] = dict(state_sha256=fingerprint, archivo=state_path.relative_to(ROOT).as_posix(),
                archivo_sha256=sha(state_path), variables=len(variables),
                arquitectura=model.to_json(), compile=model.get_compile_config())
            warmup[name] = time.perf_counter()-start
            models[name] = model
            del saved
        write_json(directory / "estados-iniciales.json", initial)
        write_json(directory / "warmup.json", warmup)
        metrics, training_history, epoch_rows = {}, {}, []
        inputs = {}
        wall_start_utc, wall_start = utc(), time.perf_counter_ns()

        def load_all():
            for split in ("train","val","test"):
                inputs[split] = namespace["load_images"](frames[split])
        record.measure("preparacion_cpu", load_all, muestras=17400)
        x,y = inputs["train"]; vx,vy = inputs["val"]; tx,ty = inputs["test"]

        class Epochs(tf.keras.callbacks.Callback):
            def __init__(self,name):
                super().__init__(); self.name=name; self.validation=0.; self.epoch_validation=0.
            def on_epoch_begin(self,epoch,logs=None):
                self.start=time.perf_counter(); self.epoch_validation=0.
                record.state("entrenamiento", modelo=self.name, epoca=epoch+1)
            def on_test_begin(self,logs=None): self.test_start=time.perf_counter()
            def on_test_end(self,logs=None):
                seconds=time.perf_counter()-self.test_start
                self.validation+=seconds; self.epoch_validation+=seconds
            def on_epoch_end(self,epoch,logs=None):
                epoch_rows.append(dict(modelo=self.name,epoca=epoch+1,
                    duracion_s=time.perf_counter()-self.start, validacion_s=self.epoch_validation, **logs))
                print(f"EPOCH {self.name} {epoch+1}/15 {logs}",flush=True)

        def fit_evaluate(name):
            tf.keras.utils.set_random_seed(config["aleatoriedad"]["semillas_modelos"][name])
            callback=Epochs(name)
            history=record.measure(f"entrenamiento_{name}", lambda:models[name].fit(
                x,y,validation_data=(vx,vy),epochs=15,batch_size=128,shuffle=True,
                callbacks=[callback],verbose=0), epocas=15,pasos=1440,muestras=12180*15,batch_size=128,
                motivo="Incluye validacion; callbacks y duraciones anidadas conservados.")
            training_history[name]=history.history
            record.add(f"validacion_anidada_{name}",callback.validation,"","",epocas=15,muestras=2610*15,
                motivo="Subintervalo de fit: no sumar al total ni tratar como fase adicional.")
            metrics[name]=record.measure(f"evaluacion_{name}",lambda:models[name].evaluate(
                tx,ty,verbose=0,return_dict=True),muestras=2610)
            write_csv(directory / "epocas.csv", epoch_rows)
            write_json(directory / "metricas-parciales.json",metrics)
        fit_evaluate("CNN-A"); fit_evaluate("CNN-B")
        best_cnn=max(("CNN-A","CNN-B"),key=lambda name:metrics[name]["accuracy"])
        probs=record.measure("diagnostico_cnn",lambda:models[best_cnn].predict(tx,verbose=0),muestras=2610)
        confusion=confusion_matrix(ty,np.argmax(probs,axis=1))
        fit_evaluate("FC-A"); fit_evaluate("FC-B")
        pca=PCA(n_components=150,random_state=42)
        projections={}
        def project():
            projections["train"]=pca.fit_transform(x.reshape(len(x),-1))
            projections["val"]=pca.transform(vx.reshape(len(vx),-1))
            projections["test"]=pca.transform(tx.reshape(len(tx),-1))
        record.measure("pca",project,muestras=17400)
        forests={}
        forest_val={}
        for name,trees,depth in (("RF-A",100,None),("RF-B",300,30)):
            forest=RandomForestClassifier(n_estimators=trees,max_depth=depth,random_state=42,n_jobs=-1)
            record.measure(f"entrenamiento_{name}",lambda:forest.fit(projections["train"],y),muestras=12180)
            pred=record.measure(f"validacion_{name}",lambda:forest.predict(projections["val"]),muestras=2610)
            forest_val[name]=accuracy_score(vy,pred); forests[name]=forest
        best_rf=max(forest_val,key=forest_val.get)
        rf_pred=record.measure("evaluacion_RF",lambda:forests[best_rf].predict(projections["test"]),muestras=2610)
        metrics[best_rf]=dict(accuracy=accuracy_score(ty,rf_pred),val_accuracy=forest_val[best_rf])
        aug_name=best_cnn+"+aug"
        fit_evaluate(aug_name)
        registry={name:models[name] for name in ("CNN-A","CNN-B","FC-A","FC-B",aug_name)}
        registry[best_rf]=forests[best_rf]
        final=max(metrics,key=lambda name:metrics[name]["accuracy"])
        own_files=sorted((ROOT/"LAB3-DATASCIENCE/imagenes_grupo").glob("*.*"))
        from PIL import Image
        own_results=[]
        def prepare_one(path):
            with Image.open(path) as image:
                return np.asarray(image.convert("RGB").resize((64,64)),dtype=np.float32)/255.
        def predict_one(name,array):
            if name.startswith("RF"):
                return int(registry[name].predict(pca.transform(array.reshape(1,-1)))[0])
            return int(np.argmax(registry[name].predict(array[np.newaxis,...],verbose=0)[0]))
        def own_inference():
            # Same repeated work as cells 62 and 64; plotting is excluded.
            for path in own_files:
                pred=predict_one(final,prepare_one(path))
                own_results.append(dict(modelo=final,etapa="final",archivo=path.name,
                    real=path.stem.split("_")[0],prediccion=classes[pred]))
            arrays=[prepare_one(path) for path in own_files]
            for name in registry:
                for path,array in zip(own_files,arrays):
                    pred=predict_one(name,array)
                    own_results.append(dict(modelo=name,etapa="todos",archivo=path.name,
                        real=path.stem.split("_")[0],prediccion=classes[pred]))
        record.measure("inferencia_grupo",own_inference,muestras=len(own_files),
                       motivo="15+90 predicciones individuales; 30 preprocesamientos; sin renderizar figuras.")
        wall_end, wall_end_utc = time.perf_counter_ns(),utc()
        record.add("pipeline_total",(wall_end-wall_start)/1e9,wall_start_utc,wall_end_utc,
                   epocas=75,pasos=7200,muestras=17400,batch_size=128)
        # Hashing and artifact persistence occur after the complete processing interval.
        fingerprints={}
        for split,(array,labels) in inputs.items():
            if array.shape!=(len(frames[split]),64,64,3) or not np.isfinite(array).all():
                raise ValueError("Invalid prepared tensors.")
            fingerprints[split]=dict(shape=list(array.shape),dtype=str(array.dtype),
                images_sha256=hashlib.sha256(array.tobytes()).hexdigest(),
                labels_sha256=hashlib.sha256(labels.tobytes()).hexdigest(),min=float(array.min()),max=float(array.max()))
        write_json(directory/"tensores.json",fingerprints)
        write_json(directory/"historiales.json",training_history)
        write_csv(directory/"predicciones_grupo.csv",own_results)
        write_csv(directory/"confusion.csv",[dict(clase=classes[i],**{c:int(v) for c,v in zip(classes,row)}) for i,row in enumerate(confusion)])
        write_json(directory/"metricas.json",dict(modelos=metrics,mejor_cnn=best_cnn,mejor_rf=best_rf,
            modelo_final=final,pca_varianza=float(pca.explained_variance_ratio_.sum()),
            grupo_accuracy={name:sum(row["real"].upper()==row["prediccion"].upper() for row in own_results
                if row["etapa"]=="todos" and row["modelo"]==name)/len(own_files) for name in registry}))
        model_paths={}
        for name in registry:
            if name.startswith("RF"): continue
            path=data_directory/f"{name}.keras"; registry[name].save(path)
            model_paths[name]=dict(ruta=path.relative_to(ROOT).as_posix(),sha256=sha(path))
        import joblib
        for name,obj in ((best_rf,forests[best_rf]),("PCA",pca)):
            path=data_directory/f"{name}.joblib"; joblib.dump(obj,path)
            model_paths[name]=dict(ruta=path.relative_to(ROOT).as_posix(),sha256=sha(path))
        write_json(directory/"modelos-guardados.json",model_paths)
        record.state("completada",pipeline_s=(wall_end-wall_start)/1e9)


def sample_process_tree(process, known):
    """Count the actual venv child interpreter, not just Windows' tiny launcher."""
    import psutil
    cpu, rss, pids = 0., 0, []
    try:
        current=[process,*process.children(recursive=True)]
    except psutil.NoSuchProcess:
        return cpu,rss,pids
    for item in current:
        try:
            if item.pid not in known:
                known[item.pid]=item
                item.cpu_percent()
            else:
                item=known[item.pid]
                cpu+=item.cpu_percent()
            rss+=item.memory_info().rss
            pids.append(item.pid)
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            continue
    return cpu,rss,pids


def run_all():
    import psutil
    from scipy.stats import t
    from statistics import mean,stdev
    config=json.loads(CONFIG.read_text(encoding="utf-8"))
    experiment_log=EXPERIMENT/"serie.json"
    if experiment_log.exists():
        raise FileExistsError("Series already started; preserve it. Resume requires explicit audit of completed runs.")
    completed=[]
    write_json(experiment_log,dict(inicio_utc=utc(),estado="en_curso",runs=completed))
    for repetition in range(1,10):
        run_id=f"RUN-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-cpu-{repetition:02d}"
        directory=EXPERIMENT/run_id
        directory.mkdir()
        command=[sys.executable,"-X","utf8",str(Path(__file__).resolve()),"worker",
                 "--directorio",str(directory),"--repeticion",str(repetition),"--experimento",EXPERIMENT.name]
        write_json(directory/"invocacion.json",dict(comando=command,directorio=str(ROOT),inicio_utc=utc()))
        context=subprocess.run([sys.executable,"-X","utf8",str(ROOT/"scripts/capturar_contexto.py"),
            "--destino",str(directory/"contexto")],cwd=ROOT,capture_output=True,text=True,encoding="utf-8")
        (directory/"contexto-captura.log").write_text(context.stdout+context.stderr,encoding="utf-8")
        if context.returncode: raise RuntimeError("Context capture failed.")
        print(f"INICIO {run_id}",flush=True)
        with (directory/"stdout.log").open("w",encoding="utf-8") as out, (directory/"stderr.log").open("w",encoding="utf-8") as err, (directory/"recursos.csv").open("w",encoding="utf-8",newline="") as monitor:
            proc=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err)
            process=psutil.Process(proc.pid)
            resource_fields=["utc","transcurrido_s","cpu_proceso_pct","rss_bytes","ram_disponible_bytes","cpu_total_pct","pids"]+[f"cpu_{i}_pct" for i in range(psutil.cpu_count())]
            writer=csv.DictWriter(monitor,fieldnames=resource_fields); writer.writeheader()
            psutil.cpu_percent(percpu=True)
            known={}
            sample_process_tree(process,known)
            start=time.perf_counter(); last_notice=0.
            while proc.poll() is None:
                time.sleep(1)
                try:
                    cores=psutil.cpu_percent(percpu=True)
                    process_cpu,process_rss,pids=sample_process_tree(process,known)
                    writer.writerow(dict(utc=utc(),transcurrido_s=time.perf_counter()-start,
                        cpu_proceso_pct=process_cpu,rss_bytes=process_rss,
                        ram_disponible_bytes=psutil.virtual_memory().available,
                        cpu_total_pct=sum(cores)/len(cores),pids=";".join(map(str,pids)),
                        **{f"cpu_{i}_pct":v for i,v in enumerate(cores)}))
                    monitor.flush()
                except psutil.NoSuchProcess: break
                if time.perf_counter()-last_notice>30:
                    try: state=(directory/"estado.json").read_text(encoding="utf-8")
                    except (FileNotFoundError,json.JSONDecodeError): state="importando/verificando entradas"
                    print(f"PROGRESO {run_id}: {state}",flush=True); last_notice=time.perf_counter()
            exit_code=proc.wait()
        write_json(directory/"salida.json",dict(fin_utc=utc(),codigo_salida=exit_code))
        if exit_code:
            write_json(experiment_log,dict(estado="fallida",runs=completed,fallo=run_id))
            raise RuntimeError(f"Run failed: {run_id}; inspect stderr.log. No automatic exclusion.")
        completed.append(run_id)
        totals=[]
        for name in completed:
            with (EXPERIMENT/name/"tiempos.csv").open(encoding="utf-8",newline="") as handle:
                totals.extend(float(row["duracion_s"]) for row in csv.DictReader(handle) if row["fase"]=="pipeline_total")
        half=(float(t.ppf(.975,len(totals)-1))*stdev(totals)/len(totals)**.5/mean(totals)) if len(totals)>1 else None
        write_json(experiment_log,dict(estado="en_curso",runs=completed,media_s=mean(totals),semiancho_relativo_ic95=half))
        print(f"FIN {run_id}; n={len(totals)}; media={mean(totals):.3f}s; semiancho_relativo={half}",flush=True)
        if repetition in (5,7,9) and (half<=.05 or repetition==9):
            write_json(experiment_log,dict(estado="completada",fin_utc=utc(),runs=completed,
                media_s=mean(totals),semiancho_relativo_ic95=half,criterio_precision_cumplido=half<=.05))
            break


def main():
    global EXPERIMENT,CONFIG
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modo",choices=["prepare","run","worker"])
    parser.add_argument("--dataset",type=Path)
    parser.add_argument("--directorio",type=Path)
    parser.add_argument("--repeticion",type=int)
    parser.add_argument("--experimento",default="EXP-002")
    args=parser.parse_args()
    if not args.experimento.startswith("EXP-") or not args.experimento[4:].isdigit():
        raise ValueError("Expected experiment ID EXP-NNN.")
    EXPERIMENT=ROOT/"resultados/secuencial"/args.experimento
    CONFIG=ROOT/"configuraciones"/(args.experimento+".json")
    if args.modo=="prepare": prepare(args.dataset)
    elif args.modo=="run": run_all()
    else:
        try: worker(args.directorio,args.repeticion)
        except Exception:
            write_json(args.directorio/"error.json",dict(utc=utc(),traceback=traceback.format_exc()))
            raise


if __name__=="__main__": main()
