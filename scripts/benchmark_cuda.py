"""Reproducible CNN-B CUDA series, preserving the historical CPU experiment.

prepare freezes config/sources; run executes new processes and monitors them;
worker performs one complete 15-epoch run. No downloads or CPU training.
"""
from __future__ import annotations
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_cpu import FIELDS, sha, write_csv, write_json, utc


def locations(experiment):
    if not experiment.startswith("EXP-") or not experiment[4:].isdigit():
        raise ValueError("Invalid experiment identity")
    return ROOT / "configuraciones" / (experiment + ".json"), ROOT / "resultados/paralelo" / experiment


def prepare(experiment="EXP-003"):
    conf, folder = locations(experiment)
    if conf.exists() or folder.exists():
        raise FileExistsError("Experiment already exists; do not overwrite")
    cpu = json.loads((ROOT / "configuraciones/EXP-002.json").read_text(encoding="utf8"))
    sources = [ROOT / "scripts/benchmark_cuda.py", ROOT / "scripts/capturar_contexto.py",
               *sorted((ROOT / "src/asl_cuda").glob("*.py")),
               ROOT / "notebooks/Lab3_ASL_CUDA.ipynb"]
    if any(not p.is_file() for p in sources):
        raise FileNotFoundError("Notebook/modules must exist before freezing")
    config = dict(es_plantilla=False, experimento_id=experiment, creado_utc=utc(),
        pregunta="Tiempo real CNN-B CUDA Python, FP32, mismos datos/pasos que fase CPU EXP-002",
        responsable="Codex, por solicitud del usuario; revision humana pendiente",
        requisitos=["REQ-06","REQ-07","REQ-09","REQ-12"], decisiones=["DEC-021","DEC-022","DEC-023"],
        codigo=dict(fuentes={p.relative_to(ROOT).as_posix():sha(p) for p in sources},
                    notebook_original_sha256=cpu["codigo"]["notebook_sha256"]),
        datos=cpu["datos"],
        alcance=dict(modelo="CNN-B", augmentation=False, epocas=15, batch=128,
                     train=12180, val=2610, test=2610, pasos=1440, ultimo_batch=20,
                     validacion="CPU cada epoca batch128, incluida en entrenamiento", evaluacion="CPU batch32"),
        cpu=dict(workers_preparacion=6, tareas_pendientes=12, intra_op=6, inter_op=1, blas=6),
        precision=dict(dtype="float32", math="CUBLAS_PEDANTIC_MATH", atomics="CUBLAS_ATOMICS_NOT_ALLOWED",
                       mixed_precision=False, tf32=False, xla=False),
        gpu=dict(backend="numba-cuda+cuBLAS", toolkit="13.1", stream=1, overlap=False,
                 max_buffers_mib=1408, workspace_columnas_mib=576),
        aleatoriedad=dict(seed=43, shuffle="NumPy PCG64, permutacion por epoca",
                          dropout="Bernoulli por elemento, probabilidades originales; claves [43,epoca,lote,capa]",
                          igualdad_mascaras_historicas=False),
        medicion=dict(repeticiones=5, extensiones=[7,9], semiancho_ic95_objetivo_pct=5,
                      principal="pared sincronizada, incluye staging/mascaras/H2D/validacionCPU/D2H",
                      eventos="intervalos stream; no son tiempo activo exclusivo de kernels",
                      warmup="148train:128+20,128val,32test; restauracion completa y verificada",
                      referencia_cpu="Cinco fases entrenamiento_CNN-B de EXP-002, no otra campana CPU",
                      cache="SO sin vaciar; hashes leidos antes de preparar", monitorizacion_s=1),
        entorno=dict(python=sys.version, paquetes={d.metadata["Name"]:d.version for d in importlib.metadata.distributions()}),
        validacion=dict(operadores="resultados/preparacion/cuda/operadores-002/validacion.json",
                        adjuntos="resultados/preparacion/cuda/adjuntos-001/validacion.json",
                        integracion="resultados/preparacion/cuda/integracion-002/validacion.json",
                        rutas="resultados/preparacion/cuda/rutas-004/validacion.json"))
    # Explicitly correct history's augmentation field for this single-model workload.
    config["datos"] = dict(config["datos"], augmentation="No incluido: CNN-B base")
    for key in ("operadores","adjuntos","integracion"):
        evidence = ROOT / config["validacion"][key]
        check = json.loads(evidence.read_text(encoding="utf8"))
        if check["estado"] != "aprobado":
            raise ValueError(f"Required validation failed: {key}")
        config["validacion"][key + "_sha256"] = sha(evidence)
    conf.parent.mkdir(exist_ok=True)
    folder.mkdir(parents=True, exist_ok=False)
    write_json(conf,config)
    (folder / "configuracion.json").write_bytes(conf.read_bytes())
    print("PREPARADO",experiment,sha(conf),flush=True)
    return config


def load_manifest(config):
    path=ROOT / "resultados/secuencial/EXP-002/manifest_splits.csv"
    if sha(path)!=config["datos"]["manifest_splits_sha256"]:
        raise ValueError("Split manifest identity changed")
    with path.open(encoding="utf8",newline="") as f:
        rows=list(csv.DictReader(f))
    result={s:sorted([r for r in rows if r["split"]==s],key=lambda r:int(r["orden"])) for s in ("train","val","test")}
    for s,n in (("train",12180),("val",2610),("test",2610)):
        if len(result[s])!=n:
            raise ValueError("Data count changed")
    return result


def load_images(rows, root, workers=6):
    """Same Pillow transformation as original; fixed order, bounded independent I/O."""
    import numpy as np
    from PIL import Image
    X=np.zeros((len(rows),64,64,3),np.float32)
    y=np.zeros(len(rows),np.int64)
    def read(row):
        with Image.open(root/row["ruta_relativa"]) as image:
            value=np.asarray(image.convert("RGB").resize((64,64)),dtype=np.float32)/255.0
        return value,int(row["label_idx"])
    if workers==1:
        for i,row in enumerate(rows):
            X[i],y[i]=read(row)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending=deque()
            iterator=iter(rows)
            for _ in range(min(12,len(rows))):
                pending.append(pool.submit(read,next(iterator)))
            i=0
            while pending:
                X[i],y[i]=pending.popleft().result()
                i+=1
                row=next(iterator,None)
                if row is not None:
                    pending.append(pool.submit(read,row))
    return X,y


def worker(experiment,directory,repetition):
    conf,folder=locations(experiment)
    config=json.loads(conf.read_text(encoding="utf8"))
    for relative,digest in config["codigo"]["fuentes"].items():
        if sha(ROOT/relative)!=digest:
            raise ValueError(f"Frozen source changed: {relative}")
    for name,value in {"OMP_NUM_THREADS":"6","MKL_NUM_THREADS":"6","OPENBLAS_NUM_THREADS":"6",
                       "TF_NUM_INTRAOP_THREADS":"6","TF_NUM_INTEROP_THREADS":"1",
                       "TF_DETERMINISTIC_OPS":"1","TF_ENABLE_ONEDNN_OPTS":"1",
                       "NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS":"0"}.items():
        os.environ[name]=value
    import numpy as np
    import psutil
    from threadpoolctl import threadpool_limits
    from asl_cuda.model import Network,cpu_reference,INITIAL,INITIAL_SHA,compiled_resources
    from asl_cuda.runtime import cuda
    if psutil.virtual_memory().available<4096*2**20:
        raise MemoryError("At least 4GiB available host RAM required")
    rows=[]
    epoch_rows=[]
    config_sha=sha(conf)

    def state(phase,**other):
        write_json(directory/"estado.json",dict(utc=utc(),fase=phase,**other))
        print(utc(),phase,other,flush=True)

    def add(phase,seconds,kind="pared",start="",end="",**load):
        row=dict.fromkeys(FIELDS,"")
        row.update(experimento_id=experiment,run_id=directory.name,dispositivo="gpu",repeticion=repetition,
            fase=phase,tipo_medicion=kind,duracion_s=float(seconds),inicio_utc=start,fin_utc=end,
            incluido_resumen="true",estado="completada",configuracion_sha256=config_sha,
            datos_sha256=config["datos"]["manifest_splits_sha256"],
            evidencia=(directory/"tiempos.csv").relative_to(ROOT).as_posix())
        row.update(load)
        rows.append(row)
        write_csv(directory/"tiempos.csv",rows,FIELDS)

    state("verificacion_fuentes_y_datos")
    manifest=load_manifest(config)
    data_root=Path(config["datos"]["ruta"])
    for split in manifest.values():
        for item in split:
            if sha(data_root/item["ruta_relativa"])!=item["sha256"]:
                raise ValueError("Dataset file changed: "+item["ruta_relativa"])
    state("inicializacion_y_warmup")
    init=time.perf_counter()
    tf,reference=cpu_reference()
    net=Network(reference,backend="cublas")
    write_json(directory/"hardware.json",dict(device=net.info,cpu=config["cpu"],
        tensorflow=tf.__version__,tf_build=tf.sysconfig.get_build_info(),
        tf_visible=[str(d) for d in tf.config.get_visible_devices()],
        cpu_tensor_device=tf.constant([1.]).device,cublas_version=net.blas.version,
        cublas_dll=str(net.blas.path),cublas_sha256=sha(net.blas.path),
        buffers_bytes=net.bytes_allocated,initial_checkpoint_sha256=sha(INITIAL)))
    if sha(INITIAL)!=INITIAL_SHA or tf.config.get_visible_devices("GPU"):
        raise ValueError("CPU mirror or initial state invalid")
    wx,wy=load_images(manifest["train"][:148],data_root,1)
    wv,wvy=load_images(manifest["val"][:128],data_root,1)
    wt,wty=load_images(manifest["test"][:32],data_root,1)
    with threadpool_limits(limits=6):
        net.train_step(wx[:128],wy[:128],0,0)
        net.train_step(wx[128:],wy[128:],0,1)
        reference.evaluate(wv,wvy,batch_size=128,verbose=0)
        reference.evaluate(wt,wty,batch_size=32,verbose=0)
        net.restore()
        # Verify real restoration, including every parameter and optimizer state.
        for p in net.params:
            if not np.array_equal(p["value"].copy_to_host(),p["initial"].ravel()):
                raise ValueError("Warmup parameter restore failed")
            if np.any(p["m"].copy_to_host()) or np.any(p["v"].copy_to_host()):
                raise ValueError("Warmup optimizer restore failed")
        if net.counter.copy_to_host()[0]!=0:
            raise ValueError("Warmup counter restore failed")
        for op in net.ops:
            if op["kind"]=="bn":
                if not np.array_equal(op["moving_mean"].copy_to_host(),op["initial_mean"]) or not np.array_equal(op["moving_var"].copy_to_host(),op["initial_var"]):
                    raise ValueError("Warmup BN restore failed")
        write_json(directory/"warmup.json",dict(duracion_s=time.perf_counter()-init,train=148,val=128,test=32,
                   restauracion_verificada=True,metricas_warmup_excluidas=True))

        state("preparacion_cpu",workers=6)
        pipeline_start_utc,pipeline_start=utc(),time.perf_counter_ns()
        prep_start=time.perf_counter_ns()
        inputs={s:load_images(manifest[s],data_root,6) for s in ("train","val","test")}
        prep_end=time.perf_counter_ns()
        add("preparacion_cpu",(prep_end-prep_start)/1e9,start=pipeline_start_utc,end=utc(),muestras=17400,
            motivo="CPU6workers; orden congelado; no atribuir mejora de esta fase a CUDA")
        x,y=inputs["train"]; vx,vy=inputs["val"]; tx,ty=inputs["test"]
        # Hash prepared bytes outside all TRAINING intervals. It is an explicit
        # nested audit overhead of the continuous pipeline, also recorded below.
        audit_start=time.perf_counter_ns()
        expected=json.loads((ROOT/"resultados/secuencial/EXP-002/RUN-20261117T171942Z-cpu-01/tensores.json").read_text(encoding="utf8"))
        tensor_hash={}
        for split,(images,labels) in inputs.items():
            fingerprint=hashlib.sha256(memoryview(images).cast("B")).hexdigest()
            label_hash=hashlib.sha256(memoryview(labels).cast("B")).hexdigest()
            if fingerprint!=expected[split]["images_sha256"] or label_hash!=expected[split]["labels_sha256"]:
                raise ValueError("CPU-prepared tensors differ from historical reference")
            tensor_hash[split]=dict(images_sha256=fingerprint,labels_sha256=label_hash,shape=list(images.shape),dtype=str(images.dtype))
        write_json(directory/"tensores.json",tensor_hash)
        audit_seconds=(time.perf_counter_ns()-audit_start)/1e9
        add("auditoria_tensores",audit_seconds,motivo="Anidado en pipeline; fuera de entrenamiento y suma comparable")
        state("entrenamiento_CNN-B",epoca=1,pasos=0)
        train_start_utc,train_start=utc(),time.perf_counter_ns()
        event_totals={}
        validations=0.
        export_wall=0.
        with net.timed("h2d_inicial"):
            net.restore()
        for phase,seconds in net.drain_events().items():
            event_totals[phase]=event_totals.get(phase,0.)+seconds
        rng=np.random.Generator(np.random.PCG64(43))
        shuffle_hashes=[]
        for epoch in range(15):
            epoch_start=time.perf_counter()
            net.reset_metrics()
            order=rng.permutation(len(x))
            # Hash tiny order outside device intervals, inside train wall like control/logging.
            shuffle_hashes.append(hashlib.sha256(order.astype(np.int64).tobytes()).hexdigest())
            for j,begin in enumerate(range(0,len(x),128)):
                indices=order[begin:begin+128]
                timings=net.train_step(x[indices],y[indices],epoch,j)
                for phase,seconds in timings.items():
                    event_totals[phase]=event_totals.get(phase,0.)+seconds
                if j%12==0:
                    state("entrenamiento_CNN-B",epoca=epoch+1,pasos=net.steps)
            export_start=time.perf_counter()
            times,total=net.export()
            export_wall+=time.perf_counter()-export_start
            for phase,seconds in times.items():
                event_totals[phase]=event_totals.get(phase,0.)+seconds
            if int(total[2])!=12180:
                raise ValueError("Epoch sample counter differs")
            state("validacion_cpu",epoca=epoch+1,pasos=net.steps)
            val_start=time.perf_counter()
            val=reference.evaluate(vx,vy,batch_size=128,verbose=0,return_dict=True)
            val_s=time.perf_counter()-val_start
            validations+=val_s
            epoch_rows.append(dict(epoca=epoch+1,pasos_acumulados=net.steps,train_loss=float(total[0]/total[2]),
                train_accuracy=float(total[1]/total[2]),val_loss=float(val["loss"]),val_accuracy=float(val["accuracy"]),
                pared_epoca_s=time.perf_counter()-epoch_start,validacion_cpu_s=val_s))
            write_csv(directory/"epocas.csv",epoch_rows)
            print("EPOCH",epoch+1,epoch_rows[-1],flush=True)
        # Last epoch already exported weights. Only optimizer state is needed here.
        # export also copies model again; this conservative real overhead is included
        # and its exact extra payload is disclosed instead of subtracting it.
        times,_=net.export(optimizer=True)
        for phase,seconds in times.items():
            event_totals[phase]=event_totals.get(phase,0.)+seconds
        train_end,train_end_utc=time.perf_counter_ns(),utc()
        add("entrenamiento_CNN-B",(train_end-train_start)/1e9,start=train_start_utc,end=train_end_utc,
            epocas=15,pasos=1440,muestras=182700,batch_size=128,motivo="Incluye H2D,D2H,CPUstaging y validacion CPU")
        add("validacion_anidada_CNN-B",validations,epocas=15,muestras=39150,
            motivo="Subintervalo entrenamiento: no sumar dos veces")
        for phase,seconds in event_totals.items():
            add(phase,seconds,kind="evento_device",epocas=15,pasos=1440,
                motivo="Intervalos stream anidados en entrenamiento; incluyen posibles huecos de envio Host")
        state("evaluacion_final_cpu",pasos=net.steps)
        eval_start_utc,eval_start=utc(),time.perf_counter_ns()
        test=reference.evaluate(tx,ty,batch_size=32,verbose=0,return_dict=True)
        eval_seconds=(time.perf_counter_ns()-eval_start)/1e9
        add("evaluacion_CNN-B",eval_seconds,start=eval_start_utc,end=utc(),muestras=2610,
            motivo="CPU tras D2H del modelo; no atribuir a inferencia CUDA")
        pipeline_end,pipeline_end_utc=time.perf_counter_ns(),utc()
        add("pipeline_total_cnn_b",(pipeline_end-pipeline_start)/1e9,start=pipeline_start_utc,end=pipeline_end_utc,
            epocas=15,pasos=1440,muestras=17400,batch_size=128,
            motivo="Continuo: preparacion+auditoria explicita+entrenamiento+evaluacion; no baseline total adaptativo CPU")
        if net.steps!=1440 or int(reference.optimizer.iterations.numpy())!=1440:
            raise ValueError("Incorrect training step count")
        metrics=dict(test={k:float(v) for k,v in test.items()},epocas=15,pasos=net.steps,
                     val_final=epoch_rows[-1],accuracy_historica_cpu_min=0.9287356,
                     comparabilidad="Mismos datos,modelo,pasos; RNG/ejecucion diferentes, no identidad numerica",)
        write_json(directory/"metricas.json",metrics)
        write_json(directory/"transferencias.json",dict(lotes=net.transfers[-1440:],
            h2d_estado_inicial_bytes=3*2245245*4+960*4+24,reinicios_metricas_h2d_bytes=15*12,
            modelo_d2h_copias=16,modelo_bytes_por_copia=2246205*4,
            optimizador_final_bytes=2*2245245*4+8,control_d2h_bytes=1440*4,
            aclaracion="Ultima copia extra de pesos por export final, incluida en pared y eventos"))
        write_json(directory/"ordenes.json",dict(seed=43,generador="PCG64",epocas_sha256=shuffle_hashes))
        state("guardado_fuera_de_timing",pasos=net.steps)
        out=ROOT/"data"/experiment/directory.name
        out.mkdir(parents=True,exist_ok=False)
        model_path=out/"CNN-B-CUDA.keras"
        reference.save(model_path)
        write_json(directory/"modelo-guardado.json",dict(ruta=model_path.relative_to(ROOT).as_posix(),
            sha256=sha(model_path),optimizer_iterations=1440,
            nota="Inferencia Keras CPU; reanudar entrenamiento CUDA requiere claves RNG y estado del adaptador"))
        geometries=[]
        for op in net.ops:
            if op["kind"]=="conv":
                geometries.append(dict(operacion="im2col",block=[256,1,1],
                    grid=[min(4096,(128*op["H"]*op["W"]*9*op["Ci"]+255)//256),1,1],
                    forma=[128,op["H"],op["W"],op["Ci"],op["Co"]]))
        write_json(directory/"lanzamientos.json",dict(propio=geometries,vector_block=256,
            reduccion_block=[32,8,1],reduccion_final_block=256,softmax_block=32,
            gemm="cuBLAS SGEMM: jerarquia interna gestionada por NVIDIA, no inventada por el equipo",
            recursos=compiled_resources()))
        net.stream.synchronize()
        net.blas.close()
        state("completada",pasos=1440)


def monitor(process,directory,stop):
    import psutil
    root=psutil.Process(process.pid)
    previous={}
    fields=["utc","fase","epoca","cpu_procesos_pct","rss_bytes","cpu_sistema_pct","ram_disponible_bytes",
            "gpu_util_pct","gpu_mem_util_pct","gpu_mem_used_mib","gpu_temperatura_c","gpu_power_w","gpu_sm_mhz"]
    core_fields=[f"cpu_logico_{i:02d}_pct" for i in range(psutil.cpu_count(logical=True))]
    fields+=core_fields
    psutil.cpu_percent(percpu=True)
    with (directory/"recursos.csv").open("w",encoding="utf8",newline="") as f, \
         (directory/"monitor-gpu.log").open("w",encoding="utf8") as raw:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        while not stop.is_set():
            start=time.perf_counter()
            cpu,rss=0.,0
            try:
                members=[root,*root.children(recursive=True)]
                for p in members:
                    try:
                        c=p.cpu_times(); cumulative=c.user+c.system
                        if p.pid in previous:
                            old,stamp=previous[p.pid]
                            cpu+=(cumulative-old)/max(start-stamp,1e-6)*100
                        previous[p.pid]=(cumulative,start)
                        rss+=p.memory_info().rss
                    except (psutil.NoSuchProcess,psutil.AccessDenied):
                        pass
            except psutil.NoSuchProcess:
                pass
            try:
                state=json.loads((directory/"estado.json").read_text(encoding="utf8"))
            except (FileNotFoundError,json.JSONDecodeError):
                state={}
            gpu=subprocess.run(["nvidia-smi","--query-gpu=utilization.gpu,utilization.memory,memory.used,temperature.gpu,power.draw,clocks.sm",
                                "--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=10)
            raw.write(utc()+" "+gpu.stdout.strip()+" "+gpu.stderr.strip()+"\n");raw.flush()
            values=gpu.stdout.strip().split(",") if gpu.returncode==0 else []
            cores=psutil.cpu_percent(percpu=True)
            row=dict(utc=utc(),fase=state.get("fase","inicio"),epoca=state.get("epoca",""),cpu_procesos_pct=cpu,
                     rss_bytes=rss,cpu_sistema_pct=sum(cores)/len(cores),ram_disponible_bytes=psutil.virtual_memory().available)
            row.update(zip(core_fields,cores))
            for name,value in zip(fields[7:13],values):
                try: row[name]=float(value.strip())
                except ValueError: row[name]=""
            writer.writerow(row);f.flush()
            stop.wait(max(0.,1-(time.perf_counter()-start)))


def run_series(experiment="EXP-003"):
    import threading
    from scipy import stats
    import numpy as np
    conf,folder=locations(experiment)
    config=json.loads(conf.read_text(encoding="utf8"))
    series_path=folder/"serie.json"
    if series_path.exists():
        old=json.loads(series_path.read_text(encoding="utf8"))
        if old.get("estado")=="completada":
            print("Serie ya completada: se conservan las observaciones existentes",flush=True)
            return old
        raise FileExistsError("An incomplete series exists: investigate, never overwrite")
    series=dict(estado="en_curso",inicio=utc(),configuracion_sha256=sha(conf),corridas=[])
    write_json(series_path,series)
    target=5
    for repetition in range(1,10):
        run=f"RUN-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-gpu-{repetition:02d}"
        directory=folder/run
        directory.mkdir(exist_ok=False)
        command=[sys.executable,"-u","-X","utf8",str(Path(__file__).resolve()),"worker",
                 "--experimento",experiment,"--run",run,"--repeticion",str(repetition)]
        write_json(directory/"invocacion.json",dict(comando=command,cwd=str(ROOT),inicio=utc()))
        (directory/"configuracion.json").write_bytes(conf.read_bytes())
        # Before running, snapshot exact code/context in a NEW directory.
        subprocess.run([sys.executable,"-X","utf8",str(ROOT/"scripts/capturar_contexto.py"),
                        "--destino",str(directory/"contexto")],cwd=ROOT,check=True,
                        stdout=(directory/"contexto-captura.log").open("w",encoding="utf8"))
        print("INICIO",run,flush=True)
        with (directory/"stdout.log").open("w",encoding="utf8") as out, (directory/"stderr.log").open("w",encoding="utf8") as err:
            process=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err)
            stop=threading.Event()
            thread=threading.Thread(target=monitor,args=(process,directory,stop),daemon=True)
            thread.start()
            exitcode=process.wait()
            stop.set();thread.join(timeout=15)
        write_json(directory/"salida.json",dict(codigo=exitcode,fin=utc()))
        series["corridas"].append(dict(run_id=run,codigo=exitcode))
        if exitcode:
            series.update(estado="fallida",fin=utc())
            write_json(series_path,series)
            raise RuntimeError("GPU run failed; see preserved logs: "+run)
        # Seal this RUN only after monitoring/process completion. Never overwrite it.
        artifacts=[]
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                artifacts.append(dict(ruta=path.relative_to(directory).as_posix(),bytes=path.stat().st_size,sha256=sha(path)))
        write_csv(directory/"artefactos.csv",artifacts)
        print("FIN",run,flush=True)
        write_json(series_path,series)
        if repetition==target:
            observed=[]
            for item in series["corridas"]:
                with (folder/item["run_id"]/"tiempos.csv").open(encoding="utf8",newline="") as f:
                    observed.append(float(next(r for r in csv.DictReader(f) if r["fase"]=="entrenamiento_CNN-B")["duracion_s"]))
            n=len(observed)
            relative=float(stats.t.ppf(.975,n-1)*np.std(observed,ddof=1)/np.sqrt(n)/np.mean(observed)*100)
            series["precision_ic95_semiancho_pct"]=relative
            if relative<=5 or target==9:
                break
            target+=2
    series.update(estado="completada",fin=utc())
    write_json(series_path,series)
    return series


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("accion",choices=["prepare","run","worker"])
    parser.add_argument("--experimento",default="EXP-003")
    parser.add_argument("--run")
    parser.add_argument("--repeticion",type=int)
    args=parser.parse_args()
    if args.accion=="prepare":
        prepare(args.experimento)
    elif args.accion=="run":
        run_series(args.experimento)
    else:
        _,folder=locations(args.experimento)
        directory=folder/args.run
        try:
            worker(args.experimento,directory,args.repeticion)
        except BaseException:
            (directory/"error.txt").write_text(traceback.format_exc(),encoding="utf8")
            raise


if __name__=="__main__":
    main()
