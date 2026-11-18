"""Audit sealed GPU runs, reuse CPU rows, produce reproducible tables/plots/PDF."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import textwrap
import zipfile
import numpy as np
from scipy import stats

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT/"scripts"))
from benchmark_cpu import sha,write_csv,write_json,FIELDS


def read_csv(path):
    with Path(path).open(encoding="utf8",newline="") as f:
        return list(csv.DictReader(f))


def describe(values):
    a=np.asarray(values,np.float64)
    n=len(a);mean=float(a.mean())
    sd=float(a.std(ddof=1)) if n>1 else None
    half=float(stats.t.ppf(.975,n-1)*sd/np.sqrt(n)) if n>1 else None
    return dict(n=n,media_s=mean,mediana_s=float(np.median(a)),desviacion_muestral_s=sd,
                min_s=float(a.min()),max_s=float(a.max()),cv_pct=sd/mean*100 if sd is not None else None,
                ic95_inferior_s=mean-half if half is not None else None,
                ic95_superior_s=mean+half if half is not None else None)


def readable(text):
    for old,new in {"15épocas":"15 épocas","1440actualizaciones":"1440 actualizaciones",
        "1440pasos":"1440 pasos","17400 imágenes":"17,400 imágenes","batch128":"batch 128",
        "batch32":"batch 32","4GiB":"4 GiB","GTX1650":"GTX 1650","6workers":"6 workers",
        "1worker":"1 worker","51pruebas":"51 pruebas","78adjuntos":"78 adjuntos",
        "10pasos":"10 pasos","16copias":"16 copias","RGB64×64":"RGB 64×64",
        "toolkit13.1":"toolkit 13.1","driver591.59":"driver 591.59"}.items():
        text=text.replace(old,new)
    return text


def generar(experiment="EXP-003"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import h5py
    folder=ROOT/"resultados/paralelo"/experiment
    if (folder/"artefactos-experimento.csv").exists():
        print("Informe ya sellado: se reutiliza sin sobrescribir",flush=True)
        return folder/"resultados.md"
    series=json.loads((folder/"serie.json").read_text(encoding="utf8"))
    if series["estado"]!="completada":
        raise ValueError("Only a completed series can be summarized")
    config_path=ROOT/"configuraciones"/(experiment+".json")
    config=json.loads(config_path.read_text(encoding="utf8"))
    evidence_folder=ROOT/"resultados/evidencias"/experiment
    selection_path=evidence_folder/"seleccion.json"
    captures=[]
    if selection_path.exists():
        selection=json.loads(selection_path.read_text(encoding="utf8"))
        for device,name in selection["capturas"].items():
            if sha(evidence_folder/name)!=selection["sha256"][device]:
                raise ValueError("Selected screenshot changed")
            captures.append((name,"Preparación CPU real" if device=="cpu" else "Entrenamiento GPU real"))
    if sha(config_path)!=series["configuracion_sha256"]:
        raise ValueError("Frozen GPU configuration changed")
    for rel,digest in config["codigo"]["fuentes"].items():
        if sha(ROOT/rel)!=digest:raise ValueError("Frozen source changed: "+rel)
    cpu_folder=ROOT/"resultados/secuencial/EXP-002"
    cpu_series=json.loads((cpu_folder/"serie.json").read_text(encoding="utf8"))
    cpu_config=json.loads((ROOT/"configuraciones/EXP-002.json").read_text(encoding="utf8"))
    transitive={"LAB3-DATASCIENCE/Lab3_ASL_Deep_Learning.ipynb":cpu_config["codigo"]["notebook_sha256"],
                "scripts/benchmark_cpu.py":cpu_config["codigo"]["benchmark_sha256"]}
    for relative,digest in transitive.items():
        if sha(ROOT/relative)!=digest:raise ValueError("Original/transitive source changed: "+relative)
    # CPU series stores run IDs differently from the independent GPU controller.
    cpu_dirs=sorted(p.parent for p in cpu_folder.glob("RUN-*/tiempos.csv")
                    if (p.parent/"salida.json").is_file())
    cpu_dirs=[p for p in cpu_dirs if json.loads((p/"salida.json").read_text(encoding="utf8")).get("codigo",0)==0]
    if len(cpu_dirs)!=5:raise ValueError("Expected exactly five original measured CPU runs")
    gpu_dirs=[folder/item["run_id"] for item in series["corridas"] if item["codigo"]==0]
    tensor_reference=json.loads((cpu_dirs[0]/"tensores.json").read_text(encoding="utf8"))
    records=[];raw_gpu=[];epoch_data=[];gpu_metrics=[];checkpoint_checks=[]
    phase_stats=[]
    for device,dirs in (("cpu",cpu_dirs),("gpu",gpu_dirs)):
        for directory in dirs:
            if device=="gpu":
                for row in read_csv(directory/"artefactos.csv"):
                    path=directory/row["ruta"]
                    if path.stat().st_size!=int(row["bytes"]) or sha(path)!=row["sha256"]:
                        raise ValueError("Sealed run artifact changed: "+str(path))
                source_capture={r["ruta_relativa"]:r["sha256"] for r in read_csv(directory/"contexto/fuentes.csv")}
                for relative,digest in transitive.items():
                    if source_capture.get(relative)!=digest:raise ValueError("Transitive source differs in RUN context")
                tensors=json.loads((directory/"tensores.json").read_text(encoding="utf8"))
                for split in ("train","val","test"):
                    for key in ("images_sha256","labels_sha256","shape","dtype"):
                        if tensors[split][key]!=tensor_reference[split][key]:raise ValueError("CPU/GPU tensors differ")
            rows=read_csv(directory/"tiempos.csv")
            phases={r["fase"]:float(r["duracion_s"]) for r in rows}
            training=next(r for r in rows if r["fase"]=="entrenamiento_CNN-B")
            if any(int(training[key])!=value for key,value in dict(epocas=15,pasos=1440,muestras=182700,batch_size=128).items()):
                raise ValueError("Historical/new workload differs")
            item=dict(dispositivo=device,run_id=directory.name,
                preparacion_cpu_s=phases["preparacion_cpu"],entrenamiento_s=phases["entrenamiento_CNN-B"],
                evaluacion_cpu_s=phases["evaluacion_CNN-B"],
                validacion_anidada_s=phases["validacion_anidada_CNN-B"],
                suma_comparable_s=sum(phases[p] for p in ("preparacion_cpu","entrenamiento_CNN-B","evaluacion_CNN-B")),
                total_continuo_gpu_s=phases.get("pipeline_total_cnn_b",""))
            if device=="gpu":
                raw_gpu+=rows
                item["h2d_eventos_s"]=sum(phases[p] for p in ("h2d_inicial","h2d_lotes"))
                item["d2h_eventos_s"]=sum(phases[p] for p in ("d2h_control","d2h_modelo","d2h_optimizador"))
                item["computo_intervalos_stream_s"]=sum(phases[p] for p in ("forward_loss","backward","adam_metricas"))
                item["administracion_y_staging_s"]=item["entrenamiento_s"]-item["validacion_anidada_s"]-item["h2d_eventos_s"]-item["d2h_eventos_s"]-item["computo_intervalos_stream_s"]
                if item["administracion_y_staging_s"]<0:raise ValueError("Overlapping/invalid interval accounting")
                metric=json.loads((directory/"metricas.json").read_text(encoding="utf8"))
                gpu_metrics.append(dict(run_id=directory.name,accuracy=metric["test"]["accuracy"],loss=metric["test"]["loss"],
                    val_accuracy=metric["val_final"]["val_accuracy"],pasos=metric["pasos"]))
                epochs=read_csv(directory/"epocas.csv")
                if len(epochs)!=15 or int(epochs[-1]["pasos_acumulados"])!=1440:raise ValueError("Incorrect epochs/steps")
                epoch_data.append(epochs)
                saved=json.loads((directory/"modelo-guardado.json").read_text(encoding="utf8"))
                model=ROOT/saved["ruta"]
                if sha(model)!=saved["sha256"]:raise ValueError("Model checksum differs")
                with zipfile.ZipFile(model) as z, h5py.File(io.BytesIO(z.read("model.weights.h5")),"r") as h:
                    if int(h["optimizer/vars/0"][()])!=1440:raise ValueError("Saved optimizer counter differs")
                    arrays=[]
                    h.visititems(lambda name,obj: arrays.append((name,obj[()])) if isinstance(obj,h5py.Dataset) else None)
                    digest=hashlib.sha256()
                    for name,a in sorted(arrays):
                        digest.update(name.encode());digest.update(str((a.shape,str(a.dtype))).encode());digest.update(a.tobytes())
                    checkpoint_checks.append(dict(run_id=directory.name,keras_sha256=saved["sha256"],
                        contenido_canonico_sha256=digest.hexdigest(),optimizer_iterations=1440,arrays=len(arrays)))
            records.append(item)
    fields=list(dict.fromkeys(key for row in records for key in row))
    write_csv(folder/"comparacion_cnn_b.csv",records,fields)
    write_csv(folder/"tiempos_medidos.csv",raw_gpu,FIELDS)
    classified=[]
    for r in raw_gpu:
        phase=r["fase"]
        executor="cpu+gpu"
        if phase in ("preparacion_cpu","auditoria_tensores","validacion_anidada_CNN-B","evaluacion_CNN-B"):
            executor="cpu"
        elif r["tipo_medicion"]=="evento_device":
            executor="transferencia_cpu_gpu" if phase.startswith(("h2d","d2h")) else "gpu"
        classified.append(dict(r,variante="gpu",ejecutor_efectivo=executor))
    write_csv(folder/"tiempos_clasificados.csv",classified,[*FIELDS,"variante","ejecutor_efectivo"])
    write_csv(folder/"calidad.csv",gpu_metrics)
    cpu_quality=[]
    for directory in cpu_dirs:
        m=json.loads((directory/"metricas.json").read_text(encoding="utf8"))["modelos"]["CNN-B"]
        cpu_quality.append(dict(dispositivo="cpu",run_id=directory.name,accuracy=m["accuracy"],loss=m["loss"]))
    compared_quality=cpu_quality+[dict(dispositivo="gpu",run_id=r["run_id"],accuracy=r["accuracy"],loss=r["loss"]) for r in gpu_metrics]
    write_csv(folder/"calidad_comparada.csv",compared_quality)
    write_json(folder/"fuentes_transitivas_verificadas.json",dict(estado="aprobado",sha256=transitive,
        comprobacion="Mismos archivos importados en captura de cada RUN y referencia EXP-002; originales intactos"))
    compatibility=[]
    initial_reference=[json.loads((p/"estados-iniciales.json").read_text(encoding="utf8"))["CNN-B"]["state_sha256"] for p in cpu_dirs]
    if len(set(initial_reference))!=1:raise ValueError("CPU initial states differ")
    initial=ROOT/"data/EXP-002/RUN-20261117T171942Z-cpu-01/CNN-B.estado-inicial.npz"
    digest=hashlib.sha256()
    with np.load(initial,allow_pickle=False) as archive:
        for i in range(len(archive.files)):
            a=archive[f"arr_{i}"];digest.update(str((a.shape,str(a.dtype))).encode());digest.update(a.tobytes())
    if digest.hexdigest()!=initial_reference[0]:raise ValueError("Imported GPU initial state differs")
    for directory in gpu_dirs:
        criteria=[("datos_y_particiones_sha256",config["datos"]["manifest_splits_sha256"],config["datos"]["manifest_splits_sha256"],"si","X/y coinciden byte a byte en las tres particiones"),
            ("estado_inicial_canonico",initial_reference[0],digest.hexdigest(),"si","Todas las variables iniciales y Adam importados del NPZ preservado"),
            ("modelo_carga_precision","CNN-B;15epocas;1440pasos;batch128/20;FP32","CNN-B;15epocas;1440pasos;batch128/20;FP32","si","Arquitectura, perdidas y Adam originales; operadores validados"),
            ("dropout_distribucion","0.25/0.25/0.30/0.40 por elemento","0.25/0.25/0.30/0.40 por elemento","si","Mismo volumen matematico de entrenamiento"),
            ("identidad_rng_orden_mascaras","TensorFlow seed43","PCG64 seed43 y claves por capa/paso","no","Identidad numerica no requerida para timing; limite explicito de calidad/trajectoria"),
            ("workers_preparacion",1,6,"no","Cambio CPU fuera de entrenamiento; no atribuir su mejora a CUDA"),
            ("entrenamiento_con_validacion","15pasadas valCPU,batch128","15pasadas valCPU,batch128,H2D y D2H incluidos","si","Se compara pared total, no kernel aislado contra fit completo"),
            ("intercalado_temperatura","Referencia historica","GPU posterior, mismo plan ASUS Recommended","no","No se afirma experimento alternado ni identidad termica; cinco observaciones por variante")]
        for criterion,c,g,equal,why in criteria:
            compatibility.append(dict(experimento_id=experiment,run_cpu=";".join(p.name for p in cpu_dirs),run_gpu=directory.name,
                criterio=criterion,valor_cpu=c,valor_gpu=g,equivalente=equal,justificacion=why))
    compat_index=ROOT/"resultados/comparacion/comparabilidad.csv"
    cf=read_csv_header(compat_index)
    write_csv(folder/"comparabilidad.csv",compatibility,cf)
    write_csv(compat_index,[r for r in read_csv(compat_index) if r["experimento_id"]!=experiment]+compatibility,cf)
    write_json(folder/"validacion_checkpoints.json",dict(estado="aprobado",modelos=checkpoint_checks,
        identicos_entre_corridas=len(set(r["contenido_canonico_sha256"] for r in checkpoint_checks))==1))
    cpu=[r for r in records if r["dispositivo"]=="cpu"]
    gpu=[r for r in records if r["dispositivo"]=="gpu"]
    for device,items in (("cpu",cpu),("gpu",gpu)):
        for phase in ("preparacion_cpu_s","entrenamiento_s","evaluacion_cpu_s","suma_comparable_s","validacion_anidada_s"):
            phase_stats.append(dict(dispositivo=device,fase=phase,**describe([r[phase] for r in items])))
    for phase in ("h2d_eventos_s","d2h_eventos_s","computo_intervalos_stream_s","administracion_y_staging_s","total_continuo_gpu_s"):
        phase_stats.append(dict(dispositivo="gpu",fase=phase,**describe([r[phase] for r in gpu])))
    write_csv(folder/"resumen.csv",phase_stats)
    lookup={(r["dispositivo"],r["fase"]):r for r in phase_stats}
    ratios=[]
    for phase in ("preparacion_cpu_s","entrenamiento_s","evaluacion_cpu_s","suma_comparable_s"):
        c,g=lookup["cpu",phase],lookup["gpu",phase]
        ratios.append(dict(fase=phase,n_cpu=c["n"],n_gpu=g["n"],media_cpu_s=c["media_s"],media_gpu_s=g["media_s"],
                          speedup=c["media_s"]/g["media_s"],
                          interpretacion="CPU1vs6workers, no CUDA" if phase=="preparacion_cpu_s" else (
                              "Ambas evaluaciones CPU" if phase=="evaluacion_cpu_s" else "Mismo alcance; incluye transferencias y validacion")))
    write_csv(folder/"speedup.csv",ratios)
    # Update mutable global indexes without touching any sealed RUN.
    index=ROOT/"resultados/paralelo/tiempos.csv"
    prior=[r for r in read_csv(index) if r["experimento_id"]!=experiment]
    write_csv(index,prior+raw_gpu,FIELDS)
    speed_index=ROOT/"resultados/comparacion/speedup.csv"
    sf=list(read_csv_header(speed_index))
    existing=[r for r in read_csv(speed_index) if r["experimento_id"]!=experiment]
    for r in ratios:
        existing.append(dict(experimento_id=experiment,fase=r["fase"],n_cpu=r["n_cpu"],n_gpu=r["n_gpu"],
            media_cpu_s=r["media_cpu_s"],media_gpu_s=r["media_gpu_s"],speedup=r["speedup"],
            alcance_transferencias=r["interpretacion"],fuente=(folder/"speedup.csv").relative_to(ROOT).as_posix()))
    write_csv(speed_index,existing,sf)
    summary_index=ROOT/"resultados/comparacion/resumen.csv"
    rf=list(read_csv_header(summary_index))
    existing=[r for r in read_csv(summary_index) if r["experimento_id"]!=experiment]
    for r in phase_stats:
        existing.append(dict(experimento_id=experiment,dispositivo=r["dispositivo"],fase=r["fase"],tipo_medicion="pared" if r["fase"] not in ("h2d_eventos_s","d2h_eventos_s","computo_intervalos_stream_s") else "evento_device",
            n=r["n"],media_s=r["media_s"],mediana_s=r["mediana_s"],desviacion_muestral_s=r["desviacion_muestral_s"],min_s=r["min_s"],max_s=r["max_s"],
            runs_incluidos=";".join(x["run_id"] for x in (cpu if r["dispositivo"]=="cpu" else gpu)),
            fuente=(folder/"resumen.csv").relative_to(ROOT).as_posix()))
    write_csv(summary_index,existing,rf)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    figures=[]
    chart_dir=folder/"graficas";chart_dir.mkdir(exist_ok=True)
    def save(fig,name):
        fig.tight_layout()
        fig.savefig(chart_dir/name,dpi=170,bbox_inches="tight")
        figures.append(fig)
    fig,ax=plt.subplots(figsize=(9,4.5))
    for items,color,label,offset in ((cpu,"#2367a2","CPU histórica (6 hilos)",-.08),(gpu,"#d27700","CUDA + transferencias + validación CPU",.08)):
        values=[r["entrenamiento_s"] for r in items]
        ax.plot(np.arange(1,len(items)+1)+offset,values,"o-",color=color,label=label)
        ax.axhline(np.mean(values),color=color,linestyle="--",alpha=.6)
    ax.set(xlabel="Repetición",ylabel="Tiempo de pared de entrenamiento (s)",ylim=(0,None),title="CNN-B · 15 épocas · batch128 · 1440 actualizaciones")
    ax.legend(fontsize=9);ax.grid(axis="y",alpha=.2)
    save(fig,"01_entrenamiento_repeticiones.png")
    fig,ax=plt.subplots(figsize=(9,4.8))
    bottom=np.zeros(len(gpu))
    for key,label,color in [("computo_intervalos_stream_s","Cómputo: intervalos del stream","#2367a2"),
            ("h2d_eventos_s","H2D","#53a78f"),("d2h_eventos_s","D2H","#8f79b7"),
            ("validacion_anidada_s","Validación CPU","#d27700"),("administracion_y_staging_s","Host, staging y administración","#999999")]:
        value=np.array([r[key] for r in gpu]);ax.bar(np.arange(1,len(gpu)+1),value,bottom=bottom,label=label,color=color);bottom+=value
    ax.set(xlabel="Repetición GPU",ylabel="Tiempo (s)",title="Desglose del entrenamiento total · intervalos sin solapamiento")
    ax.legend(loc="upper left",bbox_to_anchor=(1.01,1),fontsize=8)
    save(fig,"02_desglose_gpu.png")
    fig,axes=plt.subplots(1,2,figsize=(10,4.1))
    for key,label,color in [("train_accuracy","Entrenamiento","#2367a2"),("val_accuracy","Validación CPU","#d27700")]:
        values=np.array([[float(r[key]) for r in ep] for ep in epoch_data])
        for value in values:axes[0].plot(range(1,16),value,color=color,alpha=.2,linewidth=.8)
        axes[0].plot(range(1,16),values.mean(axis=0),color=color,label=label)
    for key,label,color in [("train_loss","Entrenamiento","#2367a2"),("val_loss","Validación CPU","#d27700")]:
        values=np.array([[float(r[key]) for r in ep] for ep in epoch_data])
        for value in values:axes[1].plot(range(1,16),value,color=color,alpha=.2,linewidth=.8)
        axes[1].plot(range(1,16),values.mean(axis=0),color=color,label=label)
    axes[0].set(xlabel="Época",ylabel="Accuracy",ylim=(0,1.02),title="Calidad por época")
    axes[1].set(xlabel="Época",ylabel="Pérdida crossentropy",title="Convergencia")
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.15)
    save(fig,"03_curvas_aprendizaje.png")
    resources=read_csv(gpu_dirs[0]/"recursos.csv")
    times=np.array([(datetime.fromisoformat(r["utc"])-datetime.fromisoformat(resources[0]["utc"])).total_seconds() for r in resources])
    def values(key):return np.array([float(r[key]) if r.get(key) else np.nan for r in resources])
    fig,axes=plt.subplots(3,1,figsize=(10,6.4),sharex=True)
    axes[0].plot(times,values("gpu_util_pct"),color="#d27700");axes[0].set(ylabel="GPU (%)",ylim=(0,105))
    axes[1].plot(times,values("cpu_procesos_pct"),color="#2367a2");axes[1].set(ylabel="CPU proceso (%)",ylim=(0,None))
    axes[2].plot(times,values("gpu_mem_used_mib"),color="#7161a6");axes[2].set(ylabel="VRAM (MiB)",xlabel="Tiempo desde primera muestra (s)",ylim=(0,4096))
    axes[0].set_title("Recursos reales, primera corrida · psutil / nvidia-smi · muestreo ≈1s")
    for ax in axes:ax.grid(alpha=.15)
    save(fig,"04_recursos_gpu_cpu.png")
    train_cpu=lookup["cpu","entrenamiento_s"];train_gpu=lookup["gpu","entrenamiento_s"]
    speed=next(r["speedup"] for r in ratios if r["fase"]=="entrenamiento_s")
    metric_values=[r["accuracy"] for r in gpu_metrics]
    resource_summary=[]
    for directory in gpu_dirs:
        rr=read_csv(directory/"recursos.csv")
        selected=[r for r in rr if r["fase"]=="entrenamiento_CNN-B"]
        nums=lambda key:[float(r[key]) for r in selected if r.get(key)]
        resource_summary.append(dict(run_id=directory.name,muestras_entrenamiento=len(selected),
            gpu_util_media_pct=float(np.mean(nums("gpu_util_pct"))),gpu_util_max_pct=max(nums("gpu_util_pct")),
            vram_max_mib=max(nums("gpu_mem_used_mib")),temperatura_max_c=max(nums("gpu_temperatura_c"))))
    write_csv(folder/"recursos_resumen.csv",resource_summary)
    lines=["# Paso 5. Resultados reales CUDA Python — CNN-B", "",
        f"Estado: **{len(gpu)} corridas completas**, 15 épocas/1440 actualizaciones cada una. EXP: {experiment}.",
        f"Referencia CPU: cinco corridas históricas EXP-002. GPU: media **{train_gpu['media_s']:.3f} s**, DE {train_gpu['desviacion_muestral_s']:.3f} s; CPU: **{train_cpu['media_s']:.3f} s**.",
        f"Aceleración del entrenamiento completo: **{speed:.3f}×** (cociente de medias). Incluye preparación de lotes/máscaras, H2D, D2H y validación CPU.",
        f"Accuracy test GPU: {min(metric_values)*100:.3f}–{max(metric_values)*100:.3f}%. No se atribuye una mejora de calidad al hardware.","",
        f"Accuracy CPU histórica CNN-B: {min(r['accuracy'] for r in cpu_quality)*100:.3f}–{max(r['accuracy'] for r in cpu_quality)*100:.3f}%; valores individuales en calidad_comparada.csv.","",
        "## Condiciones, datos y procedimiento", "",
        "Mismas 17400 imágenes, manifiestos/splits 12180/2610/2610, RGB64×64 float32 /255, pesos iniciales y Adam. Se verifican hashes fuente y de tensores preparados contra CPU. CNN-B original, cinco Conv, cuatro BN/Dropout, batch128, último20,15épocas sin EarlyStopping.",
        "PCUDA existente; TensorFlow CPU conserva versiones y6hilos intra-op/1inter-op, BLAS6. No se reinstalaron NumPy/TensorFlow. GPU GTX1650,4GiB,toolkit13.1,driver591.59. Kernels Numba-CUDA Python y cuBLAS SGEMM pedantic FP32, sin TF32/mixed precision; un stream sin solapamiento.",
        "Cada RUN es un proceso nuevo. Warmup128+20train,128val,32test, fuera de timing; restauración de pesos, Adam, BN y contador verificada. Se prepara en CPU con6workers acotados y orden fijo; GPU ejecuta exclusivamente matemática del entrenamiento. Validación CPU2610imágenes cada época, batch128, y test CPU2610,batch32.",
        "Temporización principal perf_counter_ns sincronizada; eventos CUDA por grupos. Los eventos son intervalos del stream y pueden incluir huecos Host; no equivalen a tiempo activo exclusivo de kernels. Monitoreo1s incluido, sin sustracción. Misma política de caché del SO sin vaciar; no se mide disco frío.",
        "En los CSV crudos dispositivo=gpu identifica la variante del RUN, no que toda fase se ejecute en GPU. tiempos_clasificados.csv añade ejecutor_efectivo: preparación/validación/test son CPU; entrenamiento total es heterogéneo; forward/backward/Adam son GPU; copias son CPU↔GPU.",
        "Cinco repeticiones iniciales; extensión7/9 solo si semiancho ICt95% de entrenamiento supera5%, según configuración previa. No se excluyen tiempos por ser lentos.","",
        "## Tiempos individuales", "", "| RUN GPU | Preparación CPU (s) | Entrenamiento total (s) | Test CPU (s) | Accuracy test |", "|---|---:|---:|---:|---:|"]
    for r,m in zip(gpu,gpu_metrics):lines.append(f"| {r['run_id']} | {r['preparacion_cpu_s']:.3f} | {r['entrenamiento_s']:.3f} | {r['evaluacion_cpu_s']:.3f} | {m['accuracy']*100:.3f}% |")
    lines += ["", "## Comparación por fase", "", "| Fase | CPU media (s) | Variante CUDA media (s) | Cociente CPU/variante | Interpretación |", "|---|---:|---:|---:|---|"]
    for r in ratios:lines.append(f"| {r['fase']} | {r['media_cpu_s']:.3f} | {r['media_gpu_s']:.3f} | {r['speedup']:.3f}× | {r['interpretacion']} |")
    lines += ["", "La suma comparable usa únicamente preparación + entrenamiento CNN-B + evaluación CNN-B. Es un cálculo de tres fases, también para CPU; **no es todo el notebook original** ni el total adaptativo1858.617s. El pipeline GPU continuo incluye además auditoría explícita de tensores/administración; se publica por separado y no se divide por un total CPU de otro alcance.","",
        "## Variabilidad y descomposición", "",
        f"Entrenamiento GPU: n={train_gpu['n']}, mediana{train_gpu['mediana_s']:.3f}s, rango[{train_gpu['min_s']:.3f},{train_gpu['max_s']:.3f}]s, CV{train_gpu['cv_pct']:.3f}%, ICt95% nominal[{train_gpu['ic95_inferior_s']:.3f},{train_gpu['ic95_superior_s']:.3f}]s. El intervalo supone observaciones aproximadamente independientes y no corrige deriva térmica.",
        "", "| Componente GPU | Media (s) | Alcance |", "|---|---:|---|"]
    for key,label in [("computo_intervalos_stream_s","Forward/loss/backward/Adam/métricas: eventos"),("h2d_eventos_s","H2D: estado inicial,imágenes,labels,máscaras"),("d2h_eventos_s","D2H: control,pesos,BN,Adam"),("validacion_anidada_s","Validación CPU, anidada"),("administracion_y_staging_s","Residuo pared: staging,máscaras,lanzamientos,control,carga espejo")]:
        lines.append(f"| {key} | {lookup['gpu',key]['media_s']:.3f} | {label} |")
    lines += ["", "Los componentes son parte del entrenamiento; no se suman de nuevo al total. La exportación final conservadora copia el modelo una vez adicional:16copias de8984820bytes, incluidas en tiempo/payload. Por corrida entran19505052000bytes de imágenes+labels+máscaras, más estado inicial26946804bytes y reinicios de métricas; salen pesos/BN, controles y Adam final. Payload no equivale a tráfico físico PCIe medido.","",
        "## Corrección, calidad y cambios frente al diseño", "",
        "51pruebas de operadores y78adjuntos con activaciones/dY idénticos aprobaron tolerancias; diez pasos sintéticos verificaron contador,exportación,CPUeval sin mutación y guardado/recarga exacta. Los diagnósticos globales estrictos fallidos se conservan: redondeo FP32 cambia algunas rutas MaxPooling cerca de empates (DEC-023); no se etiquetan como aprobados ni se promete trayectoria CPU/GPU idéntica.",
        "Se mantiene el mismo Dropout/distribución y raíz43; el generador y el shuffle de NumPy no reproducen máscaras/orden internos históricos de TensorFlow. Esto conserva el volumen de cálculo; limita comparación de calidad y no constituye igualdad numérica. Se reportan resultados, no se ajustan épocas o semillas después de observar accuracy.",
        "Cambio justificado DEC-022: kernels tiled explícitos fueron correctos localmente pero el ensayo≈0.60s/lote no mejoraba CPU; im2col/SGEMM≈0.26s/lote en admisión. La serie usa este último. Workspace576MiB, tope1408MiB; las jerarquías propias son256threads, reducción32×8,softmax32. Los grids internos cuBLAS los gestiona NVIDIA y no se atribuyen al equipo.",
        "Los checkpoints finales se auditan por SHA-256, contenido canónico y contador Adam1440 en HDF5. Calidad real por RUN en calidad.csv y curvas; no se selecciona solo la mejor corrida. Los binarios grandes quedan en data/,fuera de Git,referenciados por manifiestos.","",
        "## Evidencia, límites y reproducción", "",
        "La referencia CPU es histórica y no alternada con GPU: misma máquina, datos y plan ASUS Recommended, pero diferente instante/estado térmico. No se repitió la campaña CPU por indicación del usuario. Preparación cambia1→6workers; su cociente es mejora CPU, no CUDA. El muestreo GPU agregado no demuestra ocupación por SM ni saturación de toda la VRAM.",
        "La ejecución de IPython puede registrar un fallo de historial SQLite de usuario en sandbox; stdout/stderr se conservan y no afecta los procesos de entrenamiento, cuyos códigos, tiempos y checkpoints se verifican independientemente. No se utiliza ese historial como evidencia experimental.",
        "- [Configuración congelada](configuracion.json), [serie](serie.json), [tiempos crudos](tiempos_medidos.csv).",
        "- [Resumen estadístico](resumen.csv), [speedup](speedup.csv), [comparación individual](comparacion_cnn_b.csv).",
        "- [Comparabilidad y diferencias aceptadas](comparabilidad.csv), [ejecutor efectivo por fase](tiempos_clasificados.csv) y [fuentes transitivas](fuentes_transitivas_verificadas.json).",
        "- [Calidad](calidad.csv), [recursos](recursos_resumen.csv), [checkpoints](validacion_checkpoints.json).",
        "- [Notebook fuente](../../../notebooks/Lab3_ASL_CUDA.ipynb) y [procedimiento](../../../docs/trazabilidad/implementacion-cuda.md).",
        "- [Notebook que lanzó las mediciones](notebook-ejecutado.ipynb), [vista ejecutada validada](notebook-validado.ipynb) y [código de salida/serie sin repeticiones](notebook-validado.salida.json).",
        "", "Cada RUN contiene invocación exacta, contexto/fuentes, hashes, logs, CSV por época, transferencias y recursos. Los archivos artefactos.csv se verifican antes de resumir. Para no repetir corridas, el notebook reutiliza la serie si ya está completa; una serie incompleta exige diagnóstico.","",
        "## Capturas reales de uso", "",
        f"[Índice de capturas y muestras originales](../../evidencias/{experiment}/README.md). Se conservan los intentos preliminares excluidos por DPI, sin sobrescribirlos." if captures else "No hay capturas seleccionadas de este EXP; no se reutilizan imágenes de otra ejecución.","",
        *[f"![{label}](../../evidencias/{experiment}/{name})\n" for name,label in captures],
        "## Gráficas", ""]
    for image in sorted(chart_dir.glob("*.png")):lines.append(f"![{image.stem}](graficas/{image.name})\n")
    lines += ["## Informe PDF", "", "[Informe de ejecución CUDA](informe-paso-5.pdf). Los Markdown y CSV son las fuentes verificables; el PDF no sustituye datos crudos.",""]
    report=readable("\n".join(lines))
    (folder/"resultados.md").write_text(report,encoding="utf8")
    with PdfPages(folder/"informe-paso-5.pdf") as pdf:
        fig=plt.figure(figsize=(8.27,11.69));ax=fig.add_axes([.08,.06,.84,.88]);ax.axis("off")
        text=f"Proyecto 3 CUDA — Paso 5\nCNN-B · ejecución real en PCUDA\n\nCPU: {train_cpu['media_s']:.3f} ± {train_cpu['desviacion_muestral_s']:.3f} s (DE, n=5)\nGPU: {train_gpu['media_s']:.3f} ± {train_gpu['desviacion_muestral_s']:.3f} s (DE, n={len(gpu)})\nSpeedup entrenamiento completo: {speed:.3f}×\nTest accuracy GPU: {min(metric_values)*100:.3f}–{max(metric_values)*100:.3f}%\n\n"
        paragraphs=["Incluye transferencias H2D/D2H, generación de máscaras, administración y validación CPU. Mismos archivos, splits, arquitectura, pesos iniciales,15épocas, batch128 y1440pasos. No se repitió la campaña CPU; se reutilizan cinco fases CNN-B de EXP-002.",
            "La preparación CPU optimizada usa6workers; su mejora no se atribuye a CUDA. Los generadores aleatorios difieren: se conserva distribución/volumen de trabajo, no identidad de trayectorias numéricas. Referencia CPU histórica no alternada; se revela la limitación térmica/temporal.",
            "Kernels propios Numba-CUDA Python, cuBLAS SGEMM FP32 pedantic, GTX1650 con4GiB. CPU se encarga de archivos, preparación, validación y test. Jerarquías de kernels propios explícitas; geometría cuBLAS gestionada por NVIDIA.",
            "Fuentes completas: resultados.md, configuracion.json, tiempos_medidos.csv, resumen.csv, speedup.csv y carpetas RUN con hashes/logs/recursos/checkpoints. Ver DEC-021/022/023 y documentos de implementación para pruebas y límites."]
        text+="\n\n".join(textwrap.fill(p,90) for p in paragraphs)
        ax.text(0,1,readable(text),va="top",fontsize=11,linespacing=1.5)
        pdf.savefig(fig);plt.close(fig)
        fig=plt.figure(figsize=(8.27,11.69))
        fig.text(.08,.95,"Tablas comparativas de tiempos reales",fontsize=15,weight="bold")
        fig.text(.08,.92,"Medias de cinco corridas CPU históricas y cinco corridas CUDA",fontsize=10)
        ax=fig.add_axes([.08,.64,.84,.23]);ax.axis("off")
        names={"preparacion_cpu_s":"Preparación CPU (1 vs 6 workers)","entrenamiento_s":"Entrenamiento CNN-B completo",
               "evaluacion_cpu_s":"Evaluación final CPU","suma_comparable_s":"Suma de tres fases comparables"}
        table=ax.table(cellText=[[names[r['fase']],f"{r['media_cpu_s']:.3f}",f"{r['media_gpu_s']:.3f}",f"{r['speedup']:.3f}×"] for r in ratios],
            colLabels=["Fase","CPU (s)","Variante CUDA (s)","CPU/variante"],colWidths=[.46,.17,.22,.15],loc="center",cellLoc="left")
        table.auto_set_font_size(False);table.set_fontsize(8.5);table.scale(1,1.9)
        fig.text(.08,.60,"Entrenamiento incluye transferencias H2D/D2H, staging y validación CPU.\nLa mejora de preparación corresponde a workers CPU, no a CUDA.",fontsize=9)
        fig.text(.08,.54,"Resultados individuales de las corridas CUDA",fontsize=12,weight="bold")
        ax=fig.add_axes([.08,.29,.84,.21]);ax.axis("off")
        table=ax.table(cellText=[[f"GPU {i+1:02d}",f"{r['preparacion_cpu_s']:.3f}",f"{r['entrenamiento_s']:.3f}",f"{r['evaluacion_cpu_s']:.3f}",f"{m['accuracy']*100:.3f}%"] for i,(r,m) in enumerate(zip(gpu,gpu_metrics))],
            colLabels=["Repetición","Prep CPU (s)","Train total (s)","Test CPU (s)","Accuracy test"],loc="center",cellLoc="center")
        table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,1.7)
        fig.text(.08,.21,"RUN completos, fechas UTC, incertidumbre y datos sin redondear:\ncomparacion_cnn_b.csv, tiempos_medidos.csv, resumen.csv y calidad.csv.\nNo se compara el total adaptativo de todos los modelos contra una sola CNN.",fontsize=9)
        pdf.savefig(fig);plt.close(fig)
        for fig in figures:pdf.savefig(fig)
        for name,label in captures:
            evidence=evidence_folder/name
            if evidence.is_file():
                fig,ax=plt.subplots(figsize=(10,7.5))
                ax.imshow(plt.imread(evidence));ax.axis("off");ax.set_title(label)
                fig.tight_layout();pdf.savefig(fig);plt.close(fig)
    for fig in figures:plt.close(fig)
    print("INFORME",folder/"resultados.md",flush=True)
    return folder/"resultados.md"


def read_csv_header(path):
    with Path(path).open(encoding="utf8",newline="") as f:return next(csv.reader(f))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--experimento",default="EXP-003")
    generar(ap.parse_args().experimento)
