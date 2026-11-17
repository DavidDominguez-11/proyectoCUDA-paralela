"""Rebuild CPU tables, statistical report and figures from recorded runs."""
from __future__ import annotations

import csv
import argparse
from datetime import datetime, timezone
import hashlib
import json
import io
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "resultados/secuencial/EXP-002"
COMMON = ["preparacion_cpu", "entrenamiento_CNN-A", "evaluacion_CNN-A",
          "entrenamiento_CNN-B", "evaluacion_CNN-B", "entrenamiento_FC-A",
          "evaluacion_FC-A", "entrenamiento_FC-B", "evaluacion_FC-B", "pca",
          "entrenamiento_RF-A", "validacion_RF-A", "entrenamiento_RF-B", "validacion_RF-B"]


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def main():
    global EXP
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experimento",default="EXP-002")
    args=parser.parse_args()
    if not args.experimento.startswith("EXP-") or not args.experimento[4:].isdigit():
        raise ValueError("Expected EXP-NNN.")
    EXP=ROOT/"resultados/secuencial"/args.experimento
    import numpy as np
    import pandas as pd
    from scipy.stats import t
    os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"data"/EXP.name/"matplotlib-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import h5py
    config=load_json(ROOT/"configuraciones"/(EXP.name+".json"))
    series=load_json(EXP/"serie.json")
    if series["estado"] not in ("completada","completada_alcance_comun_5"):
        raise ValueError("Series is not complete; preserve partial data without final claims.")
    runs=[EXP/name for name in series["runs"]]
    tables=[pd.read_csv(run/"tiempos.csv",keep_default_na=False) for run in runs]
    data=pd.concat(tables,ignore_index=True)
    if not (data.configuracion_sha256==sha(ROOT/"configuraciones"/(EXP.name+".json"))).all():
        raise ValueError("Measured configuration differs from current configuration.")
    if data.duplicated(["run_id","fase"]).any(): raise ValueError("Duplicate phase observations.")
    for table in tables:
        if set(table[table.fase.isin(COMMON)].fase)!=set(COMMON):
            raise ValueError("A run is missing common reference phases.")
    initial=[load_json(run/"estados-iniciales.json") for run in runs]
    for snapshot in initial:
        for item in snapshot.values():
            path=ROOT/item["archivo"]
            if sha(path)!=item["archivo_sha256"]:
                raise ValueError("Initial checkpoint file identity changed.")
            fingerprint=hashlib.sha256()
            with np.load(path,allow_pickle=False) as arrays:
                if len(arrays.files)!=item["variables"]:
                    raise ValueError("Initial checkpoint variable count changed.")
                for index in range(item["variables"]):
                    value=arrays[f"arr_{index}"]
                    fingerprint.update(str((value.shape,str(value.dtype))).encode())
                    fingerprint.update(value.tobytes())
            if fingerprint.hexdigest()!=item["state_sha256"]:
                raise ValueError("Initial checkpoint content differs from recorded state.")
    tensors=[load_json(run/"tensores.json") for run in runs]
    metrics=[load_json(run/"metricas.json") for run in runs]
    step_audit=[]
    for run,table,metric in zip(runs,tables,metrics):
        epochs=pd.read_csv(run/"epocas.csv")
        if len(epochs)!=75 or not (epochs.groupby("modelo").size()==15).all():
            raise ValueError("Measured neural workload differs from 5x15 epochs.")
        if not np.isfinite(epochs.select_dtypes(include="number").to_numpy()).all():
            raise ValueError("Non-finite training metrics.")
        neural=table[table.fase.str.startswith("entrenamiento_") & ~table.fase.str.contains("RF-")]
        if len(neural)!=5 or not (neural.pasos.astype(int)==1440).all():
            raise ValueError("Measured steps differ from configured workload.")
        for item in metric["modelos"].values():
            if not all(np.isfinite(value) for value in item.values()):
                raise ValueError("Non-finite final metrics.")
        checkpoints=load_json(run/"modelos-guardados.json")
        for name,item in checkpoints.items():
            path=ROOT/item["ruta"]
            if sha(path)!=item["sha256"]:
                raise ValueError("Saved checkpoint identity changed.")
            if path.suffix==".keras":
                with zipfile.ZipFile(path) as archive:
                    with h5py.File(io.BytesIO(archive.read("model.weights.h5")),"r") as weights:
                        steps=int(weights["optimizer/vars/0"][()])
                expected=config["carga"]["epocas_fijas"]*config["carga"]["pasos_por_epoca"]
                if steps!=expected: raise ValueError(f"Unexpected Adam updates: {run.name}/{name}: {steps}")
                step_audit.append(dict(run_id=run.name,modelo=name,actualizaciones_Adam=steps,
                                       esperadas=expected,checkpoint_sha256=item["sha256"]))
    pd.DataFrame(step_audit).to_csv(EXP/"pasos_verificados.csv",index=False)
    same_initial=all({k:v["state_sha256"] for k,v in item.items()}==
                     {k:v["state_sha256"] for k,v in initial[0].items()} for item in initial)
    same_tensors=all(item==tensors[0] for item in tensors)
    same_selection=all((m["mejor_cnn"],m["mejor_rf"],m["modelo_final"])==
                       (metrics[0]["mejor_cnn"],metrics[0]["mejor_rf"],metrics[0]["modelo_final"]) for m in metrics)
    same_training_selection=len({(m["mejor_cnn"],m["mejor_rf"]) for m in metrics})==1
    # At inference, random augmentation layers are inactive. B and B+aug share
    # the same mathematical architecture, although their learned weights differ.
    same_final_architecture=len({m["modelo_final"].removesuffix("+aug") for m in metrics})==1
    same_packages=len({sha(run/"contexto/paquetes-instalados.txt") for run in runs})==1
    photo_sets=[]
    for run in runs:
        with (run/"contexto/fuentes.csv").open(encoding="utf-8",newline="") as handle:
            photo_sets.append({row["ruta_relativa"]:row["sha256"] for row in csv.DictReader(handle)
                               if row["ruta_relativa"].startswith("LAB3-DATASCIENCE/imagenes_grupo/")})
    same_photos=all(item==photo_sets[0] for item in photo_sets) and len(photo_sets[0])==15
    if not all((same_initial,same_tensors,same_packages,same_photos)):
        raise ValueError("Inputs, initial state or packages differ between runs.")
    data.to_csv(ROOT/"resultados/secuencial/tiempos.csv",index=False)
    core=[]
    for run,table in zip(runs,tables):
        selected=table[table.fase.isin(COMMON)]
        core.append(dict(run_id=run.name,repeticion=int(table.repeticion.iloc[0]),
                         duracion_s=float(selected.duracion_s.astype(float).sum()),fases=14))
    core=pd.DataFrame(core)
    core.to_csv(EXP/"tiempos_referencia_comun.csv",index=False)
    by_run={run.name:metric for run,metric in zip(runs,metrics)}
    data["fase_original"]=data.fase
    for index,row in data.iterrows():
        metric=by_run[row.run_id]
        if row.fase=="diagnostico_cnn": data.at[index,"fase"]="diagnostico_"+metric["mejor_cnn"]
        elif row.fase=="evaluacion_RF": data.at[index,"fase"]="evaluacion_"+metric["mejor_rf"]
        elif row.fase=="inferencia_grupo":
            data.at[index,"fase"]=f"inferencia_grupo_aug_{metric['mejor_cnn']}_final_{metric['modelo_final']}"
    data.to_csv(EXP/"tiempos_clasificados.csv",index=False)
    def describe(phase,values):
        values=values.astype(float)
        n=len(values); std=float(values.std(ddof=1)); avg=float(values.mean())
        half=float(t.ppf(.975,n-1))*std/np.sqrt(n) if n>1 else float("nan")
        return dict(fase=phase,n=n,media_s=avg,mediana_s=float(values.median()),
            desviacion_muestral_s=std,min_s=float(values.min()),max_s=float(values.max()),
            cv_pct=100*std/avg,ic95_inferior_s=avg-half,ic95_superior_s=avg+half,
            semiancho_relativo_ic95_pct=100*half/avg,
            anidada=phase.startswith("validacion_anidada_"))
    summary=[describe(phase,group.duracion_s.astype(float)) for phase,group in data.groupby("fase",sort=False)]
    stats=pd.DataFrame(summary)
    stats.to_csv(EXP/"resumen.csv",index=False)
    reference=pd.Series(describe("referencia_comun",core.duracion_s))
    pd.DataFrame([reference]).to_csv(EXP/"resumen_referencia_comun.csv",index=False)
    quality=[]
    for name in sorted(set().union(*(m["modelos"] for m in metrics))):
        values=[m["modelos"][name]["accuracy"] for m in metrics if name in m["modelos"]]
        quality.append(dict(modelo=name,n=len(values),accuracy_min=min(values),accuracy_max=max(values),
                            diferencia=max(values)-min(values) if len(values)>1 else float("nan")))
    pd.DataFrame(quality).to_csv(EXP/"repetibilidad-metricas.csv",index=False)
    numeric_reproducible=all(row["diferencia"]<=config["validacion"]["accuracy_tolerance_across_repetitions"] for row in quality)
    (EXP/"verificacion-equivalencia.json").write_text(json.dumps(dict(
        tensors_identical=same_tensors,initial_states_identical=same_initial,
        packages_identical=same_packages,photos_identical=same_photos,
        selection_identical=same_selection,accuracy_within_declared_tolerance=numeric_reproducible,
        training_architecture_selection_identical=same_training_selection,
        final_effective_architecture_identical=same_final_architecture,
        common_reference_load_equivalent=True,common_reference_phases=COMMON,
        global_adaptive_pipeline_load_equivalent=same_training_selection and same_final_architecture,
        accuracy_tolerance=config["validacion"]["accuracy_tolerance_across_repetitions"]),indent=2)+"\n",encoding="utf-8")
    resource_stats=[]
    for run in runs:
        resources=pd.read_csv(run/"recursos.csv")
        resource_stats.append(dict(run_id=run.name,muestras=len(resources),
            cpu_arbol_media_pct=float(resources.cpu_proceso_pct.mean()),
            cpu_arbol_max_pct=float(resources.cpu_proceso_pct.max()),
            rss_arbol_max_gib=float(resources.rss_bytes.max()/2**30),
            ram_disponible_min_gib=float(resources.ram_disponible_bytes.min()/2**30)))
    pd.DataFrame(resource_stats).to_csv(EXP/"recursos_resumen.csv",index=False)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.spines.top":False,
                         "axes.spines.right":False,"savefig.dpi":160})
    figures=EXP/"graficas"; figures.mkdir(exist_ok=True)
    totals=data[data.fase=="pipeline_total"].sort_values("repeticion")
    total=stats[stats.fase=="pipeline_total"].iloc[0]
    branches={run.name:m["mejor_cnn"]+"+aug" for run,m in zip(runs,metrics)}
    totals["rama"]=totals.run_id.map(branches)
    totals.to_csv(EXP/"pipeline_por_rama.csv",index=False)
    branch_summary=pd.DataFrame([describe(branch,group.duracion_s.astype(float))
        for branch,group in totals.groupby("rama",sort=False)])
    branch_summary.to_csv(EXP/"resumen_pipeline_por_rama.csv",index=False)
    fig,(ax,delta_ax)=plt.subplots(2,1,figsize=(9,7),layout="constrained",sharex=True,
                                  gridspec_kw={"height_ratios":[2,1]})
    colors=["#D55E00" if branches[name]=="CNN-A+aug" else "#0072B2" for name in totals.run_id]
    ax.bar(totals.repeticion.astype(int),totals.duracion_s.astype(float)/60,color=colors,width=.6)
    ax.axhline(total.media_s/60,color="#333333",label=f"Media descriptiva: {total.media_s/60:.2f} min")
    ax.axhspan(total.ic95_inferior_s/60,total.ic95_superior_s/60,color="#333333",alpha=.1,label="IC t 95% nominal")
    from matplotlib.patches import Patch
    handles,labels_legend=ax.get_legend_handles_labels()
    for branch,color in (("CNN-A+aug","#D55E00"),("CNN-B+aug","#0072B2")):
        if branch in set(branches.values()): handles.append(Patch(color=color));labels_legend.append(branch)
    ax.set(ylabel="Tiempo de pared (min)",title=f"Programa adaptativo · ramas diferentes · n={len(runs)}",
           xticks=totals.repeticion.astype(int)); ax.set_ylim(bottom=0)
    ax.legend(handles,labels_legend,loc="upper left",bbox_to_anchor=(1.01,1));ax.grid(axis="y",alpha=.2)
    delta_ax.plot(totals.repeticion.astype(int),totals.duracion_s.astype(float)-total.media_s,
                  marker="o",color="#0072B2")
    delta_ax.axhline(0,color="#D55E00",linewidth=1)
    delta_ax.set(xlabel="Repetición (proceso nuevo)",ylabel="Diferencia respecto\na la media (s)")
    delta_ax.grid(axis="y",alpha=.2)
    fig.savefig(figures/"pipeline_repeticiones.png"); plt.close(fig)
    fig,(ax,delta_ax)=plt.subplots(2,1,figsize=(9,7),layout="constrained",sharex=True,
                                  gridspec_kw={"height_ratios":[2,1]})
    ax.bar(core.repeticion,core.duracion_s/60,color="#0072B2",width=.6)
    ax.axhline(reference.media_s/60,color="#D55E00",label=f"Media: {reference.media_s/60:.3f} min")
    ax.axhspan(reference.ic95_inferior_s/60,reference.ic95_superior_s/60,color="#D55E00",alpha=.15,label="IC t 95% de la media")
    ax.set(ylabel="Suma de tiempos de pared (min)",title=f"Referencia CPU común · 14 fases iguales · n={len(runs)}",
           xticks=core.repeticion);ax.set_ylim(bottom=0)
    ax.legend(loc="upper left",bbox_to_anchor=(1.01,1));ax.grid(axis="y",alpha=.2)
    delta_ax.plot(core.repeticion,core.duracion_s-reference.media_s,marker="o",color="#0072B2")
    delta_ax.axhline(0,color="#D55E00",linewidth=1)
    delta_ax.set(xlabel="Repetición (proceso nuevo)",ylabel="Diferencia respecto\na la media (s)");delta_ax.grid(axis="y",alpha=.2)
    fig.savefig(figures/"referencia_comun_repeticiones.png");plt.close(fig)
    main_phases=stats[(~stats.anidada)&(stats.fase!="pipeline_total")].sort_values("media_s")
    fig,ax=plt.subplots(figsize=(12,max(7,len(main_phases)*.4)),layout="constrained")
    labels=[]
    for index,row in enumerate(main_phases.itertuples()):
        values=data[data.fase==row.fase].duracion_s.astype(float).to_numpy()
        error=row.desviacion_muestral_s if np.isfinite(row.desviacion_muestral_s) else None
        ax.errorbar(row.media_s,index,xerr=error,fmt="s",color="#0072B2",capsize=3)
        ax.scatter(values,index+np.linspace(-.13,.13,len(values)),color="#555555",s=14,alpha=.8)
        labels.append(row.fase.replace("_"," ")+f" (n={row.n})")
    ax.set(yticks=range(len(labels)),yticklabels=labels,xlabel="Tiempo por fase (s, escala logarítmica)",
           title="Fases CPU · cuadrados: media ± desviación muestral; puntos: repeticiones")
    ax.set_xscale("log"); ax.grid(axis="x",alpha=.2)
    fig.savefig(figures/"fases_cpu.png"); plt.close(fig)
    # Preserve measured monitoring signals; this is a derived plot, not a desktop screenshot.
    monitor=pd.read_csv(runs[0]/"recursos.csv")
    core_columns=[c for c in monitor if c.startswith("cpu_") and c.endswith("_pct") and c[4:-4].isdigit()]
    fig,axes=plt.subplots(3,1,figsize=(13,9),layout="constrained",sharex=True)
    minutes=monitor.transcurrido_s.to_numpy()/60
    image=axes[0].imshow(monitor[core_columns].to_numpy().T,aspect="auto",origin="lower",vmin=0,vmax=100,
        extent=[minutes[0],minutes[-1],-.5,len(core_columns)-.5],cmap="cividis")
    axes[0].set(ylabel="Procesador lógico",yticks=range(len(core_columns)),title="Monitorización real · primera repetición (incluye inicialización)")
    fig.colorbar(image,ax=axes[0],label="Uso CPU (%)")
    axes[1].plot(minutes,monitor.cpu_proceso_pct,label="Árbol de procesos · 100% = 1 CPU lógico",color="#0072B2")
    axes[1].set_ylabel("CPU del árbol (%)"); axes[1].set_ylim(bottom=0); axes[1].legend(loc="upper right"); axes[1].grid(alpha=.2)
    axes[2].plot(minutes,monitor.rss_bytes/2**30,label="RSS agregado del árbol",color="#0072B2")
    axes[2].plot(minutes,monitor.ram_disponible_bytes/2**30,label="RAM disponible del sistema",color="#D55E00")
    axes[2].set(xlabel="Tiempo desde inicio del proceso (min)",ylabel="Memoria (GiB)"); axes[2].legend(); axes[2].grid(alpha=.2)
    fig.savefig(figures/"recursos_cpu.png"); plt.close(fig)
    counts=load_json(EXP/"validacion_datos.json")
    common_training=data[data.fase.isin(COMMON)&data.fase.str.startswith("entrenamiento_")]
    fraction=100*common_training.duracion_s.sum()/core.duracion_s.sum()
    def fmt(value,digits=3):
        return f"{value:.{digits}f}" if np.isfinite(value) else "N/A"
    lines=["# Paso 2. Resultados detallados de ejecución CPU", "",
        f"Generado UTC: {datetime.now(timezone.utc).isoformat()}. EXP-001; PCUDA; sin reinstalación ni descarga.","",
        "## Resultado principal: referencia de carga común", "",
        f"En {len(runs)} procesos nuevos, las14 fases comunes tardaron **{reference.media_s:.3f} s ({reference.media_s/60:.3f} min) en promedio**.",
        f"DE muestral: {reference.desviacion_muestral_s:.3f} s; mediana: {reference.mediana_s:.3f} s; CV: {reference.cv_pct:.2f}%; IC t95% nominal: [{reference.ic95_inferior_s:.3f}, {reference.ic95_superior_s:.3f}] s.",
        f"Semiancho relativo del IC común: {reference.semiancho_relativo_ic95_pct:.2f}%; objetivo revisado ≤5%: {'cumplido' if reference.semiancho_relativo_ic95_pct<=5 else 'no cumplido'}.",
        "Esta referencia suma14 intervalos de pared no solapados: preparación, fit/evaluate de las cuatro redes base, PCA y fit/validación de los dos bosques. Son60 épocas neuronales base por corrida; la validación interna de fit ya está incluida. No es el tiempo continuo del programa completo ni tiempo exclusivo de kernels.",
        "La referencia común procede de DEC-015, definida en EXP-002 al detectar un cambio de arquitectura en la cuarta corrida, antes del resultado de la quinta. Se conservaron todas las observaciones; no se eligieron fases por ser más rápidas ni se descartó la rama distinta.", "",
        "## Programa adaptativo bajo el protocolo de 15 épocas: total descriptivo", "",
        f"El pipeline continuo tardó **{total.media_s:.3f} s ({total.media_s/60:.3f} min) en promedio descriptivo**, mezclando las ramas realmente elegidas.",
        f"Desviación muestral: {total.desviacion_muestral_s:.3f} s; CV: {total.cv_pct:.2f}%; IC t 95% de la media: [{total.ic95_inferior_s:.3f}, {total.ic95_superior_s:.3f}] s.",
        f"Semiancho relativo del IC del total: {total.semiancho_relativo_ic95_pct:.2f}%. La regla inicial de precisión del total se suspendió por falta de homogeneidad: no se afirma cumplida ni se usa esta mezcla como baseline de carga fija para GPU.",
        "El programa selecciona la CNN a aumentar por test accuracy. Se observó: "+"; ".join(f"repetición {i+1}: {m['mejor_cnn']}+aug" for i,m in enumerate(metrics))+". Cambiar de arquitectura modifica el cómputo de las 15 épocas. Diagnóstico e inferencia se estratifican según la selección.", "",
        "No se han medido GPU, transferencias ni speedup. El IC describe estas condiciones; no garantiza independencia ni ausencia de deriva térmica.","",
        "## Datos y alcance", "",
        "Dataset local ASL Alphabet v1: 87,000 archivos inventariados; 17,400 seleccionados (600/clase), splits 12,180/2,610/2,610. Fotos propias: 15.",
        f"Se identificaron {len(counts['duplicate_groups'])} grupos de hashes duplicados en la selección y {counts['cross_split_duplicate_groups']} grupos entre splits. No se excluyeron archivos. La ausencia de duplicados exactos no demuestra independencia de sujetos/sesiones.",
        "Se ejecutan las funciones originales de preparación y construcción, cinco entrenamientos neuronales, PCA, dos bosques y evaluación/inferencia. Se excluyen descarga, EDA, renderizado y recarga redundante. El notebook original permanece intacto.",
        "Diferencias controladas:15 épocas por red, EarlyStopping desactivado sin restaurar mejor época, seeds/threads explícitos. La longitud por red es fija, pero la arquitectura adaptativa no siempre coincide; por eso se separa la referencia común del total.","",
        "Por ese ajuste experimental, estos tiempos no equivalen a ejecutar el notebook íntegro con su parada temprana, restauración del mejor epoch, EDA y figuras. Se documenta el alcance medido y se conserva el original.", "",
        "## Condiciones y procedimiento", "",
        f"CPU: {config['cpu']['modelo']}; {config['cpu']['nucleos_fisicos']} núcleos físicos/{config['cpu']['nucleos_logicos']} lógicos; RAM {config['cpu']['ram_bytes']/2**30:.2f} GiB.",
        "PCUDA; TensorFlow CPU, GPU ocultada y dispositivo efectivo registrado. Intra-op6, inter-op1, BLAS6; RF conserva n_jobs=-1. Preparación de imágenes con el bucle original de un worker.",
        "Cada repetición valida hashes, construye/calienta candidatos y restaura todas sus variables/optimizador; después cronometra preparación y operaciones del programa. Warmup y checkpoints quedan fuera del total.",
        "Temporizador: perf_counter_ns. El total es un intervalo continuo; incluye administración y logging internos. Las fases de validación anidada ya están incluidas en fit y no deben sumarse dos veces.",
        "La caché de archivos del SO no se vació y los hashes se leyeron antes del timing: estos tiempos no describen una primera lectura de disco frío. Imports y arranque del proceso se registran pero no se incluyen en pipeline_total.",
        "Se conservaron todas las repeticiones; no se eliminaron observaciones por ser lentas ni por accuracy. Monitorización externa cada1s. Un supervisor de cierre se añadió al detectar la rama distinta: lee estado cada10s y solo acelera el sondeo durante inferencia, fuera de las14 fases comunes.",
        "El controlador original y su configuración permanecieron intactos. El supervisor lo detuvo después de cinco salidas correctas para evitar ampliar una mezcla como si fuera homogénea; estado previo y revisión en cierre-alcance.md/serie-antes-cierre.json.", "",
        "## Tiempos por repetición", "", "| Repetición | RUN | Referencia común (s) | Total adaptativo (s) | CNN aumentada | Modelo final |", "|---:|---|---:|---:|---|---|"]
    for row in totals.itertuples():
        common_seconds=float(core[core.run_id==row.run_id].duracion_s.iloc[0])
        lines.append(f"| {row.repeticion} | [{row.run_id}]({row.run_id}/registro.md) | {common_seconds:.3f} | {row.duracion_s:.3f} | {branches[row.run_id]} | {by_run[row.run_id]['modelo_final']} |")
    lines += ["", "## Total estratificado por CNN aumentada", "",
              "| Rama | n | Media total (s) | DE muestral (s) | IC t 95% nominal (s) |",
              "|---|---:|---:|---:|---|"]
    for row in branch_summary.itertuples():
        interval=f"{row.ic95_inferior_s:.3f}–{row.ic95_superior_s:.3f}" if row.n>1 else "N/A"
        lines.append(f"| {row.fase} | {row.n} | {row.media_s:.3f} | {fmt(row.desviacion_muestral_s)} | {interval} |")
    lines += ["", "## Estadísticas por fase", "", "| Fase | n | Media (s) | Mediana (s) | DE muestral (s) | Mín–máx (s) | CV (%) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in stats.itertuples():
        label=row.fase+(" (anidada)" if row.anidada else "")
        lines.append(f"| {label} | {row.n} | {row.media_s:.3f} | {row.mediana_s:.3f} | {fmt(row.desviacion_muestral_s)} | {row.min_s:.3f}–{row.max_s:.3f} | {fmt(row.cv_pct,2)} |")
    lines += ["", "## Gráficas", "",
        "![Referencia común, mismos modelos y pasos](graficas/referencia_comun_repeticiones.png)", "",
        "![Tiempos individuales y media](graficas/pipeline_repeticiones.png)", "",
        "![Desglose de fases y repeticiones](graficas/fases_cpu.png)", "",
        "![Uso medido de CPU y RAM](graficas/recursos_cpu.png)", "",
        "La última figura se deriva del CSV real del monitor; no es una captura del Administrador de tareas ni prueba de GPU. Las fases con n=1 no tienen estimación de DE/IC: se muestran N/A, sin inventar ceros.", "",
        "## Corrección y repetibilidad", "",
        f"Identidad de tensores entre corridas: {same_tensors}. Identidad del estado inicial completo por arquitectura: {same_initial}. Misma selección de arquitecturas/modelos: {same_selection}.",
        f"Paquetes idénticos: {same_packages}; fotos propias idénticas: {same_photos}; accuracy dentro de la tolerancia predefinida1e-6: {numeric_reproducible}. La repetibilidad numérica se informa separada de la equivalencia de carga.",
        f"Selección de arquitectura de entrenamiento constante: {same_training_selection}; arquitectura efectiva de inferencia constante: {same_final_architecture}. Los ganadores exactos se conservan en metricas.json por RUN; no se descartan corridas por menor accuracy.",
        "Los checkpoints iniciales incluyen variables de modelo, estadísticas BN, estados aleatorios y optimizador. Se validó su restauración tras warmup. Shapes, dtype, rango y valores finitos se comprobaron después del intervalo medido.", "",
        "Los archivos .keras finales se verificaron por SHA-256 y su contador de Adam contiene1,440 actualizaciones por red: evidencia adicional de los pasos efectivos y de que el warmup no añadió actualizaciones al entrenamiento medido. Auditoría en pasos_verificados.csv.", "",
        "| Modelo | n | Accuracy test mínimo | Accuracy test máximo | Diferencia |", "|---|---:|---:|---:|---:|"]
    for row in quality:
        lines.append(f"| {row['modelo']} | {row['n']} | {row['accuracy_min']:.6f} | {row['accuracy_max']:.6f} | {fmt(row['diferencia'],8)} |")
    lines += ["", "## Análisis de rendimiento y factores", "",
        f"Las fases de entrenamiento base y bosques representan {fraction:.2f}% del tiempo contabilizado en la referencia común. La tabla identifica operaciones dominantes sin atribuir a kernels el tiempo completo de fit.",
        f"El fit de CNN-B consume por sí solo {100*float(stats[stats.fase=='entrenamiento_CNN-B'].media_s.iloc[0])/reference.media_s:.2f}% de la referencia común; es la fase base dominante observada. Preparación CPU representa {100*float(stats[stats.fase=='preparacion_cpu'].media_s.iloc[0])/reference.media_s:.2f}%.",
        "El cambio de rama adaptativa es un factor de rendimiento observado directamente: aumentar CNN-A y aumentar CNN-B no ejecutan la misma arquitectura. Se conservan ambos tiempos con su n; mezclar sus fases ocultaría la causa de la dispersión del programa completo.",
        "La mayor profundidad, mapas de activación y BatchNormalization de CNN-B incrementan el trabajo frente a CNN-A; el aumento añade transformaciones por lote. La comparación entre modelos describe cargas diferentes, no speedup de implementaciones equivalentes.",
        "PCA reduce el coste del bosque al transformar12,288 variables en150, pero su ajuste/proyección también consume tiempo. Los bosques usan varios hilos; esta referencia CPU no debe llamarse monohilo.",
        "Lectura/decodificación, resize y normalización están combinados en preparacion_cpu; este experimento no permite atribuir causalmente sus costes individuales. Disco, caché y competencia con otros procesos influyen en esa fase.",
        "Batch128 y15 épocas fijan los pasos de cada arquitectura base y reutilizan imágenes en RAM; la selección posterior sigue siendo adaptativa. EarlyStopping habría añadido otra fuente de variación. Fit incluye validación y sobrecostes de Keras.",
        "Frecuencia dinámica, refrigeración, plan de energía, carga externa y presión de RAM pueden producir variación. Se conserva CPU/RAM por segundo; no se midió potencia ni temperatura, por lo que no se afirma throttling como causa demostrada.",
        "Las predicciones individuales de fotos propias generan numerosas invocaciones pequeñas; difieren del throughput de inferencia por lotes. El control, adaptación de datos y llamadas del framework forman parte de ese tiempo.",
        "Los resultados corresponden a este portátil, este entorno y esta selección. Elegir por test es una limitación heredada de calidad; no se pretendió corregirla ni ajustar dependencias para forzar coincidencia de accuracy durante el benchmark.","",
        "## Reproducción y artefactos", "",
        "Configuración ejecutada: [EXP-001.json](../../../configuraciones/EXP-001.json). [Manifiesto de fuentes](manifest_fuente.csv), [splits](manifest_splits.csv), [mapeo](label_mapping.json), [auditoría de datos](validacion_datos.json), [resumen por fase](resumen.csv), [referencia común](resumen_referencia_comun.csv), [serie](serie.json), [revisión del cierre](cierre-alcance.md), [preservación local de entradas](preservacion-datos.md).", "",
        "Guía de ejecución nueva y recuperación sin descarga: [reproducir-exp-002.md](../../../docs/trazabilidad/reproducir-exp-002.md). Desde la raíz, con PCUDA, regenerar únicamente el análisis:", "", "```powershell",
        ".\\PCUDA\\Scripts\\python.exe -X utf8 scripts/resumir_cpu.py", "```", "",
        "El controlador rechaza sobreescribir una serie iniciada. Para repetir experimentalmente, conservar EXP-001 y crear otro identificador/configuración con los mismos manifiestos y pesos iniciales; no borrar resultados para reutilizar el nombre.",
        "Los checkpoints se conservan localmente en data/EXP-001, con rutas y hashes en cada RUN. No se descargan modelos ni datos. Esta ejecución no demuestra todavía reconstrucción del entorno desde cero ni comparación GPU."]
    lines += ["", "Los intervalos t son nominales y el alcance común se revisó durante el estudio por una rama adaptativa observada. Se presenta esta revisión exploratoria, sus motivos y todos los datos; no se promete cobertura exacta ni inferencia sobre otras máquinas.",
              "La serie previa EXP-001 fue interrumpida por registrar el lanzador del venv en lugar del hijo TensorFlow. Se conserva como fallida y no aporta observaciones al promedio; EXP-002 corrige el monitor del árbol de procesos.",
              "El entorno instalado emitió avisos sobre retracing y un atributo de MapDataset desconocido que el runtime indica ignorar. Se conservan en stderr; no se reinstaló el stack. No se demuestra su coste causal con esta serie."]
    if not numeric_reproducible:
        lines += ["", "**Límite observado:** no se reproduce la trayectoria de aprendizaje dentro de1e-6 en todas las redes, pese a seeds, estados iniciales y tensores coincidentes. No se ha aislado si la causa está en reducciones numéricas, estados aleatorios internos u otro detalle del stack. La carga de las14 fases comunes es equivalente; la del pipeline adaptativo completo no. No se certifica entrenamiento bit a bit ni se alteran dependencias para ocultar la variación."]
    lines += ["", "## Recursos observados", "",
              "Resumen de todo el proceso (incluye warmup y persistencia): CPU100% representa un procesador lógico; RSS suma working sets del árbol y puede duplicar memoria compartida.", "",
              "| RUN | CPU árbol media (%) | CPU árbol máximo (%) | RSS árbol pico (GiB) | RAM disponible mínima (GiB) |",
              "|---|---:|---:|---:|---:|"]
    for row in resource_stats:
        lines.append(f"| {row['run_id']} | {row['cpu_arbol_media_pct']:.2f} | {row['cpu_arbol_max_pct']:.2f} | {row['rss_arbol_max_gib']:.3f} | {row['ram_disponible_min_gib']:.3f} |")
    lines += ["", "Plan de energía consultado antes de la serie: ASUS Recommended; registro en plan-energia.txt. No se alteró ese plan. No se controlaron los procesos de otras aplicaciones."]
    report=("\n".join(lines)+"\n").replace("EXP-001",EXP.name)
    # Keep the provenance of the failed series distinct from the measured series.
    report=report.replace(f"La serie previa {EXP.name} fue", "La serie previa EXP-001 fue")
    for before,after in {
        "las14":"las 14", "suma14":"suma 14", "Son60":"Son 60", "de15":"de 15",
        "controladas:15":"controladas: 15", "Intra-op6":"Intra-op 6", "inter-op1":"inter-op 1",
        "BLAS6":"BLAS 6", "cada1s":"cada 1 s", "cada10s":"cada 10 s",
        "predefinida1e-6":"predefinida 1e-6", "contiene1,440":"contiene 1,440",
        "transformar12,288":"transformar 12,288", "en150":"en 150", "Batch128":"Batch 128",
        "y15":"y 15", "dentro de1e-6":"dentro de 1e-6", "CPU100%":"CPU 100%",
    }.items(): report=report.replace(before,after)
    (EXP/"resultados.md").write_text(report,encoding="utf-8")
    for run in runs:
        invocation=load_json(run/"invocacion.json")
        own=tables[runs.index(run)]
        duration=float(own[own.fase=="pipeline_total"].iloc[0].duracion_s)
        text=f"# {run.name}\n\nEstado: completada; CPU; EXP-001.\n\n"
        text+="REQ-03/07/12; DAT-01-v1/DAT-03-EXP-001/DAT-04/DAT-05; PROC-03/04/05/06; DEC-011/012/013; CAM-003; BIT-005.\n\n"
        text+=f"Responsable: Codex por solicitud del usuario. Revisión humana pendiente. Pipeline: {duration:.6f} s.\n\n"
        text+="Comando exacto e inicio: [invocacion.json](invocacion.json); fin/código: [salida.json](salida.json).\n\n"
        text+="[Configuración](configuracion.json), [contexto y fuentes](contexto/contexto.json), [hardware efectivo](hardware.json), [estado inicial](estados-iniciales.json), [warmup](warmup.json), [tensores](tensores.json).\n\n"
        text+="[Tiempos](tiempos.csv), [épocas](epocas.csv), [métricas](metricas.json), [recursos](recursos.csv), [stdout](stdout.log), [stderr](stderr.log), [modelos](modelos-guardados.json).\n\n"
        text+="Procedimiento y límites: [informe de EXP-001](../resultados.md). Validación anidada incluida en fit; sin medición GPU. Artefactos grandes en data/EXP-001, reconstruibles con fuentes/configuración/datos registrados.\n"
        path=run/"registro.md"
        if not path.exists(): path.write_text(text.replace("EXP-001",EXP.name),encoding="utf-8")
        manifest=run/"artefactos.csv"
        if not manifest.exists():
            with manifest.open("w",encoding="utf-8",newline="") as handle:
                writer=csv.writer(handle); writer.writerow(["ruta_relativa","bytes","sha256"])
                for path in sorted(run.rglob("*")):
                    if path.is_file() and path!=manifest:
                        writer.writerow([path.relative_to(run).as_posix(),path.stat().st_size,sha(path)])
        with manifest.open(encoding="utf-8",newline="") as handle:
            for item in csv.DictReader(handle):
                path=run/item["ruta_relativa"]
                if sha(path)!=item["sha256"] or path.stat().st_size!=int(item["bytes"]):
                    raise ValueError(f"Sealed artifact changed: {path}")
    print(f"Informe: {EXP/'resultados.md'}; n={len(runs)}, referencia_comun={reference.media_s:.3f}s, CV_comun={reference.cv_pct:.2f}%, total_adaptativo_descriptivo={total.media_s:.3f}s")


if __name__=="__main__": main()
