"""Read completed NEW FC runs and historical CPU/CNN-B; never train models."""
from pathlib import Path
import json,csv,hashlib,sys
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from benchmark_cpu import write_json,write_csv,sha
BASE=ROOT/'resultados/paralelo/EXP-012'
OUT=ROOT/'resultados/comparacion/ampliacion-fc'

def read_csv(p):return list(csv.DictReader(p.open(encoding='utf8',newline='')))
def desc(a):
    a=np.asarray(a,np.float64);mean=float(a.mean());sd=float(a.std(ddof=1));half=float(t.ppf(.975,len(a)-1)*sd/np.sqrt(len(a)))
    return dict(n=len(a),media_s=mean,de_s=sd,cv_pct=100*sd/mean,ic95_inf_s=mean-half,ic95_sup_s=mean+half,min_s=float(a.min()),max_s=float(a.max()))
def table(rows):
    keys=list(rows[0]);fmt=lambda v:f'{v:.6f}' if isinstance(v,float) else str(v)
    return '| '+' | '.join(keys)+' |\n| '+' | '.join(['---']*len(keys))+' |\n'+'\n'.join('| '+' | '.join(fmt(row.get(k,'')) for k in keys)+' |' for row in rows)
def main():
    if OUT.exists() and any(OUT.iterdir()):raise FileExistsError(OUT)
    assert json.loads((BASE/'estado-notebook.json').read_text(encoding='utf8'))['estado']=='completada'
    config=json.loads((ROOT/'configuraciones/EXP-012.json').read_text(encoding='utf8'))
    for p,digest in config['codigo']['fuentes'].items():assert sha(ROOT/p)==digest,p
    OUT.mkdir(parents=True,exist_ok=True)
    obs=[];resources=[];all_epochs=[];models={};sources=set()
    for item in config['runs']:
        run=ROOT/item.get('ruta',(BASE/item['run_id']).relative_to(ROOT).as_posix());metrics=json.loads((run/'metricas.json').read_text(encoding='utf8'))
        assert metrics['pasos']==1440 and metrics['epocas']==15
        epochs=read_csv(run/'epocas.csv');assert len(epochs)==15 and int(epochs[-1]['pasos_acumulados'])==1440
        all_epochs.extend(dict(origen='CUDA',run_id=run.name,**row) for row in epochs)
        saved=json.loads((run/'modelo-guardado.json').read_text(encoding='utf8'));assert sha(ROOT/saved['ruta'])==saved['sha256']
        models.setdefault(item['modelo'],[]).append(saved['estado_canonico_sha256'])
        expected=json.loads((ROOT/'resultados/secuencial/EXP-002/RUN-20261117T171942Z-cpu-01/tensores.json').read_text(encoding='utf8'))
        for split,v in json.loads((run/'tensores.json').read_text(encoding='utf8')).items():
            assert v['images_sha256']==expected[split]['images_sha256'] and v['labels_sha256']==expected[split]['labels_sha256']
        for row in read_csv(run/'tiempos.csv'):
            obs.append(dict(origen='CUDA',modelo=item['modelo'],run_id=run.name,repeticion=item['repeticion'],fase=row['fase'],tipo_medicion=row['tipo_medicion'],duracion_s=float(row['duracion_s'])))
        r=pd.read_csv(run/'recursos.csv');train=r[r.fase.eq('entrenamiento_'+item['modelo'])]
        resources.append(dict(modelo=item['modelo'],repeticion=item['repeticion'],muestras_training=len(train),
            gpu_media_pct=float(train.gpu_util_pct.mean()),gpu_max_pct=float(train.gpu_util_pct.max()),
            vram_max_mib=float(train.gpu_mem_used_mib.max()),cpu_media_pct=float(train.cpu_procesos_pct.mean())))
        sources.update(run.glob('*.json'));sources.update(run.glob('*.csv'))
    # Reuse historical CPU data verbatim; do not import any training module.
    cpu_runs=[ROOT/'resultados/secuencial/EXP-002'/name for name in [
        'RUN-20261117T171942Z-cpu-01','RUN-20261117T175407Z-cpu-02',
        'RUN-20261117T182835Z-cpu-03','RUN-20261117T190250Z-cpu-04','RUN-20261117T192509Z-cpu-05']]
    gpu_b_runs=sorted((ROOT/'resultados/paralelo/EXP-003').glob('RUN-*-gpu-*'))
    assert len(cpu_runs)==5 and len(gpu_b_runs)==5
    for origin,runs,names in [('CPU',cpu_runs,['FC-A','FC-B','CNN-B']),('CUDA',gpu_b_runs,['CNN-B'])]:
        for n,run in enumerate(runs,1):
            for row in read_csv(run/'tiempos.csv'):
                for name in names:
                    if row['fase'] in ['entrenamiento_'+name,'evaluacion_'+name,'validacion_anidada_'+name]:
                        obs.append(dict(origen=origin,modelo=name,run_id=run.name,repeticion=n,fase=row['fase'],tipo_medicion=row['tipo_medicion'],duracion_s=float(row['duracion_s'])))
            sources.add(run/'tiempos.csv')
            if origin=='CPU':
                history=json.loads((run/'historiales.json').read_text(encoding='utf8'))
                for name in ['FC-A','FC-B']:
                    for i in range(15):all_epochs.append(dict(origen='CPU',run_id=run.name,modelo=name,repeticion=n,epoca=i+1,
                        train_loss=history[name]['loss'][i],train_accuracy=history[name]['accuracy'][i],
                        val_loss=history[name]['val_loss'][i],val_accuracy=history[name]['val_accuracy'][i]))
                sources.add(run/'historiales.json');sources.add(run/'metricas.json')
    frame=pd.DataFrame(obs);frame.to_csv(OUT/'observaciones.csv',index=False)
    pd.DataFrame(resources).to_csv(OUT/'recursos-resumen.csv',index=False)
    pd.DataFrame(all_epochs).to_csv(OUT/'epocas-comparacion.csv',index=False)
    comparison=[];stats=[];quality=[]
    for name in ['CNN-B','FC-A','FC-B']:
        for phase in ['entrenamiento_'+name,'evaluacion_'+name]:
            values={o:frame[(frame.origen==o)&(frame.fase==phase)].duracion_s.to_list() for o in ['CPU','CUDA']}
            assert all(len(v)==5 for v in values.values()),(phase,values)
            d={o:desc(v) for o,v in values.items()}
            for o in d:stats.append(dict(modelo=name,fase=phase,origen=o,**d[o]))
            speed=d['CPU']['media_s']/d['CUDA']['media_s']
            comparison.append(dict(modelo=name,fase=phase,cpu_media_s=d['CPU']['media_s'],cpu_de_s=d['CPU']['de_s'],
                cuda_media_s=d['CUDA']['media_s'],cuda_de_s=d['CUDA']['de_s'],speedup=speed,reduccion_pct=(1-1/speed)*100))
        if name!='CNN-B':
            ca=[json.loads((run/'metricas.json').read_text(encoding='utf8'))['modelos'][name] for run in cpu_runs]
            ga=[json.loads(((ROOT/item.get('ruta',(BASE/item['run_id']).relative_to(ROOT).as_posix()))/'metricas.json').read_text(encoding='utf8'))['test'] for item in config['runs'] if item['modelo']==name]
            for origin,vals in [('CPU',ca),('CUDA',ga)]:quality.append(dict(modelo=name,origen=origin,n=5,
                accuracy_media_pct=float(np.mean([v['accuracy'] for v in vals])*100),accuracy_min_pct=min(v['accuracy'] for v in vals)*100,
                accuracy_max_pct=max(v['accuracy'] for v in vals)*100,loss_media=float(np.mean([v['loss'] for v in vals]))))
    write_csv(OUT/'comparacion.csv',comparison);write_csv(OUT/'estadisticas.csv',stats);write_csv(OUT/'calidad.csv',quality)
    write_json(OUT/'resumen.json',dict(creado_utc=datetime.now(timezone.utc).isoformat(),comparacion=comparison,
        estadisticas=stats,calidad=quality,recursos=resources,estados_repetibles={k:len(set(v))==1 for k,v in models.items()}))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
    def save(fig,name):
        fig.savefig(OUT/(name+'.png'),dpi=180);fig.savefig(OUT/(name+'.svg'));plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.1),layout='constrained')
    for ax,name in zip(axes,['FC-A','FC-B']):
        row=next(r for r in comparison if r['fase']=='entrenamiento_'+name)
        ax.bar([0,1],[row['cpu_media_s'],row['cuda_media_s']],yerr=[row['cpu_de_s'],row['cuda_de_s']],
            color=['#0072B2','#D55E00'],capsize=5,width=.6)
        for i,o in enumerate(['CPU','CUDA']):
            vals=frame[(frame.origen==o)&(frame.fase=='entrenamiento_'+name)].duracion_s.to_numpy()
            ax.scatter(np.linspace(i-.12,i+.12,5),vals,c='black',s=16,zorder=5)
        ax.set(xticks=[0,1],xticklabels=['CPU histórica','CUDA nueva'],ylabel='Entrenamiento total (s)',title=f"{name} · speedup {row['speedup']:.3f}×")
        ax.set_ylim(0,max(row['cpu_media_s']+row['cpu_de_s'],row['cuda_media_s']+row['cuda_de_s'])*1.16);ax.grid(axis='y',alpha=.2)
    fig.suptitle('15 épocas, validación y transferencias incluidas · media ± DE; puntos: cinco corridas')
    save(fig,'01_fc_tiempos')
    fit=[r for r in comparison if r['fase'].startswith('entrenamiento_')]
    fig,ax=plt.subplots(figsize=(8,3.6),layout='constrained')
    bars=ax.barh([r['modelo'] for r in fit],[r['speedup'] for r in fit],color=['#777777','#D55E00','#009E73'])
    for bar,r in zip(bars,fit):ax.text(bar.get_width()+.025,bar.get_y()+bar.get_height()/2,f"{r['speedup']:.3f}×",va='center')
    ax.axvline(1,color='black',ls='--',lw=1);ax.set(xlabel='Speedup = media CPU / media CUDA',title='Tres redes entrenadas con CUDA · CNN-B histórica; FC nuevas')
    ax.set_xlim(0,max(r['speedup'] for r in fit)*1.25);save(fig,'02_tres_redes_speedup')
    breakdown=[]
    for name in ['FC-A','FC-B']:
        subset=frame[(frame.origen=='CUDA')&(frame.modelo==name)]
        def avg(phase):return float(subset[subset.fase.eq(phase)].duracion_s.mean())
        wall=avg('entrenamiento_'+name);val=avg('validacion_anidada_'+name)
        h2d=sum(avg(p) for p in ['h2d_inicial','h2d_lotes'])
        d2h=sum(avg(p) for p in ['d2h_control','d2h_modelo','d2h_optimizador'])
        math=sum(avg(p) for p in ['forward_loss','backward','adam_metricas'])
        residual=wall-val-h2d-d2h-math
        assert residual>=0,(name,residual)
        breakdown.append(dict(modelo=name,pared_s=wall,validacion_cpu_s=val,h2d_eventos_s=h2d,d2h_eventos_s=d2h,
            matematica_intervalos_stream_s=math,resto_host_s=residual))
    write_csv(OUT/'desglose.csv',breakdown)
    fig,ax=plt.subplots(figsize=(10,3.8),layout='constrained');left=np.zeros(2)
    for key,label,color in [('matematica_intervalos_stream_s','Intervalos matemática GPU','#009E73'),('h2d_eventos_s','H2D','#56B4E9'),
        ('d2h_eventos_s','D2H','#CC79A7'),('validacion_cpu_s','Validación CPU','#0072B2'),('resto_host_s','Resto host','#E69F00')]:
        vals=np.array([r[key] for r in breakdown]);ax.barh(['FC-A','FC-B'],vals,left=left,label=label,color=color);left+=vals
    ax.set(xlabel='Tiempo medio (s)',title='Desglose anidado del tiempo de entrenamiento FC');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.2),ncol=3)
    save(fig,'03_fc_desglose')
    e=pd.DataFrame(all_epochs);fig,axes=plt.subplots(2,2,figsize=(10,6.5),layout='constrained')
    for i,name in enumerate(['FC-A','FC-B']):
        for j,(key,ylabel) in enumerate([('val_accuracy','Exactitud validación (%)'),('val_loss','Pérdida validación')]):
            ax=axes[i,j]
            for o,c in [('CPU','#0072B2'),('CUDA','#D55E00')]:
                s=e[(e.modelo==name)&(e.origen==o)].copy();s[key]=pd.to_numeric(s[key])*(100 if key=='val_accuracy' else 1)
                for _,g in s.groupby('run_id'):ax.plot(pd.to_numeric(g.epoca),g[key],color=c,alpha=.17,lw=.9)
                m=s.groupby(pd.to_numeric(s.epoca))[key].mean();ax.plot(m.index,m.values,color=c,label=o,lw=2)
            ax.set(xlabel='Época',ylabel=ylabel,title=name);ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Evolución real: media y cinco trayectorias por implementación; RNG diferente')
    save(fig,'04_fc_calidad')
    sources.update([ROOT/'configuraciones/EXP-012.json',BASE/'FC_CUDA.ejecutado.ipynb'])
    write_csv(OUT/'fuentes.csv',[dict(ruta_relativa=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(sources)])
    fit_cpu=sum(r['cpu_media_s'] for r in fit);fit_gpu=sum(r['cuda_media_s'] for r in fit)
    text=f'''# Ampliación FC-A/FC-B: implementación y resultados reales

EXP-012, DEC-027, CAM-012. Cinco RUN nuevos por red; 150 épocas y14,400 actualizaciones nuevas en total. No se repitió entrenamiento CPU ni CNN-B. Comparación histórica contra cinco RUN CPU EXP-002; CNN-B se reutiliza de EXP-003. Responsable de ejecución: Codex por solicitud del usuario.

## Procedimiento aplicado y equivalencia

PCUDA existente, GTX1650sm75 de4GiB, CUDA13.1, Numba-CUDA/cuBLAS SGEMM FP32 PEDANTIC; sin TF32 ni mixed precision. Preparación de imágenes en CPU con6workers. Mismo manifiesto17,400imágenes,12180/2610/2610, RGB64×64 /255. Tensores verificados SHA-256 contra CPU en cada RUN; inicializaciones FC completas restauradas del primer RUN CPU histórico (incluyendo Adam). FC-A [128,64],Dropout0.3,semilla44; FC-B [512,256,128],Dropout0.4,semilla45; BN de entrada12,288features,momentum0.99,epsilon0.001.

Warmup por RUN: lotes128+20GPU,128validaciónCPU y32testCPU; restauración completa verificada. Quince épocas fijas,96lotes/época,1440actualizaciones,último lote20,validaciónCPU128cadaépoca ytestCPU32. Tiempo principal pared sincronizada con H2D/D2H,staging,máscaras yvalidación incluidos; compilación,inicialización,gráficas yguardado fuera. RNG PCG64 reproducible en GPU; no son las mismas máscaras ni orden interno CPU. Se reutiliza la referencia histórica por instrucción del usuario. No se afirma comparación alternada/pareada ni identidad de trayectoria de aprendizaje.

## Tiempos y speedup por fase

Speedup = media CPU / media CUDA. Reducción = (1−mediaCUDA/mediaCPU)×100. Evaluación final sigue en CPU: su razón no mide inferencia GPU.

{table(comparison)}

Las tres fases de entrenamiento seleccionadas suman {fit_cpu:.6f}sCPU y{fit_gpu:.6f}sCUDA: cociente agregado {fit_cpu/fit_gpu:.6f}×. Es suma de medias de fases compatibles medidas en campañas distintas, **no** una ejecución continua del pipeline completo. No sumar speedups individuales ni extrapolar al total adaptativo original.

![Tiempos FC](01_fc_tiempos.png)

![Tres redes](02_tres_redes_speedup.png)

## Dispersión de las mediciones

{table(stats)}

DE muestral; IC95 nominal Student t con4grados de libertad. Repeticiones deterministas en el mismo equipo no muestrean todas las condiciones de otro hardware o sesión.

## Desglose y causas

{table(breakdown)}

Los eventos son intervalos de un único stream y pueden contener huecos de envío del host; no prueban tiempo activo exclusivo por kernel. Validación CPU y transferencias consumen tiempo aunque las matrices entren en VRAM. Redes densas con lotes pequeños tienen menos trabajo por lanzamiento que una convolución, y su aceleración puede quedar limitada por lanzamientos, sincronización por lote y copia de parámetros grandes. Esta explicación es una hipótesis compatible con los tiempos por fase; no se midió ocupación SM ni ancho de banda efectivo con profiler.

![Desglose](03_fc_desglose.png)

## Calidad de los modelos

{table(quality)}

![Curvas](04_fc_calidad.png)

Calidad y rendimiento son métricas distintas. Mantener arquitectura y número de épocas no garantiza exactitud idéntica con máscaras, orden de lotes y redondeo diferentes. Las cinco repeticiones por red GPU reinician el mismo estado/RNG: su variación mide principalmente tiempo, no incertidumbre estadística del aprendizaje.

## Evidencia de hardware

{table(resources)}

La captura original por red está en el primer RUN, junto con JSON que identifica muestra UTC y SHA-256. Es una captura real de ventana monitor en vivo alimentada por psutil/nvidia-smi; no es una gráfica presentada como captura. Datos por núcleo CPU y actividad/VRAMGPU de todos los RUN están en recursos.csv ymonitor-gpu.log. La toma de pantalla se realizó en primera repetición y su coste no se descontó; las otras cuatro conservan el muestreo sin ventana. No se afirma saturación total de VRAM ni ocupación por SM.

## Reproducción

La plantilla congelada FC_CUDA.plantilla.ipynb y el ejecutado FC_CUDA.ejecutado.ipynb contienen el código nativo por fases. Ejecutar de nuevo requiere otra identidad EXP/RUN y no sobrescribir esta campaña. scripts/resumir_ampliacion_fc.py solo lee resultados y genera tablas/gráficas; no entrena. CSVfuente,PNG/SVG y SHA-256 en fuentes.csv. Modelos grandes en data/EXP-012, referenciados por modelo-guardado.json. No se genera PDF.
'''
    (OUT/'resultados.md').write_text(text,encoding='utf8')
    print(table(comparison));print('RESULTADOS',OUT)

if __name__=='__main__':main()
