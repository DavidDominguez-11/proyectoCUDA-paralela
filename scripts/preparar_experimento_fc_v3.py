"""Freeze EXP-012 and native FC-only notebook. No training in this script."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,sys,importlib.metadata
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'resultados/paralelo/EXP-012'

BOOT='''from pathlib import Path
import os, sys, json, hashlib, time, csv, gc, subprocess
ROOT=Path.cwd()
while not (ROOT/'docs/Proyecto 3 CUDA.pdf').is_file():
    if ROOT==ROOT.parent: raise FileNotFoundError('Raíz del proyecto')
    ROOT=ROOT.parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
for key,value in {'OMP_NUM_THREADS':'6','MKL_NUM_THREADS':'6','OPENBLAS_NUM_THREADS':'6',
    'TF_NUM_INTRAOP_THREADS':'6','TF_NUM_INTEROP_THREADS':'1','TF_DETERMINISTIC_OPS':'1',
    'TF_ENABLE_ONEDNN_OPTS':'1','NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS':'0'}.items():os.environ[key]=value
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display,Image
from threadpoolctl import threadpool_limits
from benchmark_cpu import sha,write_json,write_csv,utc
from benchmark_cuda import load_manifest,load_images
from registro_notebook import iniciar_monitor,mostrar_figura
CONFIG=ROOT/'configuraciones/EXP-012.json'
config=json.loads(CONFIG.read_text(encoding='utf8'))
for relative,digest in config['codigo']['fuentes'].items():
    assert sha(ROOT/relative)==digest,relative
assert sha(ROOT/config['validacion']['ruta'])==config['validacion']['sha256']
assert json.loads((ROOT/config['validacion']['ruta']).read_text(encoding='utf8'))['estado']=='aprobado'
manifest=load_manifest(config)
data_root=Path(config['datos']['ruta'])
for split in manifest.values():
    for row in split:
        assert sha(data_root/row['ruta_relativa'])==row['sha256'],row['ruta_relativa']
limits=threadpool_limits(limits=6);limits.__enter__()
print('EXP-012 | SOLO FC-A / FC-B CUDA | CPU y CNN-B anteriores NO ejecutados')
print('Configuración SHA-256:',sha(CONFIG),'Python:',sys.executable)
print(json.dumps(config,indent=2,ensure_ascii=False))
'''

MAGIC='''from IPython.core.magic import register_cell_magic
import types
@register_cell_magic
def modulo_fc(line,cell):
    name=line.strip();path=ROOT/'src'/Path(*name.split('.')).with_suffix('.py')
    module=types.ModuleType(name);module.__file__=str(path);module.__package__=name.rpartition('.')[0]
    sys.modules[name]=module
    exec(compile(cell,str(path),'exec'),module.__dict__)
    parent,_,attr=name.rpartition('.')
    if parent in sys.modules:setattr(sys.modules[parent],attr,module)
    print('Módulo CUDA Python definido:',name,'SHA-256',sha(path))
import asl_cuda
'''

HELPERS='''from asl_cuda.fc import FCNetwork,fc_reference,finish_fc
from asl_cuda.model import compiled_resources
from asl_cuda.runtime import cuda
expected=json.loads((ROOT/'resultados/secuencial/EXP-002/RUN-20261117T171942Z-cpu-01/tensores.json').read_text(encoding='utf8'))
all_results=[dict(run_id='RUN-20261118T134323Z-fc-a-01',**json.loads((ROOT/'resultados/paralelo/EXP-011/RUN-20261118T134323Z-fc-a-01/metricas.json').read_text(encoding='utf8')))]
def state(phase,**other):
    write_json(run/'estado.json',dict(utc=utc(),fase=phase,modelo=name,**other))
def add(phase,seconds,kind='pared',**other):
    timings.append(dict(experimento_id='EXP-012',run_id=run.name,modelo=name,repeticion=repetition,
        fase=phase,tipo_medicion=kind,duracion_s=float(seconds),configuracion_sha256=sha(CONFIG),**dict({'inicio_utc':'','fin_utc':''},**other)))
def total_events(extra):
    for key,value in extra.items():events[key]=events.get(key,0.)+float(value)
'''

INIT='''run=ROOT/'resultados/paralelo/EXP-012'/run_id
run.mkdir(exist_ok=False)
timings=[];epoch_rows=[];events={};shuffle_hashes=[]
state('inicializacion')
stop,monitor=iniciar_monitor(run)
if repetition==1:
    capture_log=(run/'capturador.log').open('x',encoding='utf8')
    capture=subprocess.Popen([sys.executable,'-X','utf8',str(ROOT/'scripts/capturar_fc_real.py'),
        '--run',str(run),'--tipo','cuda','--modelo',name,'--nombre',name.lower()],
        stdout=capture_log,stderr=subprocess.STDOUT)
init_start=time.perf_counter()
tf,reference,seed,initial=fc_reference(name)
assert sha(initial)==config['estados_iniciales'][name]['sha256']
net=FCNetwork(reference,seed)
reference.summary()
print('GPU REAL',net.info,'Arena bytes',net.bytes_allocated,'BN momentum',[op['momentum'] for op in net.ops if op['kind']=='bn'])
write_json(run/'hardware.json',dict(device=net.info,tensorflow=tf.__version__,
    tf_build=tf.sysconfig.get_build_info(),cpu_tensor_device=tf.constant([1.]).device,
    tf_visible=[str(d) for d in tf.config.get_visible_devices()],cublas_version=net.blas.version,
    cublas_dll=str(net.blas.path),cublas_sha256=sha(net.blas.path),buffers_bytes=net.bytes_allocated,
    parametros=sum(p['value'].size for p in net.params),initial_checkpoint_sha256=sha(initial)))
assert not tf.config.get_visible_devices('GPU')
wx,wy=load_images(manifest['train'][:148],data_root,1)
wv,wvy=load_images(manifest['val'][:128],data_root,1)
wt,wty=load_images(manifest['test'][:32],data_root,1)
net.train_step(wx[:128],wy[:128],0,0);net.train_step(wx[128:],wy[128:],0,1)
reference.evaluate(wv,wvy,batch_size=128,verbose=0)
reference.evaluate(wt,wty,batch_size=32,verbose=0)
net.restore()
for p in net.params:
    assert np.array_equal(p['value'].copy_to_host(),p['initial'].ravel())
    assert not np.any(p['m'].copy_to_host()) and not np.any(p['v'].copy_to_host())
for op in net.ops:
    if op['kind']=='bn':
        assert np.array_equal(op['moving_mean'].copy_to_host(),op['initial_mean'])
        assert np.array_equal(op['moving_var'].copy_to_host(),op['initial_var'])
assert net.counter.copy_to_host()[0]==0
write_json(run/'warmup.json',dict(duracion_s=time.perf_counter()-init_start,train=148,val=128,test=32,
    restauracion_verificada=True,excluido_timing=True))
print('Warmup nuevo FC terminado y estado inicial restaurado; no incluido en entrenamiento.')
'''

PREP='''state('preparacion_cpu',workers=6)
start=time.perf_counter()
inputs={split:load_images(manifest[split],data_root,6) for split in ['train','val','test']}
add('preparacion_cpu',time.perf_counter()-start)
tensors={}
for split,(images,labels) in inputs.items():
    a=hashlib.sha256(memoryview(images).cast('B')).hexdigest()
    b=hashlib.sha256(memoryview(labels).cast('B')).hexdigest()
    assert a==expected[split]['images_sha256'] and b==expected[split]['labels_sha256']
    tensors[split]=dict(images_sha256=a,labels_sha256=b,shape=list(images.shape),dtype=str(images.dtype))
write_json(run/'tensores.json',tensors)
x,y=inputs['train'];vx,vy=inputs['val'];tx,ty=inputs['test']
print('Preparación SOLO CPU (6 workers); bytes idénticos a EXP-002:',tensors)
'''

TRAIN='''state('entrenamiento_'+name,epoca=1,pasos=0)
train_start_utc=utc();train_start=time.perf_counter_ns()
validations=0.;export_wall=0.
with net.timed('h2d_inicial'):net.restore()
total_events(net.drain_events())
rng=np.random.Generator(np.random.PCG64(seed))
for epoch in range(15):
    epoch_start=time.perf_counter()
    state('entrenamiento_'+name,epoca=epoch+1,pasos=net.steps)
    net.reset_metrics()
    order=rng.permutation(len(x))
    shuffle_hashes.append(hashlib.sha256(order.astype(np.int64).tobytes()).hexdigest())
    for j,begin in enumerate(range(0,len(x),128)):
        indices=order[begin:begin+128]
        total_events(net.train_step(x[indices],y[indices],epoch,j))
    export_start=time.perf_counter()
    extra,total=net.export();total_events(extra)
    export_wall+=time.perf_counter()-export_start
    assert int(total[2])==12180
    state('validacion_cpu',epoca=epoch+1,pasos=net.steps)
    val_start=time.perf_counter()
    val=reference.evaluate(vx,vy,batch_size=128,verbose=0,return_dict=True)
    val_s=time.perf_counter()-val_start;validations+=val_s
    epoch_rows.append(dict(modelo=name,repeticion=repetition,epoca=epoch+1,pasos_acumulados=net.steps,
        train_loss=float(total[0]/total[2]),train_accuracy=float(total[1]/total[2]),
        val_loss=float(val['loss']),val_accuracy=float(val['accuracy']),
        pared_epoca_s=time.perf_counter()-epoch_start,validacion_cpu_s=val_s))
    write_csv(run/'epocas.csv',epoch_rows)
    print('EPOCH',name,repetition,epoch_rows[-1],flush=True)
extra,_=net.export(optimizer=True);total_events(extra)
train_seconds=(time.perf_counter_ns()-train_start)/1e9
train_end_utc=utc()
assert net.steps==1440 and int(reference.optimizer.iterations.numpy())==1440
add('entrenamiento_'+name,train_seconds,inicio_utc=train_start_utc,fin_utc=train_end_utc)
add('validacion_anidada_'+name,validations)
for phase,seconds in events.items():add(phase,seconds,'evento_device')
print('ENTRENAMIENTO CUDA REAL',name,'corrida',repetition,'segundos',train_seconds,
    '| incluye staging, máscaras, H2D, GPU, D2H y validación CPU; pasos',net.steps)
'''

FINISH='''state('evaluacion_final_cpu',pasos=net.steps)
start=time.perf_counter_ns()
test=reference.evaluate(tx,ty,batch_size=32,verbose=0,return_dict=True)
eval_seconds=(time.perf_counter_ns()-start)/1e9
add('evaluacion_'+name,eval_seconds)
metrics=dict(modelo=name,repeticion=repetition,test={k:float(v) for k,v in test.items()},epocas=15,pasos=1440,
    entrenamiento_s=train_seconds,evaluacion_cpu_s=eval_seconds,
    comparabilidad='Datos, estado inicial, arquitectura y pasos equivalentes; CPU histórica; RNG interno diferente')
write_json(run/'metricas.json',metrics);write_csv(run/'tiempos.csv',timings)
write_json(run/'eventos.json',dict(eventos_s=events,validacion_cpu_s=validations,export_cpu_pared_s=export_wall,
    nota='Intervalos stream anidados; no equivalen a tiempo activo exclusivo de kernels; no sumar al tiempo pared'))
write_json(run/'ordenes.json',dict(seed=seed,generador='PCG64',epocas_sha256=shuffle_hashes))
param_count=sum(p['value'].size for p in net.params)
bn_count=sum(op['C']*2 for op in net.ops if op['kind']=='bn')
write_json(run/'transferencias.json',dict(lotes=net.transfers[-1440:],modelo_d2h_copias=16,
    modelo_bytes_por_copia=(param_count+bn_count)*4,optimizador_final_bytes=param_count*8+8,
    control_d2h_bytes=1440*4,reinicios_metricas_h2d_bytes=15*12,
    nota='Copia final adicional de pesos incluida; sin restar costes'))
state('guardado_fuera_timing')
model_dir=ROOT/'data/EXP-012'/run.name;model_dir.mkdir(parents=True,exist_ok=False)
model_path=model_dir/(name+'-CUDA.keras');reference.save(model_path)
fingerprint=hashlib.sha256()
for variable in [*reference.variables,*reference.optimizer.variables]:
    array=variable.numpy();fingerprint.update(str(array.dtype).encode());fingerprint.update(str(array.shape).encode());fingerprint.update(array.tobytes())
write_json(run/'modelo-guardado.json',dict(ruta=model_path.relative_to(ROOT).as_posix(),sha256=sha(model_path),
    estado_canonico_sha256=fingerprint.hexdigest(),optimizer_iterations=1440))
write_json(run/'lanzamientos.json',dict(vector_block=256,bn_parcial_block=[32,8,1],
    bn_parcial_grid=[1,384,1],bn_final_block=256,bn_final_grid=[48,1,1],
    softmax_block=32,softmax_grid=[128,1,1],ultimo_batch=20,recursos=compiled_resources(),
    fc_finish_regs=str(finish_fc.get_regs_per_thread()),
    gemm='cuBLAS SGEMM FP32 PEDANTIC; jerarquía interna de biblioteca, no fijada por el equipo'))
state('completada',pasos=1440)
stop.set();monitor.join(timeout=15)
if repetition==1:
    try:capture.wait(timeout=8)
    except subprocess.TimeoutExpired:capture.terminate();capture.wait();print('Captura no disponible; revisar capturador.log')
    capture_log.close()
display(pd.DataFrame(timings).fillna(''))
display(pd.DataFrame(epoch_rows))
print('MÉTRICAS REALES',json.dumps(metrics,indent=2,ensure_ascii=False))
fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
for ax,keys,title in zip(axes,[('train_loss','val_loss'),('train_accuracy','val_accuracy')],['Pérdida','Exactitud']):
    for key in keys:ax.plot(range(1,16),[r[key] for r in epoch_rows],label=key)
    ax.set(xlabel='Época',ylabel=title,title=name+' · corrida '+str(repetition));ax.legend();ax.grid(alpha=.2)
fig.savefig(run/'curvas.png',dpi=160);fig.savefig(run/'curvas.svg');mostrar_figura(fig)
resource=pd.read_csv(run/'recursos.csv')
fig,axes=plt.subplots(2,1,figsize=(10,5),sharex=True,layout='constrained')
axes[0].plot(resource.transcurrido_s,resource.cpu_procesos_pct,color='#0072B2',label='CPU proceso e hijos (100% por núcleo)')
axes[1].plot(resource.transcurrido_s,resource.gpu_util_pct,color='#D55E00',label='Actividad GPU (%)')
axes[0].set(ylabel='Uso CPU (%)');axes[1].set(ylabel='Actividad GPU (%)',xlabel='Tiempo desde inicio del RUN (s)',ylim=(0,105))
for ax in axes:ax.legend(loc='upper right');ax.grid(alpha=.2)
fig.suptitle(name+' · monitorización real, corrida '+str(repetition))
fig.savefig(run/'recursos.png',dpi=160);fig.savefig(run/'recursos.svg');mostrar_figura(fig)
for picture in sorted(run.glob('captura-*.png')):display(Image(filename=str(picture)))
all_results.append(dict(run_id=run.name,**metrics));write_json(ROOT/'resultados/paralelo/EXP-012/progreso.json',all_results)
net.stream.synchronize();net.blas.close()
del net,reference,inputs,x,y,vx,vy,tx,ty,wx,wy,wv,wvy,wt,wty,p,op
tf.keras.backend.clear_session();gc.collect();cuda.current_context().deallocations.clear()
'''

def main():
    BASE.mkdir(parents=True,exist_ok=False)
    checks=ROOT/'resultados/preparacion/ampliacion-fc/validacion-01/validacion.json'
    assert json.loads(checks.read_text(encoding='utf8'))['estado']=='aprobado'
    # Add capture model filter without modifying the frozen earlier capturer.
    capture=(ROOT/'scripts/capturar_notebook_real.py').read_text(encoding='utf8')
    capture=capture.replace("ap.add_argument('--nombre')","ap.add_argument('--nombre')\n    ap.add_argument('--modelo',choices=['FC-A','FC-B'],required=True)")
    capture=capture.replace("row['fase']=='entrenamiento_CNN-B' and gpu>20","row['fase']=='entrenamiento_'+args.modelo and number('epoca')>=2 and gpu>0")
    (ROOT/'scripts/capturar_fc_real.py').write_text(capture,encoding='utf8')
    nb=nbf.v4.new_notebook()
    nb.metadata=dict(kernelspec=dict(name='pcuda',display_name='PCUDA',language='python'),language_info=dict(name='python'))
    md=lambda s:nb.cells.append(nbf.v4.new_markdown_cell(s))
    code=lambda s:nb.cells.append(nbf.v4.new_code_cell(s))
    md('# Ampliación CUDA: FC-A y FC-B\n\nEjecución nueva y nativa, exclusiva de estas dos redes. Cinco corridas por arquitectura. Las campañas CPU y CNN-B anteriores **no se ejecutan**. Preparación y validación en CPU; entrenamiento matemático en CUDA Python/cuBLAS. Datos e inicialización históricos verificados. Los contadores pertenecen a esta sesión nueva, distinta del notebook CNN-B previo.')
    code(BOOT);md('## CUDA Python: código efectivamente definido en esta sesión\n\nSe muestran las dependencias reutilizadas, sin instanciar ni entrenar CNN-B. El último módulo contiene la extensión FC y la corrección de momentum BN.');code(MAGIC)
    for module in ['runtime','kernels','blas','model','fc']:
        code('%%modulo_fc asl_cuda.'+module+'\n'+(ROOT/f'src/asl_cuda/{module}.py').read_text(encoding='utf8'))
    code(HELPERS)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    runs=[dict(modelo='FC-A',repeticion=1,run_id='RUN-20261118T134323Z-fc-a-01',ruta='resultados/paralelo/EXP-011/RUN-20261118T134323Z-fc-a-01',reutilizado_sin_reentrenar=True)]
    for name in ['FC-A','FC-B']:
        md('## '+name+' — cinco ejecuciones nuevas\n\nBatch128, último lote20, 15épocas, 1440actualizaciones. La medida pared incluye transferencias y validación CPU; guardado y gráficas quedan fuera. La primera corrida incorpora captura en vivo; su coste no se resta.')
        for repetition in range(2 if name=='FC-A' else 1,6):
            rid=f'RUN-{stamp}-{name.lower()}-{repetition:02d}'
            runs.append(dict(modelo=name,repeticion=repetition,run_id=rid))
            md(f'### {name}: corrida {repetition}\n\nRUN: `{rid}`. Inicialización/restauración → preparación CPU → entrenamiento CUDA → evaluación CPU y evidencias.')
            code(f'name={name!r};repetition={repetition};run_id={rid!r}\n'+INIT)
            code(PREP);code(TRAIN);code(FINISH)
    md('## Resumen de las diez corridas nuevas\n\nNo se mezclan las observaciones adicionales n=1 de otros notebooks con las campañas de cinco repeticiones.')
    code("display(pd.DataFrame([{k:v for k,v in row.items() if k!='test'}|{'test_accuracy':row['test']['accuracy'],'test_loss':row['test']['loss']} for row in all_results]))\nlimits.__exit__(None,None,None)\nprint('FIN EXP-012: 150 épocas nuevas FC, cero entrenamientos CPU/CNN-B previos repetidos.')")
    template=BASE/'FC_CUDA.plantilla.ipynb';nbf.write(nb,template)
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    cpu=json.loads((ROOT/'configuraciones/EXP-002.json').read_text(encoding='utf8'))
    sources=[ROOT/'scripts/preparar_experimento_fc_v3.py',ROOT/'scripts/ejecutar_fc_notebook_v3.py',ROOT/'scripts/capturar_fc_real.py',
        ROOT/'scripts/registro_notebook.py',ROOT/'scripts/benchmark_cuda.py',ROOT/'scripts/benchmark_cpu.py',
        *[ROOT/f'src/asl_cuda/{m}.py' for m in ['__init__','runtime','kernels','blas','model','fc']],template]
    cfg=dict(experimento_id='EXP-012',creado_utc=datetime.now(timezone.utc).isoformat(),responsable='Codex por instrucción del usuario',
        decisiones=['DEC-027'],cambios=['CAM-012'],codigo=dict(fuentes={p.relative_to(ROOT).as_posix():sha(p) for p in sources}),
        datos=cpu['datos'],alcance=dict(modelos=['FC-A','FC-B'],repeticiones=5,epocas=15,batch=128,ultimo_batch=20,pasos=1440,
            preparacion='CPU6workers',validacion='CPU128 cada época incluida en fit',test='CPU32 separado',augmentation=False),
        precision=dict(dtype='float32',cublas='PEDANTIC_MATH',atomics=False,tf32=False),
        aleatoriedad=dict(seeds={'FC-A':44,'FC-B':45},shuffle='PCG64 por época',dropout='Bernoulli [seed,epoch,batch,layer]',
            identidad_mascaras_cpu=False),medicion=dict(principal='pared sincronizada incluyendo staging,H2D,D2H,validaciónCPU',
            eventos='intervalos stream anidados, no suma independiente',warmup='128+20 train;128val;32test;restauración total',
            referencia_cpu='EXP-002 cinco RUN históricos, no volver a ejecutar',capturas='primera corrida por modelo, coste no descontado'),
        validacion=dict(ruta=checks.relative_to(ROOT).as_posix(),sha256=sha(checks)),
        estados_iniciales={name:dict(ruta=f'data/EXP-002/RUN-20261117T171942Z-cpu-01/{name}.estado-inicial.npz',
            sha256=sha(ROOT/f'data/EXP-002/RUN-20261117T171942Z-cpu-01/{name}.estado-inicial.npz')) for name in ['FC-A','FC-B']},
        entorno=dict(python=sys.version,paquetes={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}),runs=runs)
    (ROOT/'configuraciones/EXP-012.json').write_text(json.dumps(cfg,indent=2,ensure_ascii=False),encoding='utf8')
    (BASE/'configuracion.json').write_bytes((ROOT/'configuraciones/EXP-012.json').read_bytes())
    print('PREPARADO EXP-012',len(nb.cells),'celdas;',len(runs),'RUN nuevos')

if __name__=='__main__':main()
