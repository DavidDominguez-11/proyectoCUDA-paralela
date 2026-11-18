"""Prepare additive FC CUDA sources; does not execute any training."""
from pathlib import Path
import hashlib
import json
import shutil
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'resultados/preparacion/ampliacion-fc'

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    prior={}
    for relative in ['entregables/Informe_Proyecto_3_CUDA.docx','entregables/notebooks/ASL_CUDA.ipynb','entregables/notebooks/ASL_Secuencial.ipynb']:
        p=ROOT/relative
        prior[relative]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
        shutil.copy2(p,OUT/p.name)
    (OUT/'identidades-previas.json').write_text(json.dumps(prior,indent=2),encoding='utf8')
    # Derive a new constructor, preserving the frozen CNN-B implementation.
    original=(ROOT/'src/asl_cuda/model.py').read_text(encoding='utf8')
    ctor=original[original.index('    def __init__(self, reference,'):original.index('\n    def alloc(',original.index('class Network:'))]
    ctor=ctor.replace('def __init__(self, reference, backend="explicit"):', 'def __init__(self, reference, seed):')
    ctor=ctor.replace('        self.info = device_info()', '        backend = "cublas"\n        self.seed = int(seed)\n        self.info = device_info()')
    ctor=ctor.replace('1536 * 2**20','512 * 2**20')
    ctor=ctor.replace('C=C, gamma=', 'C=C, momentum=float(layer.momentum), gamma=')
    ctor=ctor.replace('self.alloc(128 * 64 * 64 * 32), self.alloc(128 * 64 * 64 * 32)', 'self.alloc(max(128 * 12288, max(128 * op["sample_elements"] for op in self.ops))), self.alloc(max(128 * 12288, max(128 * op["sample_elements"] for op in self.ops)))')
    ctor=ctor.replace('self.alloc(16 * 2**20 // 4)', 'self.alloc(256)')
    ctor=ctor.replace('self.columns = self.alloc(128 * 64 * 64 * (9 * 32))', 'self.columns = None  # FC has no convolutions or im2col workspace')
    ctor=ctor.replace('(1408 if backend == "cublas" else 768)', '384')
    ctor=ctor.replace('if sum(p["value"].size for p in self.params) != 2245245:', 'if sum(p["value"].size for p in self.params) != sum(int(np.prod(v.shape)) for v in reference.trainable_variables):')
    ctor=ctor.replace('        self.events = []\n        self.restore()', '        if any(op["kind"] not in ("dense", "bn", "drop") for op in self.ops):\n            raise TypeError("FC profile only")\n        self.events = []\n        self.restore()')
    upload=original[original.index('    def upload('):original.index('\n    def forward(')]
    upload=upload.replace('[43, epoch, batch_index, d + 1]', '[self.seed, epoch, batch_index, d + 1]')
    content='''"""FC-A/FC-B CUDA extension. Frozen CNN-B code is reused, never edited.

All arithmetic training executes through explicit Numba CUDA launches and
cuBLAS FP32 SGEMM. CPU is restricted to preparation, validation and checkpoint I/O.
The BN finishing reduction uses one thread per input feature (12,288 features),
with momentum taken from the original FC layer, rather than CNN-B's 0.9.
"""
import hashlib
import json
import os
import sys
import numpy as np
from pathlib import Path
from .runtime import cuda, device_info
from .model import Network, grid, ROOT
from . import kernels as k
from numba import float32


@cuda.jit(cache=True, fastmath=False)
def finish_fc(p1,p2,out1,out2,movmean,movvar,mean,inv,P,C,L,mode,momentum):
    for c in range(cuda.grid(1),C,cuda.gridsize(1)):
        a,b=float32(0),float32(0)
        for p in range(P):
            a+=p1[p*C+c]
            b+=p2[p*C+c]
        if mode==0:
            out1[c]=a/float32(L)
        elif mode==1:
            var=a/float32(L)
            inv[c]=float32(1)/(var+float32(.001))**float32(.5)
            complement=float32(1)-momentum
            movmean[c]=momentum*movmean[c]+complement*mean[c]
            movvar[c]=momentum*movvar[c]+complement*var
        elif mode==2:
            out1[c],out2[c]=a,b
        else:
            out1[c]=a


def fc_reference(name):
    """Restore historical architecture, ALL variables and zero Adam state."""
    if name not in ('FC-A','FC-B'):raise ValueError(name)
    for key,value in {'TF_NUM_INTRAOP_THREADS':'6','TF_NUM_INTEROP_THREADS':'1',
        'OMP_NUM_THREADS':'6','OPENBLAS_NUM_THREADS':'6','MKL_NUM_THREADS':'6',
        'TF_DETERMINISTIC_OPS':'1','TF_ENABLE_ONEDNN_OPTS':'1'}.items():os.environ[key]=value
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(6)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.mixed_precision.set_global_policy('float32')
    seed=44 if name=='FC-A' else 45
    tf.keras.utils.set_random_seed(seed)
    sys.path.insert(0,str(ROOT/'scripts'))
    from benchmark_cpu import original_namespace
    ns=original_namespace();ns.update(keras=tf.keras,layers=tf.keras.layers,NUM_CLASSES=29)
    model=ns['build_fc']([128,64] if name=='FC-A' else [512,256,128],
        dropout=.3 if name=='FC-A' else .4,name=name)
    model.jit_compile=False
    model.optimizer.build(model.trainable_variables)
    initial=ROOT/'data/EXP-002/RUN-20261117T171942Z-cpu-01'/f'{name}.estado-inicial.npz'
    index=json.loads((ROOT/'resultados/secuencial/EXP-002/RUN-20261117T171942Z-cpu-01/estados-iniciales.json').read_text(encoding='utf8'))
    # Both the checkpoint bytes and its source metadata are frozen by EXP-010.
    with np.load(initial,allow_pickle=False) as data:
        variables=[*model.variables,*model.optimizer.variables]
        if len(variables)!=len(data.files):raise ValueError('Initial variable count')
        for i,v in enumerate(variables):
            a=data[f'arr_{i}']
            if tuple(v.shape)!=a.shape or str(v.dtype)!=str(a.dtype):raise ValueError('Initial shape/dtype')
            v.assign(a)
    if int(model.optimizer.iterations.numpy())!=0:raise ValueError('Nonzero initial step')
    if any(np.any(v.numpy()) for v in [*model.optimizer._momentums,*model.optimizer._velocities]):raise ValueError('Nonzero Adam')
    return tf,model,seed,initial


class FCNetwork(Network):
'''+ctor+'\n\n'+upload+'''
    def reduce(self,x,dy,op,L,C,mode,out1,out2):
        P=(L+2047)//2048
        if P*C>self.p1.size:raise ValueError('FC reduction workspace')
        mean,inv=op.get('mean',self.dummy),op.get('inv',self.dummy)
        k.reduce_channels[(P,(C+31)//32),(32,8),self.stream](
            x,dy,mean,inv,self.p1,self.p2,L,C,2 if mode==2 else (1 if mode==1 else 0))
        finish_fc[grid(C),256,self.stream](self.p1,self.p2,out1,out2,
            op.get('moving_mean',self.dummy),op.get('moving_var',self.dummy),
            mean,inv,P,C,L,mode,np.float32(op.get('momentum',.99)))
'''
    (ROOT/'src/asl_cuda/fc.py').write_text(content,encoding='utf8')
    for f,title,text in [
        ('decisiones.md','DEC-027 — Ampliación CUDA a FC-A y FC-B',
         'Aceptada por el usuario. Ejecutar exclusivamente implementaciones/pruebas nuevas FC; reutilizar cinco mediciones CPU EXP-002 y CNN-B EXP-003, sin repetirlas. Misma arquitectura, estado inicial histórico, FP32, 15 épocas, batch128 (último20), validación CPU incluida y test CPU separado. Comparación histórica con RNG distinto ya reconocido. FC conserva BN momentum0.99 y dropout0.3/0.4. Código aditivo, memoria dimensionada sin workspace de convoluciones. Cinco RUN nuevos por modelo; EXP-010. Preservar diseño DOCX editado por usuario y salidas previas de notebooks.'),
        ('cambios.md','CAM-012 — Dos redes FC CUDA adicionales',
         'En preparación: módulo src/asl_cuda/fc.py, pruebas nuevas, notebook nativo adicional FC y ampliación de notebook CUDA/informe. No modificar fuentes ni evidencias CNN-B/CPU congeladas. Identidades previas y copias del DOCX editado y notebooks en resultados/preparacion/ampliacion-fc/identidades-previas.json. DEC-027; BIT-014.'),
        ('bitacora.md','BIT-014 — Implementación y evaluación de FC-A/FC-B CUDA',
         'Inicio UTC: '+datetime.now(timezone.utc).isoformat()+'. Responsable: Codex por instrucción del usuario. Revisados PDF normativo (tres páginas), matriz, README, decisiones, bitácora, procedimientos y reproducibilidad. Solo nuevas ejecuciones FC; PCUDA existente, sin instalaciones ni descargas. Se preserva DOCX editado de752593bytes y ambos notebooks mediante copia/hash. Desarrollo y resultados pendientes; EXP-010; DEC-027; CAM-012.')]:
        with (ROOT/'docs/trazabilidad'/f).open('a',encoding='utf8') as stream:stream.write('\n\n## '+title+'\n\n'+text+'\n')
    print('PREPARADO',OUT)

if __name__=='__main__':main()
