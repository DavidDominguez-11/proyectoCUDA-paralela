"""FC-A/FC-B CUDA extension. Frozen CNN-B code is reused, never edited.

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
    def __init__(self, reference, seed):
        backend = "cublas"
        self.seed = int(seed)
        self.info = device_info()
        if self.info["free_bytes"] < 512 * 2**20:
            raise MemoryError("Insufficient free VRAM for the frozen profile")
        self.stream = cuda.stream()
        if backend not in ("explicit", "cublas"):
            raise ValueError("Unknown backend")
        self.backend = backend
        self.allocations, self.params, self.ops, self.transfers = [], [], [], []
        self.reference = reference
        self.input = self.alloc(128 * 64 * 64 * 3)
        self.host_x = cuda.pinned_array(128 * 64 * 64 * 3, dtype=np.float32)
        self.labels = self.alloc(128, np.int64)
        self.host_y = cuda.pinned_array(128, dtype=np.int64)
        shape, previous = (64, 64, 3), self.input
        for layer in reference.layers:
            name = type(layer).__name__
            item = dict(input=previous, shape=shape, layer=layer)
            if name == "Conv2D":
                H, W, Ci = shape
                Co = layer.filters
                item.update(kind="conv", H=H, W=W, Ci=Ci, Co=Co,
                            weights=self.param(layer.kernel), bias=self.param(layer.bias), relu=True)
                shape = (H, W, Co)
            elif name == "Dense":
                I, O = int(layer.kernel.shape[0]), int(layer.kernel.shape[1])
                item.update(kind="dense", I=I, O=O, relu=O != 29,
                            weights=self.param(layer.kernel), bias=self.param(layer.bias))
                shape = (O,)
            elif name == "BatchNormalization":
                C = shape[-1]
                item.update(kind="bn", C=C, momentum=float(layer.momentum), gamma=self.param(layer.gamma), beta=self.param(layer.beta),
                            moving_mean=self.alloc(C), moving_var=self.alloc(C),
                            initial_mean=layer.moving_mean.numpy().copy(), initial_var=layer.moving_variance.numpy().copy(),
                            mean=self.alloc(C), inv=self.alloc(C))
            elif name == "MaxPooling2D":
                H, W, C = shape
                item.update(kind="pool", H=H, W=W, C=C)
                shape = (H // 2, W // 2, C)
                item["arg"] = self.alloc(128 * int(np.prod(shape)), np.uint8)
            elif name == "Dropout":
                E = 128 * int(np.prod(shape))
                item.update(kind="drop", p=layer.rate, mask=self.alloc(E, np.uint8),
                            host_mask=cuda.pinned_array(E, dtype=np.uint8))
            elif name == "Flatten":
                shape = (int(np.prod(shape)),)
                continue
            elif name == "InputLayer":
                continue
            else:
                raise TypeError(f"Unsupported original layer {name}")
            item["sample_elements"] = int(np.prod(shape))
            item["output"] = self.alloc(128 * item["sample_elements"])
            self.ops.append(item)
            previous = item["output"]
        self.probs = self.alloc(128 * 29)
        self.loss, self.correct = self.alloc(128), self.alloc(128, np.uint8)
        self.g1, self.g2 = self.alloc(max(128 * 12288, max(128 * op["sample_elements"] for op in self.ops))), self.alloc(max(128 * 12288, max(128 * op["sample_elements"] for op in self.ops)))
        self.parts = self.alloc(256)
        self.p1, self.p2 = self.alloc(65536), self.alloc(65536)
        self.dummy = self.alloc(256)
        self.total = self.alloc(3)
        self.error = self.alloc(1, np.int32)
        self.counter = self.alloc(1, np.int64)
        self.host_error = cuda.pinned_array(1, dtype=np.int32)
        self.host_total = cuda.pinned_array(3, dtype=np.float32)
        if backend == "cublas":
            from .blas import Blas
            self.blas = Blas(self.stream)
            self.columns = None  # FC has no convolutions or im2col workspace
        self.bytes_allocated = sum(a.nbytes for a in self.allocations)
        if self.bytes_allocated > 384 * 2**20:
            raise MemoryError(f"Arena exceeds design: {self.bytes_allocated}")
        if sum(p["value"].size for p in self.params) != sum(int(np.prod(v.shape)) for v in reference.trainable_variables):
            raise ValueError("Architecture parameter count changed")
        if any(op["kind"] not in ("dense", "bn", "drop") for op in self.ops):
            raise TypeError("FC profile only")
        self.events = []
        self.restore()


    def upload(self, x, labels, epoch, batch_index):
        b = len(x)
        np.copyto(self.host_x[:x.size], x.ravel())
        np.copyto(self.host_y[:b], labels)
        drops = [op for op in self.ops if op["kind"] == "drop"]
        for d, op in enumerate(drops):
            E = b * op["sample_elements"]
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([self.seed, epoch, batch_index, d + 1])))
            op["host_mask"][:E] = rng.random(E, dtype=np.float32) >= np.float32(op["p"])
        with self.timed("h2d_lotes"):
            self.input[:x.size].copy_to_device(self.host_x[:x.size], stream=self.stream)
            self.labels[:b].copy_to_device(self.host_y[:b], stream=self.stream)
            for op in drops:
                E = b * op["sample_elements"]
                op["mask"][:E].copy_to_device(op["host_mask"][:E], stream=self.stream)
        self.transfers.append(dict(batch=b, imagenes_bytes=x.nbytes, etiquetas_bytes=labels.nbytes,
                                   mascaras_bytes=sum(b * op["sample_elements"] for op in drops)))

    def reduce(self,x,dy,op,L,C,mode,out1,out2):
        P=(L+2047)//2048
        if P*C>self.p1.size:raise ValueError('FC reduction workspace')
        mean,inv=op.get('mean',self.dummy),op.get('inv',self.dummy)
        k.reduce_channels[(P,(C+31)//32),(32,8),self.stream](
            x,dy,mean,inv,self.p1,self.p2,L,C,2 if mode==2 else (1 if mode==1 else 0))
        finish_fc[grid(C),256,self.stream](self.p1,self.p2,out1,out2,
            op.get('moving_mean',self.dummy),op.get('moving_var',self.dummy),
            mean,inv,P,C,L,mode,np.float32(op.get('momentum',.99)))
