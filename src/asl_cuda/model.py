"""Complete CNN-B training on explicitly launched CUDA Python kernels.

TensorFlow is used only as a CPU oracle and for CPU validation/checkpoint I/O.
The original network and its initial state are imported without editing them.
"""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from .runtime import cuda, device_info
from . import kernels as k

ROOT = Path(__file__).resolve().parents[2]
INITIAL = ROOT / "data/EXP-002/RUN-20261117T171942Z-cpu-01/CNN-B.estado-inicial.npz"
INITIAL_SHA = "10558d723213752b759626c3b196665f2547a0b65cbce296e47b3a4f992305fd"


def cpu_reference():
    """Build original definitions, confirm CPU placement and restore ALL 80 arrays."""
    import os
    for name, value in {"TF_NUM_INTRAOP_THREADS":"6", "TF_NUM_INTEROP_THREADS":"1",
                        "OMP_NUM_THREADS":"6", "OPENBLAS_NUM_THREADS":"6",
                        "MKL_NUM_THREADS":"6", "TF_DETERMINISTIC_OPS":"1",
                        "TF_ENABLE_ONEDNN_OPTS":"1"}.items():
        os.environ[name] = value
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(6)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(43)
    tf.keras.mixed_precision.set_global_policy("float32")
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from benchmark_cpu import original_namespace
    ns = original_namespace()
    ns.update(keras=tf.keras, layers=tf.keras.layers, NUM_CLASSES=29)
    model = ns["build_cnn_b"]()
    model.jit_compile = False
    model.optimizer.build(model.trainable_variables)
    if hashlib.sha256(INITIAL.read_bytes()).hexdigest() != INITIAL_SHA:
        raise ValueError("Initial checkpoint identity changed")
    with np.load(INITIAL, allow_pickle=False) as archive:
        variables = list(model.variables) + list(model.optimizer.variables)
        if len(variables) != len(archive.files):
            raise ValueError("Initial state variable count differs")
        for index, variable in enumerate(variables):
            value = archive[f"arr_{index}"]
            if tuple(variable.shape) != value.shape or str(variable.dtype) != str(value.dtype):
                raise ValueError(f"Incompatible state variable {index}")
            variable.assign(value)
    if int(model.optimizer.iterations.numpy()) != 0:
        raise ValueError("Expected optimizer iteration zero")
    if any(np.any(v.numpy()) for v in [*model.optimizer._momentums, *model.optimizer._velocities]):
        raise ValueError("Nonzero initial Adam requires explicit state mapping")
    return tf, model


def grid(E):
    return max(1, min(4096, (int(E) + 255) // 256))


def compiled_resources():
    """Numba-CUDA 0.30 API expects signature objects, not .signatures tuples."""
    result={}
    for name,kern in vars(k).items():
        if hasattr(kern,"get_regs_per_thread") and kern.signatures:
            regs=kern.get_regs_per_thread()
            shared=kern.get_shared_mem_per_block()
            local=kern.get_local_mem_per_thread()
            result[name]=[dict(firma=str(sig),registros=value,shared_bytes=shared[sig],local_bytes=local[sig])
                          for sig,value in regs.items()]
    return result


class Network:
    def __init__(self, reference, backend="explicit"):
        self.info = device_info()
        if self.info["free_bytes"] < 1536 * 2**20:
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
                item.update(kind="bn", C=C, gamma=self.param(layer.gamma), beta=self.param(layer.beta),
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
        self.g1, self.g2 = self.alloc(128 * 64 * 64 * 32), self.alloc(128 * 64 * 64 * 32)
        self.parts = self.alloc(16 * 2**20 // 4)
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
            self.columns = self.alloc(128 * 64 * 64 * (9 * 32))
        self.bytes_allocated = sum(a.nbytes for a in self.allocations)
        if self.bytes_allocated > (1408 if backend == "cublas" else 768) * 2**20:
            raise MemoryError(f"Arena exceeds design: {self.bytes_allocated}")
        if sum(p["value"].size for p in self.params) != 2245245:
            raise ValueError("Architecture parameter count changed")
        self.events = []
        self.restore()

    def alloc(self, count, dtype=np.float32):
        array = cuda.device_array(int(count), dtype=dtype, stream=self.stream)
        self.allocations.append(array)
        return array

    def param(self, variable):
        values = variable.numpy().copy()
        p = dict(initial=values, value=self.alloc(values.size), grad=self.alloc(values.size),
                 m=self.alloc(values.size), v=self.alloc(values.size), variable=variable)
        self.params.append(p)
        return p

    def restore(self):
        for p in self.params:
            p["value"].copy_to_device(p["initial"].ravel(), stream=self.stream)
            for state in ("m", "v"):
                p[state].copy_to_device(np.zeros(p["initial"].size, np.float32), stream=self.stream)
        for op in self.ops:
            if op["kind"] == "bn":
                op["moving_mean"].copy_to_device(op["initial_mean"], stream=self.stream)
                op["moving_var"].copy_to_device(op["initial_var"], stream=self.stream)
        for a in (self.error, self.counter, self.total):
            a.copy_to_device(np.zeros(a.size, dtype=a.dtype), stream=self.stream)
        self.stream.synchronize()
        self.steps = 0
        self.events = []

    @contextmanager
    def timed(self, phase):
        start, end = cuda.event(timing=True), cuda.event(timing=True)
        start.record(self.stream)
        yield
        end.record(self.stream)
        self.events.append((phase, start, end))

    def drain_events(self):
        self.stream.synchronize()
        result = {}
        for phase, start, end in self.events:
            result[phase] = result.get(phase, 0.) + cuda.event_elapsed_time(start, end) / 1000.
        self.events.clear()
        return result

    def gemm(self, mode, a, v, bias, out, M, N, K, H=1, W=1, Ci=1, Co=1, I=1, O=1, relu=False):
        if self.backend == "cublas":
            if mode in (0, 2):
                R = M if mode == 0 else K
                width = 9 * Ci
                k.im2col[grid(R * width), 256, self.stream](a, self.columns, R, H, W, Ci)
                self.blas.gemm(self.columns, v, out, M, N, K, transpose_a=mode == 2)
            elif mode == 1:
                self.blas.gemm(a, v, self.columns, M, 9 * Ci, Co, transpose_b=True)
                k.col2im_gather[grid(M * Ci), 256, self.stream](self.columns, out, M, H, W, Ci)
            else:
                self.blas.gemm(a, v, out, M, N, K, transpose_a=mode == 5, transpose_b=mode == 4)
            if mode in (0, 3):
                k.bias_activation[grid(M * N), 256, self.stream](out, bias, M * N, N, relu)
            return
        P = (K + 2047) // 2048 if mode in (2, 5) else 1
        if mode in (2, 5) and P * M * N > self.parts.size:
            raise MemoryError("Partial gradient workspace exhausted")
        for m0 in range(0, M, 16384):
            geom = ((N + 15) // 16, (min(16384, M - m0) + 15) // 16, P)
            k.GEMM[mode][geom, (16, 16), self.stream](
                a, v, bias, out, M, N, K, H, W, Ci, Co, I, O, relu, m0)

    def reduce(self, x, dy, op, L, C, mode, out1, out2):
        P = (L + 2047) // 2048
        if P > 256 or P * C > self.p1.size:
            raise ValueError("Reduction shape exceeds frozen profile")
        mean, inv = op.get("mean", self.dummy), op.get("inv", self.dummy)
        k.reduce_channels[(P, (C + 31) // 32), (32, 8), self.stream](
            x, dy, mean, inv, self.p1, self.p2, L, C, 2 if mode == 2 else (1 if mode == 1 else 0))
        k.finish_channels[C, 256, self.stream](self.p1, self.p2, out1, out2,
            op.get("moving_mean", self.dummy), op.get("moving_var", self.dummy),
            mean, inv, P, C, L, mode)

    def upload(self, x, labels, epoch, batch_index):
        b = len(x)
        np.copyto(self.host_x[:x.size], x.ravel())
        np.copyto(self.host_y[:b], labels)
        drops = [op for op in self.ops if op["kind"] == "drop"]
        for d, op in enumerate(drops):
            E = b * op["sample_elements"]
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([43, epoch, batch_index, d + 1])))
            op["host_mask"][:E] = rng.random(E, dtype=np.float32) >= np.float32(op["p"])
        with self.timed("h2d_lotes"):
            self.input[:x.size].copy_to_device(self.host_x[:x.size], stream=self.stream)
            self.labels[:b].copy_to_device(self.host_y[:b], stream=self.stream)
            for op in drops:
                E = b * op["sample_elements"]
                op["mask"][:E].copy_to_device(op["host_mask"][:E], stream=self.stream)
        self.transfers.append(dict(batch=b, imagenes_bytes=x.nbytes, etiquetas_bytes=labels.nbytes,
                                   mascaras_bytes=sum(b * op["sample_elements"] for op in drops)))

    def forward(self, b):
        for op in self.ops:
            kind, x, y, E = op["kind"], op["input"], op["output"], b * op["sample_elements"]
            if kind == "conv":
                H, W, Ci, Co = (op[z] for z in ("H", "W", "Ci", "Co"))
                self.gemm(0, x, op["weights"]["value"], op["bias"]["value"], y,
                          b * H * W, Co, 9 * Ci, H, W, Ci, Co, relu=True)
            elif kind == "dense":
                self.gemm(3, x, op["weights"]["value"], op["bias"]["value"], y,
                          b, op["O"], op["I"], I=op["I"], O=op["O"], relu=op["relu"])
            elif kind == "bn":
                C, L = op["C"], E // op["C"]
                self.reduce(x, self.dummy, op, L, C, 0, op["mean"], self.dummy)
                self.reduce(x, self.dummy, op, L, C, 1, self.dummy, self.dummy)
                k.bn_forward[grid(E), 256, self.stream](x, y, op["gamma"]["value"],
                    op["beta"]["value"], op["mean"], op["inv"], E, C)
            elif kind == "pool":
                k.pool_forward[grid(E), 256, self.stream](x, y, op["arg"], b, op["H"], op["W"], op["C"])
            else:
                k.dropout[grid(E), 256, self.stream](x, op["mask"], y, np.float32(1 / (1 - op["p"])), E)
            k.finite_check[grid(E), 256, self.stream](y, E, self.error)
        k.softmax_loss[b, 32, self.stream](self.ops[-1]["output"], self.labels,
            self.probs, self.g1, self.loss, self.correct, b)
        k.finite_check[grid(b), 256, self.stream](self.loss, b, self.error)

    def backward(self, b, audit_hook=None):
        current, other = self.g1, self.g2
        for index in range(len(self.ops) - 1, -1, -1):
            op = self.ops[index]
            kind, x = op["kind"], op["input"]
            E = b * op["sample_elements"]
            if kind in ("conv", "dense"):
                if op["relu"]:
                    k.relu_backward[grid(E), 256, self.stream](current, op["output"], E)
                if kind == "conv":
                    H, W, Ci, Co = (op[z] for z in ("H", "W", "Ci", "Co"))
                    L = b * H * W
                    target = op["weights"]["grad"] if self.backend == "cublas" else self.parts
                    self.gemm(2, x, current, self.dummy, target, 9 * Ci, Co, L, H, W, Ci, Co)
                    if self.backend == "explicit":
                        k.merge_partials[grid(9 * Ci * Co), 256, self.stream](self.parts,
                            op["weights"]["grad"], 9 * Ci * Co, (L + 2047) // 2048)
                    if index:
                        self.gemm(1, current, op["weights"]["value"], self.dummy, other,
                                  L, Ci, 9 * Co, H, W, Ci, Co)
                    C = Co
                else:
                    I, O, L, C = op["I"], op["O"], b, op["O"]
                    target = op["weights"]["grad"] if self.backend == "cublas" else self.parts
                    self.gemm(5, x, current, self.dummy, target, I, O, b, I=I, O=O)
                    if self.backend == "explicit":
                        k.merge_partials[grid(I * O), 256, self.stream](self.parts, op["weights"]["grad"], I * O, 1)
                    self.gemm(4, current, op["weights"]["value"], self.dummy, other, b, I, O, I=I, O=O)
                self.reduce(current, self.dummy, {}, L, C, 3, op["bias"]["grad"], self.dummy)
            elif kind == "bn":
                C, L = op["C"], E // op["C"]
                self.reduce(x, current, op, L, C, 2, op["beta"]["grad"], op["gamma"]["grad"])
                k.bn_backward[grid(E), 256, self.stream](x, current, other, op["gamma"]["value"],
                    op["mean"], op["inv"], op["beta"]["grad"], op["gamma"]["grad"], L, C)
            elif kind == "pool":
                E_in = b * op["H"] * op["W"] * op["C"]
                k.pool_backward[grid(E_in), 256, self.stream](current, op["arg"], other,
                    b, op["H"], op["W"], op["C"])
            else:
                k.dropout[grid(E), 256, self.stream](current, op["mask"], other,
                    np.float32(1 / (1 - op["p"])), E)
            if audit_hook is not None:
                self.stream.synchronize()
                audit_hook(index, op, current, other)
            current, other = other, current

    def update(self, b):
        t = np.float32(self.steps + 1)
        alpha = np.float32(.001) * np.sqrt(np.float32(1) - np.float32(.999)**t) / (
            np.float32(1) - np.float32(.9)**t)
        for p in self.params:
            k.adam[grid(p["value"].size), 256, self.stream](p["value"], p["grad"],
                p["m"], p["v"], np.float32(alpha), p["value"].size, self.error)
        k.metrics[1, 256, self.stream](self.loss, self.correct, b, self.total)
        k.advance_step[1, 1, self.stream](self.counter)
        self.steps += 1

    def train_step(self, x, labels, epoch=0, batch_index=0):
        self.upload(x, labels, epoch, batch_index)
        with self.timed("forward_loss"):
            self.forward(len(x))
        with self.timed("backward"):
            self.backward(len(x))
        with self.timed("adam_metricas"):
            self.update(len(x))
        with self.timed("d2h_control"):
            self.error.copy_to_host(self.host_error, stream=self.stream)
        times = self.drain_events()
        if self.host_error[0]:
            raise FloatingPointError(f"CUDA sticky error {self.host_error[0]}, step {self.steps}")
        return times

    def reset_metrics(self):
        self.total.copy_to_device(np.zeros(3, np.float32), stream=self.stream)

    def export(self, optimizer=False):
        """D2H model state, then install CPU mirror outside the transfer interval."""
        host = []
        with self.timed("d2h_modelo"):
            for p in self.params:
                host.append((p["variable"], p["value"].copy_to_host(stream=self.stream).reshape(p["initial"].shape)))
            for op in self.ops:
                if op["kind"] == "bn":
                    host.append((op["layer"].moving_mean, op["moving_mean"].copy_to_host(stream=self.stream)))
                    host.append((op["layer"].moving_variance, op["moving_var"].copy_to_host(stream=self.stream)))
            self.total.copy_to_host(self.host_total, stream=self.stream)
        times = self.drain_events()
        for variable, value in host:
            variable.assign(value)
        if optimizer:
            with self.timed("d2h_optimizador"):
                states = [(p["m"].copy_to_host(stream=self.stream), p["v"].copy_to_host(stream=self.stream)) for p in self.params]
                count = self.counter.copy_to_host(stream=self.stream)
            extra = self.drain_events()
            times.update(extra)
            if int(count[0]) != self.steps:
                raise ValueError("Device update counter differs")
            self.reference.optimizer.iterations.assign(int(count[0]))
            for i, (moment, velocity) in enumerate(states):
                self.reference.optimizer._momentums[i].assign(moment.reshape(self.params[i]["initial"].shape))
                self.reference.optimizer._velocities[i].assign(velocity.reshape(self.params[i]["initial"].shape))
        return times, self.host_total.copy()
