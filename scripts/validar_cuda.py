"""Small correctness checks against TensorFlow CPU. Never a timing benchmark."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from asl_cuda.model import Network, cpu_reference, grid
from asl_cuda.runtime import cuda
from asl_cuda import kernels as k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destino", type=Path, required=True)
    parser.add_argument("--rendimiento-lote", action="store_true")
    parser.add_argument("--diagnostico", action="store_true", help="Collect all discrepancies; still fails overall")
    parser.add_argument("--backend", choices=("explicit", "cublas"), default="explicit")
    args = parser.parse_args()
    args.destino.mkdir(parents=True, exist_ok=False)
    report = dict(tipo="validacion_cuda_pequena_no_benchmark", inicio=datetime.now(timezone.utc).isoformat(), pruebas=[])

    def check(name, actual, expected, atol=1e-5, rtol=1e-4, l2max=1e-4):
        actual, expected = np.asarray(actual), np.asarray(expected)
        delta = actual.astype(np.float64) - expected.astype(np.float64)
        relative = float(np.linalg.norm(delta.ravel()) / max(np.linalg.norm(expected.astype(np.float64).ravel()), 1e-12))
        ok = bool(np.isfinite(actual).all() and np.allclose(actual, expected, atol=atol, rtol=rtol) and relative <= l2max)
        report["pruebas"].append(dict(nombre=name, shape=list(actual.shape), atol=atol, rtol=rtol,
            max_abs=float(np.max(np.abs(delta))), relativo_l2=relative, aprobado=ok))
        print(name, "OK" if ok else "FAIL", "max_abs", float(np.max(np.abs(delta))), "l2", relative, flush=True)
        if not ok and not args.diagnostico:
            raise AssertionError(f"Failed {name}")

    try:
        tf, reference = cpu_reference()
        net = Network(reference, backend=args.backend)
        report["hardware"] = net.info
        report["buffers_bytes"] = net.bytes_allocated
        # Copy smoke executes a real compiled kernel on the selected device.
        sample = np.arange(17, dtype=np.float32)
        device, output = cuda.to_device(sample), cuda.device_array(17, np.float32)
        k.copy_flat[1, 32](device, output, 17)
        check("smoke_copia_17", output.copy_to_host(), sample, 0, 0, 0)
        # Pool ties and negative maxima: every lane writes its own dX (including zero).
        px = np.array([[-3, -3, -8, -7], [-4, -5, -6, -9], [0, 2, 3, 1], [2, 1, 3, 2]], np.float32).reshape(1, 4, 4, 1)
        pxd = cuda.to_device(px.ravel())
        py, pi, pdx = cuda.device_array(4, np.float32), cuda.device_array(4, np.uint8), cuda.device_array(16, np.float32)
        k.pool_forward[1, 32](pxd, py, pi, 1, 4, 4, 1)
        k.pool_backward[1, 32](cuda.to_device(np.ones(4, np.float32)), pi, pdx, 1, 4, 4, 1)
        with tf.GradientTape() as tape:
            p = tf.Variable(px)
            yp = tf.nn.max_pool2d(p, 2, 2, "VALID")
            lp = tf.reduce_sum(yp)
        check("pool_negativos_empates", py.copy_to_host().reshape(yp.shape), yp.numpy(), 0, 0, 0)
        check("pool_backward_empates", pdx.copy_to_host().reshape(px.shape), tape.gradient(lp, p).numpy(), 0, 0, 0)

        rng = np.random.default_rng(239)
        for b in (2, 20):
            net.restore()
            x = rng.random((b, 64, 64, 3), dtype=np.float32)
            labels = np.arange(b, dtype=np.int64) % 29
            net.upload(x, labels, 0, 0)
            net.stream.synchronize()
            with tf.GradientTape() as tape:
                a = tf.convert_to_tensor(x)
                expected_outputs = []
                means = []
                for op in net.ops:
                    layer, kind = op["layer"], op["kind"]
                    if kind == "conv":
                        a = tf.nn.relu(tf.nn.conv2d(a, layer.kernel, 1, "SAME") + layer.bias)
                    elif kind == "dense":
                        a = tf.reshape(a, (b, op["I"]))
                        a = tf.matmul(a, layer.kernel) + layer.bias
                        if op["relu"]:
                            a = tf.nn.relu(a)
                    elif kind == "bn":
                        mean, var = tf.nn.moments(a, list(range(len(a.shape) - 1)))
                        means.append((op, mean.numpy(), var.numpy()))
                        a = tf.nn.batch_normalization(a, mean, var, layer.beta, layer.gamma, .001)
                    elif kind == "pool":
                        a = tf.nn.max_pool2d(a, 2, 2, "VALID")
                    else:
                        mask = op["host_mask"][:b * op["sample_elements"]].reshape(a.shape)
                        a = a * tf.constant(mask, tf.float32) * np.float32(1 / (1 - op["p"]))
                    expected_outputs.append(a)
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=a)
                loss = tf.reduce_mean(losses)
            gradients = tape.gradient(loss, reference.trainable_variables)
            net.forward(b)
            net.backward(b)
            net.stream.synchronize()
            for i, (op, expected) in enumerate(zip(net.ops, expected_outputs)):
                actual = op["output"][:b * op["sample_elements"]].copy_to_host().reshape(expected.shape)
                if op["kind"] == "pool":
                    H,W,C=op["H"],op["W"],op["C"]
                    def windows(value):
                        return value.reshape(b,H//2,2,W//2,2,C).transpose(0,1,3,5,2,4).reshape(-1,4)
                    cpu_windows=windows(expected_outputs[i-1].numpy())
                    gpu_windows=windows(op["input"][:b*H*W*C].copy_to_host().reshape(b,H,W,C))
                    cpu_arg=cpu_windows.argmax(axis=1)
                    gpu_arg=op["arg"][:b*op["sample_elements"]].copy_to_host().astype(np.int64)
                    different=cpu_arg!=gpu_arg
                    gaps=cpu_windows.max(axis=1)-cpu_windows[np.arange(len(cpu_arg)),gpu_arg]
                    bounds=2*np.max(np.abs(cpu_windows-gpu_windows),axis=1)+1e-7
                    route=dict(batch=b,operacion=i,ventanas=len(cpu_arg),rutas_distintas=int(different.sum()),
                        todas_dentro_cota_redondeo=bool(np.all(gaps[different]<=bounds[different])),
                        max_gap=float(gaps[different].max()) if different.any() else 0.)
                    report.setdefault("rutas_pool",[]).append(route)
                    print("RUTAS_POOL",route,flush=True)
                check(f"b{b}_forward_{i}_{op['kind']}", actual, expected.numpy(), 5e-5, 5e-4, 5e-4)
            check(f"b{b}_loss", net.loss[:b].copy_to_host(), losses.numpy(), 5e-4, 2e-3, 2e-3)
            for i, (p, grad) in enumerate(zip(net.params, gradients)):
                if grad is None or tuple(p["initial"].shape) != tuple(grad.shape):
                    raise ValueError("Reference parameter/gradient mapping differs")
                check(f"b{b}_grad_{i}", p["grad"].copy_to_host().reshape(grad.shape), grad.numpy(), 1e-4, 1e-3, 1e-3)
            for i, (op, mean, var) in enumerate(means):
                check(f"b{b}_moving_mean_{i}", op["moving_mean"].copy_to_host(),
                      np.float32(.9) * op["initial_mean"] + np.float32(.1) * mean, 5e-5, 5e-4, 5e-4)
                check(f"b{b}_moving_var_{i}", op["moving_var"].copy_to_host(),
                      np.float32(.9) * op["initial_var"] + np.float32(.1) * var, 5e-5, 5e-4, 5e-4)
            # Adam oracle receives IDENTICAL gradients; near-zero BN bias gradients
            # cannot amplify unrelated reduction rounding into a false optimizer error.
            gpu_gradients = [p["grad"].copy_to_host() for p in net.params]
            net.update(b)
            net.stream.synchronize()
            alpha = np.float32(.001) * np.sqrt(np.float32(1) - np.float32(.999)) / (np.float32(1) - np.float32(.9))
            for i, (p, grad) in enumerate(zip(net.params, gpu_gradients)):
                moment, velocity = grad * np.float32(.1), grad * grad * np.float32(.001)
                expected = p["initial"].ravel() - moment * alpha / (np.sqrt(velocity) + np.float32(1e-7))
                check(f"b{b}_adam_{i}", p["value"].copy_to_host(), expected, 1e-7, 1e-5, 1e-5)
            if int(net.counter.copy_to_host()[0]) != 1:
                raise AssertionError("Adam device counter")

        if args.rendimiento_lote:
            # Admission estimate only; never mixed with full repeated timing observations.
            x = rng.random((128, 64, 64, 3), dtype=np.float32)
            y = np.arange(128, dtype=np.int64) % 29
            net.restore()
            net.train_step(x, y)
            net.restore()
            start = time.perf_counter()
            report["estimacion_lote_128_eventos"] = net.train_step(x, y)
            report["estimacion_lote_128_pared_s"] = time.perf_counter() - start
            print("ESTIMACION 128", report["estimacion_lote_128_pared_s"], report["estimacion_lote_128_eventos"], flush=True)
        if not all(row["aprobado"] for row in report["pruebas"]):
            raise AssertionError("One or more diagnostic checks failed; not accepted")
        report["estado"] = "aprobado"
    except BaseException:
        report["estado"] = "fallido"
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["fin"] = datetime.now(timezone.utc).isoformat()
        sources = [ROOT / "src/asl_cuda/kernels.py", ROOT / "src/asl_cuda/model.py", Path(__file__)]
        report["fuentes"] = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        (args.destino / "validacion.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8")


if __name__ == "__main__":
    main()
