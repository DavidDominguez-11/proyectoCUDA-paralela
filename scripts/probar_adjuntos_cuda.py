"""Validate every real-shape local backward with IDENTICAL activations/dY.

Full-network finite-precision max-pool decisions can differ near ties. This
separates an incorrect Jacobian from that legitimate nonsmooth path sensitivity.
No thresholds are relaxed; independent per-element operator tolerances apply.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from asl_cuda.model import Network, cpu_reference


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--destino",type=Path,required=True)
    args=ap.parse_args()
    args.destino.mkdir(parents=True,exist_ok=False)
    report=dict(tipo="adjuntos_locales_activaciones_identicas",inicio=datetime.now(timezone.utc).isoformat(),pruebas=[])
    try:
        tf,ref=cpu_reference()
        net=Network(ref,backend="cublas")
        rng=np.random.default_rng(239)
        for b in (20,128):
            net.restore()
            x=rng.random((b,64,64,3),dtype=np.float32)
            y=np.arange(b,dtype=np.int64)%29
            net.upload(x,y,0,0)
            net.forward(b)
            net.stream.synchronize()

            def check(name,a,r):
                a,r=np.asarray(a,np.float64),np.asarray(r,np.float64)
                error=np.abs(a-r)
                scaled=float(np.max(error/(1e-4+1e-3*np.abs(r))))
                row=dict(nombre=name,max_abs=float(error.max()),max_error_escalado=scaled,
                         atol=1e-4,rtol=1e-3,aprobado=bool(np.isfinite(a).all() and scaled<=1))
                report["pruebas"].append(row)
                print(name,"OK" if row["aprobado"] else "FAIL",scaled,flush=True)
                if not row["aprobado"]:
                    raise AssertionError(name)

            def audit(index,op,dy_device,dx_device):
                shape=(b,*op["shape"])
                xh=op["input"][:int(np.prod(shape))].copy_to_host().reshape(shape)
                E=b*op["sample_elements"]
                dy=dy_device[:E].copy_to_host()
                layer,kind=op["layer"],op["kind"]
                variables=[]
                with tf.GradientTape() as tape:
                    xt=tf.Variable(xh)
                    if kind=="conv":
                        out=tf.nn.conv2d(xt,layer.kernel,1,"SAME")+layer.bias
                        variables=[layer.kernel,layer.bias]
                    elif kind=="dense":
                        out=tf.matmul(tf.reshape(xt,(b,op["I"])),layer.kernel)+layer.bias
                        variables=[layer.kernel,layer.bias]
                    elif kind=="bn":
                        mu,var=tf.nn.moments(xt,list(range(len(shape)-1)))
                        out=tf.nn.batch_normalization(xt,mu,var,layer.beta,layer.gamma,.001)
                        variables=[layer.gamma,layer.beta]
                    elif kind=="pool":
                        out=tf.nn.max_pool2d(xt,2,2,"VALID")
                    else:
                        mask=op["host_mask"][:E].reshape(shape)
                        out=xt*tf.constant(mask,tf.float32)*np.float32(1/(1-op["p"]))
                    # dy for Conv/Dense already includes the exact retained ReLU
                    # derivative, matching the contract of the linear backward.
                    local_loss=tf.reduce_sum(out*tf.constant(dy.reshape(out.shape)))
                grads=tape.gradient(local_loss,[xt,*variables])
                if index:
                    check(f"b{b}_op{index}_{kind}_dx",dx_device[:xh.size].copy_to_host().reshape(shape),grads[0].numpy())
                if variables:
                    keys=("gamma","beta") if kind=="bn" else ("weights","bias")
                    for key,g in zip(keys,grads[1:]):
                        p=op[key]
                        check(f"b{b}_op{index}_{kind}_{key}",p["grad"].copy_to_host().reshape(g.shape),g.numpy())

            net.backward(b,audit_hook=audit)
        report["estado"]="aprobado"
    except BaseException:
        report["estado"]="fallido"
        report["error"]=traceback.format_exc()
        raise
    finally:
        report["fin"]=datetime.now(timezone.utc).isoformat()
        report["fuentes"]={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__),*(ROOT/"src/asl_cuda").glob("*.py")]}
        (args.destino/"validacion.json").write_text(json.dumps(report,indent=2),encoding="utf8")


if __name__=="__main__":
    main()
