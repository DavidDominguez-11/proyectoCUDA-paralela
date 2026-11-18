"""Independent local Jacobian tests on identical inputs; no full CPU training."""
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
from asl_cuda.model import Network, cpu_reference, grid
from asl_cuda.runtime import cuda
from asl_cuda import kernels as k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--destino", required=True, type=Path)
    ap.add_argument("--backend", choices=["explicit", "cublas"], default="cublas")
    args = ap.parse_args()
    args.destino.mkdir(parents=True, exist_ok=False)
    report = dict(tipo="pruebas_operadores_mismos_datos", backend=args.backend,
                  inicio=datetime.now(timezone.utc).isoformat(), pruebas=[])

    def check(name, actual, ref, atol=1e-4, rtol=1e-3):
        a, r = np.asarray(actual, np.float64), np.asarray(ref, np.float64)
        error = np.abs(a - r)
        scaled = float(np.max(error / (atol + rtol * np.abs(r))))
        passed = bool(np.isfinite(a).all() and scaled <= 1)
        report["pruebas"].append(dict(nombre=name, max_abs=float(error.max()),
            max_error_escalado=scaled, atol=atol, rtol=rtol, aprobado=passed))
        print(name, "OK" if passed else "FAIL", "scaled", scaled, flush=True)
        if not passed:
            raise AssertionError(name)

    try:
        tf, ref = cpu_reference()
        net = Network(ref, backend=args.backend)
        report["hardware"], report["buffers_bytes"] = net.info, net.bytes_allocated
        rng = np.random.default_rng(939)
        for b, H, W, Ci, Co in [(2, 5, 7, 3, 5), (3, 17, 41, 3, 7)]:
            R, P = b * H * W, (b * H * W + 2047) // 2048
            x = rng.normal(size=(b,H,W,Ci)).astype(np.float32)
            w = rng.normal(size=(3,3,Ci,Co)).astype(np.float32)
            bias = rng.normal(size=Co).astype(np.float32)
            dy = rng.normal(size=(b,H,W,Co)).astype(np.float32)
            xd, wd, bd, dd = [cuda.to_device(v.ravel(), stream=net.stream) for v in (x,w,bias,dy)]
            y, dx, dw, db = [cuda.device_array(v, np.float32, stream=net.stream) for v in (R*Co,R*Ci,w.size,Co)]
            with tf.GradientTape() as tape:
                xt, wt, bt = [tf.Variable(v.astype(np.float64)) for v in (x,w,bias)]
                yt = tf.nn.conv2d(xt, wt, 1, "SAME") + bt
                loss = tf.reduce_sum(yt * dy.astype(np.float64))
            gx, gw, gb = tape.gradient(loss, [xt,wt,bt])
            net.gemm(0, xd, wd, bd, y, R, Co, 9*Ci, H,W,Ci,Co)
            net.gemm(1, dd, wd, net.dummy, dx, R, Ci, 9*Co,H,W,Ci,Co)
            target = dw if args.backend == "cublas" else net.parts
            net.gemm(2, xd, dd, net.dummy, target, 9*Ci,Co,R,H,W,Ci,Co)
            if args.backend == "explicit":
                k.merge_partials[grid(w.size),256,net.stream](net.parts,dw,w.size,P)
            net.reduce(dd,net.dummy,{},R,Co,3,db,net.dummy)
            net.stream.synchronize()
            check(f"conv_{R}_forward",y.copy_to_host().reshape(yt.shape),yt.numpy())
            check(f"conv_{R}_dx",dx.copy_to_host().reshape(gx.shape),gx.numpy())
            check(f"conv_{R}_dw",dw.copy_to_host().reshape(gw.shape),gw.numpy())
            check(f"conv_{R}_db",db.copy_to_host(),gb.numpy())

        for L,C,constant in [(1,5,True),(20,29,False),(2049,32,False)]:
            x = (np.ones((L,C)) if constant else rng.normal(size=(L,C))).astype(np.float32)
            dy = rng.normal(size=(L,C)).astype(np.float32)
            gamma,beta = rng.normal(size=(2,C)).astype(np.float32)
            dev = lambda a: cuda.to_device(np.ascontiguousarray(a,np.float32).ravel(),stream=net.stream)
            xd, dd, gd, bed = [dev(a) for a in (x,dy,gamma,beta)]
            op = dict(mean=dev(np.zeros(C)),inv=dev(np.zeros(C)),
                      moving_mean=dev(np.zeros(C)),moving_var=dev(np.ones(C)))
            y,dx,dg,db = [cuda.device_array(E,np.float32,stream=net.stream) for E in (L*C,L*C,C,C)]
            net.reduce(xd,net.dummy,op,L,C,0,op["mean"],net.dummy)
            net.reduce(xd,net.dummy,op,L,C,1,net.dummy,net.dummy)
            k.bn_forward[grid(L*C),256,net.stream](xd,y,gd,bed,op["mean"],op["inv"],L*C,C)
            net.reduce(xd,dd,op,L,C,2,db,dg)
            k.bn_backward[grid(L*C),256,net.stream](xd,dd,dx,gd,op["mean"],op["inv"],db,dg,L,C)
            with tf.GradientTape() as tape:
                xt,gt,bt = [tf.Variable(a.astype(np.float64)) for a in (x,gamma,beta)]
                mean,var = tf.nn.moments(xt,[0])
                yt = tf.nn.batch_normalization(xt,mean,var,bt,gt,.001)
                loss = tf.reduce_sum(yt*dy.astype(np.float64))
            gx,gg,gb = tape.gradient(loss,[xt,gt,bt])
            net.stream.synchronize()
            check(f"bn_{L}_forward",y.copy_to_host().reshape(L,C),yt.numpy())
            check(f"bn_{L}_dx",dx.copy_to_host().reshape(L,C),gx.numpy())
            check(f"bn_{L}_dgamma",dg.copy_to_host(),gg.numpy())
            check(f"bn_{L}_dbeta",db.copy_to_host(),gb.numpy())

        for b in (1,20,128):
            I,O=17,29
            x,w,dy = [rng.normal(size=shape).astype(np.float32) for shape in ((b,I),(I,O),(b,O))]
            xd,wd,dd = [cuda.to_device(a.ravel(),stream=net.stream) for a in (x,w,dy)]
            yd,dx,dw = [cuda.device_array(E,np.float32,stream=net.stream) for E in (b*O,b*I,I*O)]
            zero=cuda.to_device(np.zeros(O,np.float32),stream=net.stream)
            net.gemm(3,xd,wd,zero,yd,b,O,I,I=I,O=O)
            net.gemm(4,dd,wd,zero,dx,b,I,O,I=I,O=O)
            target=dw if args.backend=="cublas" else net.parts
            net.gemm(5,xd,dd,zero,target,I,O,b,I=I,O=O)
            if args.backend=="explicit":
                k.merge_partials[grid(I*O),256,net.stream](net.parts,dw,I*O,1)
            net.stream.synchronize()
            check(f"dense_{b}_forward",yd.copy_to_host().reshape(b,O),x.astype(np.float64)@w.astype(np.float64))
            check(f"dense_{b}_dx",dx.copy_to_host().reshape(b,I),dy.astype(np.float64)@w.astype(np.float64).T)
            check(f"dense_{b}_dw",dw.copy_to_host().reshape(I,O),x.astype(np.float64).T@dy.astype(np.float64))
        for b in (1,20,128):
            logits=rng.normal(size=(b,29)).astype(np.float32)*30+10000
            labels=np.arange(b,dtype=np.int64)%29
            ld=cuda.to_device(logits.ravel(),stream=net.stream)
            yd=cuda.to_device(labels,stream=net.stream)
            pd,dz,loss,correct=[cuda.device_array(E,dtype,stream=net.stream)
                for E,dtype in [(b*29,np.float32),(b*29,np.float32),(b,np.float32),(b,np.uint8)]]
            k.softmax_loss[b,32,net.stream](ld,yd,pd,dz,loss,correct,b)
            net.stream.synchronize()
            probs=tf.nn.softmax(logits).numpy()
            expected_grad=(probs-np.eye(29,dtype=np.float32)[labels])/np.float32(b)
            expected_loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits).numpy()
            check(f"softmax_{b}",pd.copy_to_host().reshape(b,29),probs)
            check(f"loss_{b}",loss.copy_to_host(),expected_loss)
            check(f"dloss_{b}",dz.copy_to_host().reshape(b,29),expected_grad)
        for p in (.25,.3,.4):
            values=rng.normal(size=127).astype(np.float32)
            mask=(rng.random(127)>=p).astype(np.uint8)
            vd,md=cuda.to_device(values,stream=net.stream),cuda.to_device(mask,stream=net.stream)
            out=cuda.device_array(127,np.float32,stream=net.stream)
            factor=np.float32(1/(1-p))
            k.dropout[1,256,net.stream](vd,md,out,factor,127)
            net.stream.synchronize()
            check(f"dropout_{p}",out.copy_to_host(),values*mask.astype(np.float32)*factor)
        theta=rng.normal(size=17).astype(np.float32)
        moment=(rng.normal(size=17)*.001).astype(np.float32)
        velocity=(rng.random(17)*.01).astype(np.float32)
        td,md,vd=[cuda.to_device(a,stream=net.stream) for a in (theta,moment,velocity)]
        error=cuda.to_device(np.zeros(1,np.int32),stream=net.stream)
        for step in (1,2,1440):
            g=rng.normal(size=17).astype(np.float32);g[0]=0
            gd=cuda.to_device(g,stream=net.stream)
            t=np.float32(step)
            alpha=np.float32(.001)*np.sqrt(np.float32(1)-np.float32(.999)**t)/(np.float32(1)-np.float32(.9)**t)
            k.adam[1,256,net.stream](td,gd,md,vd,np.float32(alpha),17,error)
            moment+= (g-moment)*np.float32(.1)
            velocity+=(g*g-velocity)*np.float32(.001)
            theta-=moment*alpha/(np.sqrt(velocity)+np.float32(1e-7))
            net.stream.synchronize()
            check(f"adam_{step}_theta",td.copy_to_host(),theta,1e-7,1e-5)
            check(f"adam_{step}_m",md.copy_to_host(),moment,1e-7,1e-5)
            check(f"adam_{step}_v",vd.copy_to_host(),velocity,1e-7,1e-5)
        bad=cuda.to_device(np.array([1.,np.inf],np.float32),stream=net.stream)
        k.finite_check[1,32,net.stream](bad,2,error)
        net.stream.synchronize()
        check("flag_inf",error.copy_to_host(),np.array([3],np.int32),0,1e-8)
        report["estado"]="aprobado"
    except BaseException:
        report["estado"]="fallido"
        report["error"]=traceback.format_exc()
        raise
    finally:
        report["fin"]=datetime.now(timezone.utc).isoformat()
        report["fuentes"]={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__),*(ROOT/"src/asl_cuda").glob("*.py")]}
        (args.destino/"validacion.json").write_text(json.dumps(report,indent=2),encoding="utf8")


if __name__=="__main__":
    main()
