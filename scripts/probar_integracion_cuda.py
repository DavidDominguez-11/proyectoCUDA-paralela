"""Ten short synthetic steps, restore/export/CPU evaluation/save admission test."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from asl_cuda.model import Network,cpu_reference,compiled_resources


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--destino",type=Path,required=True)
    args=ap.parse_args()
    args.destino.mkdir(parents=True,exist_ok=False)
    result=dict(tipo="admision_integracion_sintetica_no_benchmark",inicio=datetime.now(timezone.utc).isoformat())
    try:
        tf,ref=cpu_reference()
        net=Network(ref,backend="cublas")
        rng=np.random.default_rng(13)
        x=rng.random((128,64,64,3),dtype=np.float32)
        y=np.arange(128,dtype=np.int64)%29
        for step in range(10):
            b=20 if step==9 else 128
            net.train_step(x[:b],y[:b],0,step)
        _,total=net.export(optimizer=True)
        assert int(ref.optimizer.iterations.numpy())==10
        assert int(net.counter.copy_to_host()[0])==10
        assert total[2]==9*128+20
        before=[v.numpy().copy() for v in ref.weights]
        val=ref.evaluate(x[:32],y[:32],batch_size=32,verbose=0,return_dict=True)
        assert all(np.array_equal(a,v.numpy()) for a,v in zip(before,ref.weights))
        checkpoint=ROOT/"data/preparacion-cuda"/(args.destino.name+".keras")
        checkpoint.parent.mkdir(exist_ok=True)
        if checkpoint.exists():
            raise FileExistsError(checkpoint)
        ref.save(checkpoint)
        loaded=tf.keras.models.load_model(checkpoint)
        assert int(loaded.optimizer.iterations.numpy())==10
        assert all(np.array_equal(a,v.numpy()) for a,v in zip(before,loaded.weights))
        net.restore()
        assert int(net.counter.copy_to_host()[0])==0
        result.update(estado="aprobado",pasos=10,muestras=1172,evaluacion_cpu={k:float(v) for k,v in val.items()},
            guardado_y_recarga_exactos=True,restauracion_final=True,
            modelo_prueba_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            recursos_compilados=compiled_resources())
        print(json.dumps(result,indent=2),flush=True)
    except BaseException:
        result.update(estado="fallido",error=traceback.format_exc())
        raise
    finally:
        result["fin"]=datetime.now(timezone.utc).isoformat()
        result["fuentes"]={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__),*(ROOT/"src/asl_cuda").glob("*.py")]}
        (args.destino/"validacion.json").write_text(json.dumps(result,indent=2),encoding="utf8")


if __name__=="__main__":
    main()
