"""NEW FC-only oracle checks: same masks, gradients, BN state and Adam update."""
import os
os.environ['NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS']='0'
from pathlib import Path
import sys,json,time,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from asl_cuda.fc import fc_reference,FCNetwork
from benchmark_cpu import write_json,utc
from threadpoolctl import threadpool_limits

def main():
    dest=ROOT/'resultados/preparacion/ampliacion-fc/validacion-01'
    dest.mkdir(exist_ok=False)
    result=dict(inicio_utc=utc(),estado='ejecutando',pruebas=[])
    try:
        with threadpool_limits(limits=6):
            for name in ['FC-A','FC-B']:
                tf,model,seed,initial=fc_reference(name)
                net=FCNetwork(model,seed)
                for b in [20,128]:
                    net.restore()
                    # Reset CPU weights/BN/optimizer to the exact initial archive.
                    with np.load(initial,allow_pickle=False) as a:
                        for i,v in enumerate([*model.variables,*model.optimizer.variables]):v.assign(a[f'arr_{i}'])
                    rng=np.random.default_rng(720+b)
                    x=rng.random((b,64,64,3),dtype=np.float32)
                    y=rng.integers(0,29,b,dtype=np.int64)
                    net.upload(x,y,0,0)
                    net.forward(b);net.backward(b);net.stream.synchronize()
                    dropops=iter([op for op in net.ops if op['kind']=='drop'])
                    with tf.GradientTape() as tape:
                        z=tf.convert_to_tensor(x)
                        for layer in model.layers:
                            if type(layer).__name__=='Dropout':
                                op=next(dropops);mask=op['host_mask'][:b*op['sample_elements']].reshape(z.shape)
                                z=z*tf.cast(mask,tf.float32)/np.float32(1-layer.rate)
                            else:z=layer(z,training=True)
                        loss=tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y,z))
                    gradients=tape.gradient(loss,model.trainable_variables)
                    gp=net.probs.copy_to_host()[:b*29].reshape(b,29)
                    probs_max=float(np.max(np.abs(gp-z.numpy())))
                    np.testing.assert_allclose(gp,z.numpy(),rtol=2e-3,atol=2e-5)
                    checks=[]
                    for p,g in zip(net.params,gradients):
                        got=p['grad'].copy_to_host().reshape(g.shape);want=g.numpy()
                        maxabs=float(np.max(np.abs(got-want)))
                        relative=float(np.linalg.norm((got-want).ravel())/max(np.linalg.norm(want.ravel()),1e-12))
                        if relative>2e-3 or maxabs>2e-4:raise AssertionError(('gradient',name,b,p['variable'].path,maxabs,relative))
                        checks.append(dict(variable=p['variable'].path,max_abs=maxabs,relative_l2=relative))
                    bn=next(op for op in net.ops if op['kind']=='bn')
                    np.testing.assert_allclose(bn['moving_mean'].copy_to_host(),bn['layer'].moving_mean.numpy(),rtol=2e-5,atol=2e-7)
                    np.testing.assert_allclose(bn['moving_var'].copy_to_host(),bn['layer'].moving_variance.numpy(),rtol=2e-5,atol=2e-7)
                    # Keep the CPU update for comparison BEFORE export replaces it.
                    model.optimizer.apply_gradients(zip(gradients,model.trainable_variables))
                    cpu_after=[v.numpy().copy() for v in model.trainable_variables]
                    net.update(b);net.stream.synchronize()
                    adam_max=max(float(np.max(np.abs(p['value'].copy_to_host().reshape(a.shape)-a))) for p,a in zip(net.params,cpu_after))
                    if adam_max>3e-4:raise AssertionError(('Adam',name,b,adam_max))
                    net.export(optimizer=True)
                    assert net.steps==1 and int(model.optimizer.iterations.numpy())==1
                    net.restore()
                    assert net.counter.copy_to_host()[0]==0
                    assert all(np.array_equal(p['value'].copy_to_host(),p['initial'].ravel()) for p in net.params)
                    assert all(not np.any(p[s].copy_to_host()) for p in net.params for s in ['m','v'])
                    result['pruebas'].append(dict(modelo=name,batch=b,probabilities_max_abs=probs_max,
                        adam_max_abs=adam_max,gradientes=checks,bn_momentum=bn['momentum'],
                        buffers_bytes=net.bytes_allocated,parametros=sum(p['value'].size for p in net.params),restauracion=True))
                    print('APROBADO',name,b,'prob',probs_max,'adam',adam_max,flush=True)
                net.blas.close()
                del net,model
                import gc;gc.collect()
                tf.keras.backend.clear_session()
            result['estado']='aprobado'
    except Exception:
        result['estado']='fallido';result['error']=traceback.format_exc();raise
    finally:
        result['fin_utc']=utc();write_json(dest/'validacion.json',result)

if __name__=='__main__':main()
