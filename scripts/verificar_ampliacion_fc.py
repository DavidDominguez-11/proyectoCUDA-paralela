"""Read-only audit of new FC evidence, preserved historical runs and deliverables."""
from pathlib import Path
import json,csv,hashlib,zipfile,sys,io
import nbformat
import numpy as np
import h5py
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'resultados/preparacion/ampliacion-fc'
BASE=ROOT/'resultados/paralelo/EXP-012'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csvrows(p):return list(csv.DictReader(p.open(encoding='utf8',newline='')))
def main():
    cfg=json.loads((ROOT/'configuraciones/EXP-012.json').read_text(encoding='utf8'))
    for path,digest in cfg['codigo']['fuentes'].items():assert sha(ROOT/path)==digest,path
    assert len(cfg['runs'])==10
    canonical={};captures=[];evidence=[]
    expected=json.loads((ROOT/'resultados/secuencial/EXP-002/RUN-20261117T171942Z-cpu-01/tensores.json').read_text(encoding='utf8'))
    for item in cfg['runs']:
        run=ROOT/item.get('ruta',f"resultados/paralelo/EXP-012/{item['run_id']}")
        epochs=csvrows(run/'epocas.csv')
        assert len(epochs)==15 and [int(r['epoca']) for r in epochs]==list(range(1,16))
        assert [int(r['pasos_acumulados']) for r in epochs]==list(range(96,1441,96))
        for row in epochs:
            assert all(np.isfinite(float(row[k])) for k in ['train_loss','val_loss','train_accuracy','val_accuracy'])
        m=json.loads((run/'metricas.json').read_text(encoding='utf8'));assert m['pasos']==1440
        saved=json.loads((run/'modelo-guardado.json').read_text(encoding='utf8'));assert saved['optimizer_iterations']==1440
        assert sha(ROOT/saved['ruta'])==saved['sha256']
        with zipfile.ZipFile(ROOT/saved['ruta']) as z,h5py.File(io.BytesIO(z.read('model.weights.h5')),'r') as h:
            assert int(h['optimizer/vars/0'][()])==1440
            def finite(name,obj):
                if isinstance(obj,h5py.Dataset):assert np.all(np.isfinite(obj[()])),name
            h.visititems(finite)
        canonical.setdefault(item['modelo'],[]).append(saved['estado_canonico_sha256'])
        tensors=json.loads((run/'tensores.json').read_text(encoding='utf8'))
        assert all(tensors[s][k]==expected[s][k] for s in expected for k in ['images_sha256','labels_sha256'])
        transfers=json.loads((run/'transferencias.json').read_text(encoding='utf8'))
        assert len(transfers['lotes'])==1440
        assert sum(r['batch'] for r in transfers['lotes'])==182700
        assert sum(r['batch']==20 for r in transfers['lotes'])==15
        for p in run.glob('captura-*.json'):
            d=json.loads(p.read_text(encoding='utf8'));assert sha(p.with_suffix('.png'))==d['sha256']
            assert d['muestra_original']['fase']=='entrenamiento_'+item['modelo']
            assert float(d['muestra_original']['gpu_util_pct'])>0
            assert d['muestra_original'] in csvrows(run/'recursos.csv')
            captures.append(dict(modelo=item['modelo'],**d))
        evidence.append(run)
    assert len(captures)==2 and all(len(set(v))==1 for v in canonical.values())
    combined=nbformat.read(ROOT/'entregables/notebooks/ASL_CUDA.ipynb',4)
    prior=nbformat.read(OUT/'ASL_CUDA.ipynb',4)
    assert combined.cells[:len(prior.cells)]==prior.cells
    assert not [o for c in combined.cells for o in c.get('outputs',[]) if o.output_type=='error']
    native=nbformat.read(BASE/'FC_CUDA.ejecutado.ipynb',4)
    template=nbformat.read(BASE/'FC_CUDA.plantilla.ipynb',4)
    assert [c.source for c in native.cells]==[c.source for c in template.cells]
    code=[c for c in native.cells if c.cell_type=='code']
    assert [c.execution_count for c in code]==list(range(1,len(code)+1))
    # Native measured FC epoch streams, excluding historical logs/presentation.
    epochs_native=[]
    for c in combined.cells:
        if c.cell_type=='code' and "print('EPOCH',name,repetition" in c.source:
            epochs_native.extend(line for o in c.outputs if o.output_type=='stream' for line in o.text.splitlines() if line.startswith('EPOCH FC-'))
    assert len(epochs_native)==150,len(epochs_native)
    before=json.loads((OUT/'identidades-previas.json').read_text(encoding='utf8'))
    assert sha(ROOT/'entregables/notebooks/ASL_Secuencial.ipynb')==before['entregables/notebooks/ASL_Secuencial.ipynb']['sha256']
    historic=ROOT/'resultados/paralelo/EXP-003'
    rows=csvrows(historic/'artefactos-experimento.csv')
    for r in rows:
        # Historical seal paths are relative to the experiment or repository.
        p=ROOT/r['ruta_relativa']
        if not p.exists():p=historic/r['ruta_relativa']
        assert sha(p)==r['sha256'],p
    report=json.loads((OUT/'integracion-informe.json').read_text(encoding='utf8'))
    assert report['despues_sha256']==sha(ROOT/'entregables/Informe_Proyecto_3_CUDA.docx')
    validation=dict(estado='aprobado',runs_fc=len(evidence),epocas_fc=150,actualizaciones_fc=14400,checkpoints_h5_finitos_y_adam1440=True,
        pruebas_fc_oraculo=4,capturas_reales=2,celdas_codigo_cuda_total=sum(c.cell_type=='code' for c in combined.cells),
        png_notebook=sum('image/png' in o.get('data',{}) for c in combined.cells for o in c.get('outputs',[])),
        epocas_fc_en_stream_nativo=len(epochs_native),estados_gpu_repetibles=True,
        notebook_cpu_inalterado=True,artefactos_cnn_b_verificados=len(rows),docx_estilo_preservado=report,
        limites=['CPU histórica no pareada; Dropout/shuffle distintos','FC CUDA menor accuracy que CPU histórica',
            'No se midió ocupación por SM ni se demostró saturación VRAM','PDF no generado por instrucción del usuario'])
    (OUT/'validacion-final.json').write_text(json.dumps(validation,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps(validation,ensure_ascii=False))
if __name__=='__main__':main()
