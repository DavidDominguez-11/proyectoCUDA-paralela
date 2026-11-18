"""Verify two fully executed delivery notebooks and their native training outputs."""
import argparse
import base64
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import zipfile
import h5py
import nbformat
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csv_rows(p):
    with p.open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--revision',required=True)
    args=ap.parse_args()
    base=ROOT/'resultados/preparacion'/args.revision
    destination=base/'validacion-final.json'
    if destination.exists():raise FileExistsError(destination)
    result=dict(fecha_utc=datetime.now(timezone.utc).isoformat(),notebooks=[],historicos_intactos=True)
    reference=read(ROOT/'resultados/secuencial/EXP-002/RUN-20261117T171942Z-cpu-01/tensores.json')
    for item in read(base/'ejecuciones.json'):
        path=Path(item['salida']);run=Path(item['run'])
        nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
        template=nbformat.read(item['plantilla'],as_version=4)
        assert [(c.cell_type,c.source) for c in nb.cells[:len(template.cells)]]==[(c.cell_type,c.source) for c in template.cells]
        cells=[c for c in nb.cells if c.cell_type=='code']
        assert all(c.execution_count is not None for c in cells)
        assert [c.execution_count for c in cells]==list(range(1,len(cells)+1))
        assert not any(o.output_type=='error' for c in cells for o in c.outputs)
        fresh=[]
        for c in nb.cells:
            if c.cell_type=='markdown' and c.source.startswith('## Cinco mediciones históricas'):break
            fresh.append(c)
        streams=''.join(o.text for c in fresh for o in c.get('outputs',[]) if o.output_type=='stream')
        epoch_lines=[line for line in streams.splitlines() if line.startswith('EPOCH ')]
        expected=75 if item['tipo']=='cpu' else 15
        assert len(epoch_lines)==expected,(path,len(epoch_lines),expected)
        saved=read(run/'salida.json');assert saved['codigo']==0
        tensors=read(run/'tensores.json')
        for split in ('train','val','test'):
            for key in ('images_sha256','labels_sha256'):assert tensors[split][key]==reference[split][key]
        models=read(run/'modelos-guardados.json') if item['tipo']=='cpu' else {'CNN-B':read(run/'modelo-guardado.json')}
        optimizer=[]
        for name,meta in models.items():
            if not meta['ruta'].endswith('.keras'):continue
            checkpoint=ROOT/meta['ruta'];assert sha(checkpoint)==meta['sha256']
            with zipfile.ZipFile(checkpoint) as archive,h5py.File(io.BytesIO(archive.read('model.weights.h5')),'r') as weights:
                steps=int(weights['optimizer/vars/0'][()]);assert steps==1440
            optimizer.append(dict(modelo=name,pasos=steps))
        all_streams=''.join(o.text for c in cells for o in c.outputs if o.output_type=='stream')
        old='EXP-002' if item['tipo']=='cpu' else 'EXP-003'
        branch='secuencial' if item['tipo']=='cpu' else 'paralelo'
        historical=ROOT/'resultados'/branch/old
        series=read(historical/'serie.json')
        ids=series['runs'] if item['tipo']=='cpu' else [r['run_id'] for r in series['corridas']]
        for name in ids:
            for stream in ('stdout','stderr'):
                text=(historical/name/(stream+'.log')).read_text(encoding='utf8')
                assert text in all_streams,(name,stream,'log incomplete')
        images=[base64.b64decode(o.data['image/png']) for c in cells for o in c.outputs
                if o.output_type in ('display_data','execute_result') and 'image/png' in o.data]
        captures=list(run.glob('captura-*.png'));assert captures,'Missing new live capture'
        for capture in captures:
            assert capture.read_bytes() in images
            assert sha(capture)==read(capture.with_suffix('.json'))['sha256']
        for record in csv_rows(run/'artefactos.csv'):
            target=run/record['ruta_relativa']
            assert target.stat().st_size==int(record['bytes']) and sha(target)==record['sha256']
        result['notebooks'].append(dict(archivo=path.relative_to(ROOT).as_posix(),sha256=sha(path),
            experimento=item['experimento'],run_id=item['run_id'],celdas_codigo=len(cells),
            errores=0,epocas_nuevas=len(epoch_lines),imagenes_incrustadas=len(images),
            modelos=optimizer,logs_historicos_completos=10,capturas_nuevas=len(captures)))
    for row in csv_rows(ROOT/'resultados/paralelo/EXP-003/artefactos-experimento.csv'):
        assert sha(ROOT/row['ruta_relativa'])==row['sha256']
    assert sha(ROOT/'entregables/Informe_Proyecto_3_CUDA.docx')=='c1fa886f3bf4b747c1dbb17937dc193eabf645a088e3cabd46c2edea802a22c7'
    result['estado']='aprobado'
    result['script_sha256']=sha(Path(__file__))
    destination.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
