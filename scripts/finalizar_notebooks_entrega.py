"""Preserve native notebook originals; add full native-kernel diagnostic annexes."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import nbformat

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--revision',required=True)
    args=ap.parse_args()
    base=ROOT/'resultados/preparacion'/args.revision
    done=json.loads((base/'completadas.json').read_text(encoding='utf8'))
    assert len(done)==2 and all(d['codigo']==0 for d in done)
    contracts=json.loads((base/'ejecuciones.json').read_text(encoding='utf8'))
    archive=base/'notebooks-nativos'
    archive.mkdir(exist_ok=False)
    raw=(base/'controlador.stderr.log').read_text(encoding='utf8',errors='replace')
    marker='[IPKernelApp] WARNING'
    first=raw.find(marker);second=raw.find(marker,first+len(marker))
    assert first>=0 and second>first,'Cannot safely separate CPU/GPU native stderr'
    segments=[raw[:second],raw[second:]]
    changes=[]
    for item,completed,stderr in zip(contracts,done,segments):
        path=Path(item['salida'])
        assert sha(path)==completed['notebook_sha256']
        preserved=archive/path.name
        shutil.copyfile(path,preserved)
        assert sha(preserved)==sha(path)
        n=nbformat.read(path,as_version=4)
        original_code=[(c.source,c.get('execution_count'),c.get('outputs')) for c in n.cells if c.cell_type=='code']
        n.cells.append(nbformat.v4.new_markdown_cell('''## Alcance de esta ejecución adicional

Las celdas de cálculo anteriores fueron ejecutadas directamente con PCUDA y
conservan sus salidas nativas. Esta corrida adicional no reemplaza las cinco
mediciones históricas del informe. **Los totales continuos de este notebook
incluyen presentación de tablas/gráficas entre fases**; los intervalos fit/evaluate
individuales mantienen sus fronteras. No se mezclan alcances para calcular speedup.

La configuración y el RUN se conservan. Para repetir el entrenamiento debe
prepararse otro EXP/RUN; abrir el notebook permite leer todas sus salidas sin
volver a entrenar. Los avisos del kernel nativo se anexan abajo porque TensorFlow
puede escribir directamente a stderr del sistema, fuera del canal de salida de
las celdas. Este anexo documental no simula una ejecución de código.
'''))
        n.cells.append(nbformat.v4.new_markdown_cell('## Diagnósticos nativos completos del kernel\n\n'
            'Texto literal del tramo de stderr del supervisor correspondiente a este kernel. '
            'Los avisos se conservan; la corrida completó con código0.\n\n```text\n'+stderr+'\n```'))
        n.metadata['cierre_documental']=dict(fecha_utc=datetime.now(timezone.utc).isoformat(),
            codigo_y_salidas_nativas_sin_cambios=True,original_sha256=sha(preserved),
            original_preservado=preserved.relative_to(ROOT).as_posix(),
            diagnosticos_nativos_sha256=hashlib.sha256(stderr.encode()).hexdigest())
        assert original_code==[(c.source,c.get('execution_count'),c.get('outputs')) for c in n.cells if c.cell_type=='code']
        nbformat.validate(n)
        nbformat.write(n,path)
        changes.append(dict(notebook=path.relative_to(ROOT).as_posix(),sha256=sha(path),
                            original_sha256=sha(preserved),diagnosticos_caracteres=len(stderr),
                            celdas_codigo_sin_cambios=True))
    (base/'cierre-documental.json').write_text(json.dumps(changes,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps(changes,ensure_ascii=False))


if __name__=='__main__':main()
