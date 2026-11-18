"""Seal new FC artifacts and final deliverables; never modify historical seals."""
from pathlib import Path
from datetime import datetime,timezone
import csv,json,hashlib,zipfile
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'resultados/paralelo/EXP-012'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_csv(p,rows):
    with p.open('x',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def main():
    seal=BASE/'artefactos-ampliacion.csv'
    if seal.exists():raise FileExistsError(seal)
    cfg=json.loads((ROOT/'configuraciones/EXP-012.json').read_text(encoding='utf8'))
    valid=json.loads((ROOT/'resultados/preparacion/ampliacion-fc/validacion-final.json').read_text(encoding='utf8'))
    assert valid['estado']=='aprobado'
    sources=set(ROOT/p for p in cfg['codigo']['fuentes'])
    sources.update(ROOT/'configuraciones'/f'EXP-{i:03d}.json' for i in [10,11,12])
    sources.add(ROOT/cfg['validacion']['ruta'])
    sources.add(ROOT/'resultados/secuencial/EXP-002/manifest_splits.csv')
    sources.add(ROOT/'resultados/secuencial/EXP-002/RUN-20261117T171942Z-cpu-01/tensores.json')
    sources.add(ROOT/'resultados/secuencial/EXP-002/RUN-20261117T171942Z-cpu-01/estados-iniciales.json')
    sources.update((ROOT/'scripts').glob('*fc*.py'))
    with zipfile.ZipFile(BASE/'fuentes-congeladas.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(sources):z.write(p,p.relative_to(ROOT).as_posix())
    write_csv(BASE/'fuentes-congeladas.csv',[dict(ruta_relativa=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(sources)])
    models=[];runs=[]
    for item in cfg['runs']:
        run=ROOT/item.get('ruta',f"resultados/paralelo/EXP-012/{item['run_id']}")
        saved=json.loads((run/'modelo-guardado.json').read_text(encoding='utf8'))
        p=ROOT/saved['ruta'];assert sha(p)==saved['sha256']
        models.append(dict(modelo=item['modelo'],repeticion=item['repeticion'],ruta_relativa=saved['ruta'],
            bytes=p.stat().st_size,sha256=saved['sha256'],estado_canonico_sha256=saved['estado_canonico_sha256']))
        runs.append(dict(**item,estado='entrenamiento_y_checkpoint_completados',metricas_sha256=sha(run/'metricas.json')))
    write_csv(BASE/'modelos.csv',models)
    series=dict(estado='completada',cerrado_utc=datetime.now(timezone.utc).isoformat(),runs=runs,
        epocas=150,actualizaciones=14400,exclusiones=['EXP-010 incompleto: CSV sin cierre/modelo'],
        recuperacion='FC-A1 EXP-011 guardada antes de error Styler; se reutiliza sin entrenar otra vez',
        historicos_no_ejecutados=['CPU EXP-002','CNN-B EXP-003','notebooks EXP-008/009'],
        limitaciones=valid['limites'])
    (BASE/'serie.json').write_text(json.dumps(series,indent=2,ensure_ascii=False),encoding='utf8')
    paths=set()
    for relative in ['resultados/paralelo/EXP-010','resultados/paralelo/EXP-011','resultados/paralelo/EXP-012',
        'resultados/preparacion/ampliacion-fc','resultados/comparacion/ampliacion-fc']:
        paths.update(p for p in (ROOT/relative).rglob('*') if p.is_file())
    paths.update(sources)
    paths.update([ROOT/'entregables/Informe_Proyecto_3_CUDA.docx',ROOT/'entregables/notebooks/ASL_CUDA.ipynb',
        ROOT/'entregables/notebooks/ASL_Secuencial.ipynb',ROOT/'entregables/notebooks/README.md',ROOT/'README.md'])
    paths.update((ROOT/'docs/trazabilidad').glob('*.md'))
    paths.discard(seal)
    write_csv(seal,[dict(ruta_relativa=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(paths)])
    print('SELLADO',len(paths),'artefactos, SHA-256',sha(seal))
if __name__=='__main__':main()
