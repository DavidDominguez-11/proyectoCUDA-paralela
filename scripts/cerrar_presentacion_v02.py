"""Finalize snapshot naming and seal after the documentation link audit."""
from pathlib import Path
from datetime import datetime,timezone
import csv,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'entregables/presentacion'
QA=ROOT/'resultados/preparacion/presentacion-v01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    mappings=[]
    directory=(QA/'antecedentes').resolve()
    assert directory.is_relative_to(ROOT.resolve())
    for p in directory.glob('*.md'):
        dest=p.with_suffix('.md.bak')
        assert not dest.exists()
        digest=sha(p);p.rename(dest);assert sha(dest)==digest
        mappings.append(dict(antes=p.relative_to(ROOT).as_posix(),ahora=dest.relative_to(ROOT).as_posix(),sha256=digest))
    (QA/'reubicacion-snapshots.json').write_text(json.dumps(mappings,indent=2),encoding='utf8')
    for rel in ['docs/trazabilidad/presentacion.md','docs/trazabilidad/bitacora.md']:
        with (ROOT/rel).open('a',encoding='utf8') as f:f.write('\n\nAuditoría documental final:las cuatro copias históricas de Markdown se renombraron .md.bak sin cambiar sus bytes; sus enlaces son relativos a su ubicación original yno documentación activa. El primer control detectó esos enlaces de las copias. El manifiesto inicial artefactos.csv se conserva comoantecedente; artefactos-v02.csv registra elcierre actual ylas rutas de snapshots corregidas. Ninguna diapositiva odato experimental se cambió.\n')
    with (OUT/'README.md').open('a',encoding='utf8') as f:f.write('\n\n[Manifiesto SHA-256 de entrega](artefactos-v02.csv). Las copias previas de trazabilidad se conservan con extensión.md.bak para distinguirlas de documentación activa.\n')
    paths={ROOT/'entregables/Presentacion_Proyecto_3_CUDA.pptx',ROOT/'docs/trazabilidad/presentacion.md',ROOT/'docs/trazabilidad/bitacora.md'}
    paths.update(p for p in OUT.rglob('*') if p.is_file())
    paths.update(p for p in QA.rglob('*') if p.is_file())
    paths.update((ROOT/'scripts').glob('*presentacion*'))
    seal=OUT/'artefactos-v02.csv';paths.discard(seal)
    with seal.open('x',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['ruta_relativa','bytes','sha256']);writer.writeheader()
        for p in sorted(paths):writer.writerow(dict(ruta_relativa=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    print('CIERRE',len(paths),'artefactos; snapshots idénticos yseparados deMarkdownactivo.')
if __name__=='__main__':main()
