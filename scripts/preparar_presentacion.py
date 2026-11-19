"""Read existing measurements and prepare visual assets; NO model execution."""
from pathlib import Path
from datetime import datetime,timezone
import json,csv,hashlib,shutil
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'entregables/presentacion'
QA=ROOT/'resultados/preparacion/presentacion-v01'
def read(path):return list(csv.DictReader((ROOT/path).open(encoding='utf8',newline='')))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def avg(a):return sum(a)/len(a)
def main():
    OUT.mkdir(exist_ok=False);(OUT/'recursos').mkdir()
    QA.mkdir(parents=True,exist_ok=False);(QA/'antecedentes').mkdir()
    initial={}
    for rel in ['entregables/Informe_Proyecto_3_CUDA.docx','entregables/InformeFinal_Proyecto_3_CUDA.pdf',
        'entregables/notebooks/ASL_Secuencial.ipynb','entregables/notebooks/ASL_CUDA.ipynb','entregables/SKILL-P.md']:
        p=ROOT/rel
        if p.exists():initial[rel]=dict(sha256=sha(p),bytes=p.stat().st_size)
    (QA/'identidades-previas.json').write_text(json.dumps(initial,indent=2),encoding='utf8')
    for name in ['README.md','decisiones.md','cambios.md','bitacora.md']:
        shutil.copy2(ROOT/'docs/trazabilidad'/name,QA/'antecedentes'/name)
    now=datetime.now(timezone.utc).isoformat()
    for name,title,text in [
        ('decisiones.md','DEC-028 — Presentación visual del proyecto',
         'Crear PPTX local editable con los seis pasos y ampliación FC, tablas/gráficas nativas, capturas originales y notas del expositor. Aplicar entregables/SKILL-P.md. Reutilizar resultados existentes sin entrenar ni modificar DOCX/PDF/notebooks. Verificar archivo y renderizar con PowerPoint local. Las herramientas auxiliares mencionadas en la habilidad no están incluidas; se sustituyen por validación OOXML y revisión nativa equivalentes. No se requiere otra campaña experimental.'),
        ('cambios.md','CAM-013 — Presentación y material de exposición',
         'Trabajo aditivo: entregables/Presentacion_Proyecto_3_CUDA.pptx, fuentes/guion/recursos en entregables/presentacion y QA en resultados/preparacion/presentacion-v01. Identidades previas de informe PDF/DOCX, notebooks y habilidad registradas. El informe actual tiene ediciones posteriores del usuario y se conserva íntegro.'),
        ('bitacora.md','BIT-015 — Presentación del proyecto',
         'Inicio UTC:'+now+'. Responsable:Codex por petición del usuario. Revisados PDF normativo, trazabilidad, resultados CPU/CUDA/FC y habilidad local SKILL-P.md. Node pptxgenjs no estaba disponible; se añadió solamente a data/presentacion-tooling, sin modificar PCUDA. Presentación sin nuevos entrenamientos; documentos existentes protegidos mediante hashes.')]:
        with (ROOT/'docs/trazabilidad'/name).open('a',encoding='utf8') as f:f.write('\n\n## '+title+'\n\n'+text+'\n')
    sources=set()
    def rows(p):sources.add(ROOT/p);return read(p)
    compare=rows('resultados/comparacion/ampliacion-fc/comparacion.csv')
    observation=rows('resultados/comparacion/ampliacion-fc/observaciones.csv')
    quality=rows('resultados/comparacion/ampliacion-fc/calidad.csv')
    cnnq=rows('resultados/paralelo/EXP-003/calidad_comparada.csv')
    for origin,tag in [('CPU','cpu'),('CUDA','gpu')]:
        a=[float(r['accuracy'])*100 for r in cnnq if r['dispositivo']==tag]
        l=[float(r['loss']) for r in cnnq if r['dispositivo']==tag]
        quality.append(dict(modelo='CNN-B',origen=origin,n=5,accuracy_media_pct=avg(a),accuracy_min_pct=min(a),accuracy_max_pct=max(a),loss_media=avg(l)))
    baseline=rows('resultados/secuencial/EXP-002/resumen.csv')
    common=rows('resultados/secuencial/EXP-002/resumen_referencia_comun.csv')[0]
    phasecompare=rows('resultados/comparacion/paso-6-revision-02/comparacion_fases.csv')
    breakdown=rows('resultados/comparacion/ampliacion-fc/desglose.csv')
    cb=rows('resultados/comparacion/paso-6-revision-02/desglose_entrenamiento.csv')
    fcepochs=rows('resultados/comparacion/ampliacion-fc/epocas-comparacion.csv')
    resources=rows('resultados/comparacion/ampliacion-fc/recursos-resumen.csv')
    cnnresources=rows('resultados/paralelo/EXP-003/recursos_resumen.csv')
    cfg=json.loads((ROOT/'configuraciones/EXP-012.json').read_text(encoding='utf8'))
    sources.add(ROOT/'configuraciones/EXP-012.json')
    captures={
        'cpu':'resultados/secuencial/EXP-008/RUN-20261118T041129Z-cpu-notebook-01/captura-cpu-sostenido.png',
        'cnn':'resultados/paralelo/EXP-009/RUN-20261118T041129Z-cuda-notebook-01/captura-cuda.png',
        'prep':'resultados/evidencias/EXP-003/20261117T233947Z-cpu.png',
        'fcA':'resultados/paralelo/EXP-011/RUN-20261118T134323Z-fc-a-01/captura-fc-a.png',
        'fcB':'resultados/paralelo/EXP-012/RUN-20261118T134514Z-fc-b-01/captura-fc-b.png'}
    images={};capturemeta={}
    for key,rel in captures.items():
        p=ROOT/rel;target=OUT/'recursos'/('captura-'+key+'.png');shutil.copy2(p,target)
        sources.add(p);meta=p.with_suffix('.json');sources.add(meta)
        capturemeta[key]=json.loads(meta.read_text(encoding='utf8'))
        with Image.open(target) as im:w,h=im.size
        images[key]=dict(path=target.relative_to(ROOT).as_posix(),w=w,h=h,sha256=sha(target))
    manifest=rows('resultados/secuencial/EXP-002/manifest_splits.csv')
    for label in ['A','B','C','M','N','S']:
        row=next(r for r in manifest if r['clase']==label and r['split']=='train')
        source=Path(cfg['datos']['ruta'])/row['ruta_relativa'];assert sha(source)==row['sha256']
        target=OUT/'recursos'/f'ASL-{label}.jpg';shutil.copy2(source,target)
        with Image.open(target) as im:w,h=im.size
        images[label]=dict(path=target.relative_to(ROOT).as_posix(),w=w,h=h,sha256=sha(target),original=row)
    cepochs=[]
    cpuseries=json.loads((ROOT/'resultados/secuencial/EXP-002/serie.json').read_text(encoding='utf8'))
    gpuseries=json.loads((ROOT/'resultados/paralelo/EXP-003/serie.json').read_text(encoding='utf8'))
    commonruns=[]
    for tag,runs,folder in [('CPU',cpuseries['runs'],'secuencial/EXP-002'),('CUDA',[r['run_id'] for r in gpuseries['corridas']],'paralelo/EXP-003')]:
        for i,rid in enumerate(runs,1):
            if tag=='CPU':
                p=ROOT/'resultados'/folder/rid/'historiales.json';h=json.loads(p.read_text(encoding='utf8'))['CNN-B'];sources.add(p)
                for e in range(15):cepochs.append(dict(origen=tag,modelo='CNN-B',repeticion=i,epoca=e+1,val_accuracy=h['val_accuracy'][e],val_loss=h['val_loss'][e]))
                times=read(f'resultados/{folder}/{rid}/tiempos.csv');sources.add(ROOT/f'resultados/{folder}/{rid}/tiempos.csv')
                phases=['preparacion_cpu','entrenamiento_CNN-A','evaluacion_CNN-A','entrenamiento_CNN-B','evaluacion_CNN-B','entrenamiento_FC-A','evaluacion_FC-A','entrenamiento_FC-B','evaluacion_FC-B','pca','entrenamiento_RF-A','validacion_RF-A','entrenamiento_RF-B','validacion_RF-B']
                value=sum(float(r['duracion_s']) for r in times if r['fase'] in phases)
                commonruns.append(value)
            else:
                p=ROOT/'resultados'/folder/rid/'epocas.csv';sources.add(p)
                for r in read(p.relative_to(ROOT).as_posix()):cepochs.append(dict(origen=tag,modelo='CNN-B',repeticion=i,epoca=int(r['epoca']),val_accuracy=r['val_accuracy'],val_loss=r['val_loss']))
    curves=[]
    for name,raw in [('CNN-B',cepochs),('FC-A',fcepochs),('FC-B',fcepochs)]:
        for origin in ['CPU','CUDA']:
            for e in range(1,16):
                a=[r for r in raw if r['modelo']==name and r['origen']==origin and int(r['epoca'])==e]
                assert len(a)==5
                curves.append(dict(modelo=name,origen=origin,epoca=e,val_accuracy_pct=100*avg([float(r['val_accuracy']) for r in a]),val_loss=avg([float(r['val_loss']) for r in a])))
    data=dict(compare=compare,observations=observation,quality=quality,baseline=baseline,common=common,
        commonruns=commonruns,phasecompare=phasecompare,breakdown=breakdown,cnn_breakdown=cb,
        curves=curves,resources=resources,cnnresources=cnnresources,images=images,capturemeta=capturemeta)
    (OUT/'datos-presentacion.json').write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf8')
    with (OUT/'fuentes.csv').open('x',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['ruta_relativa','bytes','sha256']);writer.writeheader()
        for p in sorted(sources):writer.writerow(dict(ruta_relativa=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    print('DATOS PREPARADOS',len(sources),'fuentes; sin entrenamientos.')
if __name__=='__main__':main()
