"""Create a shorter COPY of the current user-edited PowerPoint; no training."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,sys
import win32com.client
from pptx import Presentation
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'entregables/Presentacion_Proyecto_3_CUDA.pptx'
TARGET=ROOT/'entregables/Presentacion_Proyecto_3_CUDA_Corta.pptx'
OUT=ROOT/'entregables/presentacion-corta'
QA=ROOT/'resultados/preparacion/presentacion-corta-v01'
KEEP=[1,3,5,6,7,8,9,11,12,14,15,17,20,23,26]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    if '--finalizar' in sys.argv:
        initial=json.loads((QA/'identidades-previas.json').read_text(encoding='utf8'))
        return finalize(initial)
    if TARGET.exists():raise FileExistsError(TARGET)
    OUT.mkdir(exist_ok=False);QA.mkdir(parents=True,exist_ok=False)
    initial={}
    for rel in ['entregables/Presentacion_Proyecto_3_CUDA.pptx','entregables/Informe_Proyecto_3_CUDA.docx',
        'entregables/InformeFinal_Proyecto_3_CUDA.pdf','entregables/notebooks/ASL_Secuencial.ipynb','entregables/notebooks/ASL_CUDA.ipynb']:
        p=ROOT/rel
        if p.exists():initial[rel]=dict(sha256=sha(p),bytes=p.stat().st_size)
    (QA/'identidades-previas.json').write_text(json.dumps(initial,indent=2),encoding='utf8')
    for rel,title,text in [
        ('decisiones.md','DEC-029 — Versión corta independiente',
         'El usuario solicita una segunda presentación más corta y conservar intacta la actual. Seleccionar15diapositivas esenciales de la versión actual editada por el usuario, preservando su estilo, tablas/gráficas nativas, capturas ynotas. Sin regenerar ni sobrescribir la presentación larga. Mantener los seis pasos ypriorizar resultados;sin experimentos nuevos.'),
        ('cambios.md','CAM-014 — Copia corta de15diapositivas',
         'Salida nueva Presentacion_Proyecto_3_CUDA_Corta.pptx,guion yvista previa propios. Identidades previas delPPTX31actual,PDF/DOCXynotebooks registradas;no se modifican. Copia creada yrecortada mediante PowerPointlocal para preservar las ediciones delusuario.'),
        ('bitacora.md','BIT-016 — Preparación de presentación corta',
         'InicioUTC:'+datetime.now(timezone.utc).isoformat()+'. RevisadosREADME,trazabilidad,decisiones,bitácora yhabilidad localSKILL-P.md. Origen:presentación actual de31diapositivas,SHA-256:'+sha(SOURCE)+'. Se conservarán exactamente15diapositivas: '+','.join(map(str,KEEP))+'. Ningún modelo ejecutado.')]:
        with (ROOT/'docs/trazabilidad'/rel).open('a',encoding='utf8') as f:f.write('\n\n## '+title+'\n\n'+text+'\n')
    app=win32com.client.DispatchEx('PowerPoint.Application');opened=None
    try:
        app.DisplayAlerts=1
        opened=app.Presentations.Open(str(SOURCE),ReadOnly=True,Untitled=False,WithWindow=False)
        assert opened.Slides.Count==31
        opened.SaveCopyAs(str(TARGET),24)
        opened.Close();opened=None
        assert sha(SOURCE)==initial['entregables/Presentacion_Proyecto_3_CUDA.pptx']['sha256']
        opened=app.Presentations.Open(str(TARGET),ReadOnly=False,Untitled=False,WithWindow=False)
        for i in range(31,0,-1):
            if i not in KEEP:opened.Slides(i).Delete()
        assert opened.Slides.Count==15
        # Update slide-number fields in the new copy through native PowerPoint.
        for i in range(1,16):
            try:opened.Slides(i).HeadersFooters.SlideNumber.Visible=True
            except Exception:pass
        opened.Save();opened.Close();opened=None
    finally:
        if opened is not None:opened.Close()
        if app.Presentations.Count==0:app.Quit()
    assert all(sha(ROOT/p)==v['sha256'] for p,v in initial.items())
    return finalize(initial)
def finalize(initial):
    assert all(sha(ROOT/p)==v['sha256'] for p,v in initial.items())
    original=Presentation(SOURCE);short=Presentation(TARGET)
    mapping=[]
    for new,old in enumerate(KEEP,1):
        slide=short.slides[new-1];before=original.slides[old-1]
        title=next(s.text for s in slide.shapes if s.has_text_frame and s.text.strip())
        before_text=[s.text for s in before.shapes if s.has_text_frame and not (s.is_placeholder and s.placeholder_format.type==13)]
        after_text=[s.text for s in slide.shapes if s.has_text_frame and not (s.is_placeholder and s.placeholder_format.type==13)]
        assert before_text==after_text,(new,old,'User slide text changed')
        assert len([s for s in before.shapes if s.has_chart])==len([s for s in slide.shapes if s.has_chart])
        notes=slide.notes_slide.notes_text_frame.text.strip()
        notes=notes.replace('Abrir preguntas y usar los anexos para los detalles.','Abrir preguntas y consultar la presentación completa para los detalles.')
        mapping.append(dict(numero=new,original=old,titulo=title,notas=notes))
    result=dict(origen=SOURCE.relative_to(ROOT).as_posix(),origen_sha256=sha(SOURCE),
        salida=TARGET.relative_to(ROOT).as_posix(),salida_sha256=sha(TARGET),seleccion=KEEP,diapositivas=mapping,
        graficas=sum(s.has_chart for slide in short.slides for s in slide.shapes),
        tablas=sum(s.has_table for slide in short.slides for s in slide.shapes),origen_inalterado=True)
    (OUT/'estructura-y-notas.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
    (OUT/'Guion_corto.md').write_text('# Guion de la presentación corta\n\n15diapositivas. Se conserva la versión completa de31sin modificar. Las notas del expositor están también en elPowerPoint;este guion añade el mapa con la versión larga para consultas.\n\n'+
        '\n'.join('## '+str(r['numero'])+'. '+r['titulo']+'\n\n'+r['notas']+'\n\nDiapositiva de referencia en versión completa: '+str(r['original'])+'.\n' for r in mapping),encoding='utf8')
    print('CREADA',len(short.slides),'diapositivas;',result['graficas'],'gráficas;',result['tablas'],'tablas. Presentación original intacta.')
if __name__=='__main__':main()
