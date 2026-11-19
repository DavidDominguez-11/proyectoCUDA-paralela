"""Validate OOXML and render with real local PowerPoint. No experiment execution."""
from pathlib import Path
from datetime import datetime,timezone
import json,zipfile,hashlib,posixpath,re,sys
from urllib.parse import unquote
from lxml import etree
from pptx import Presentation
from PIL import Image,ImageDraw
import win32com.client
ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'entregables/Presentacion_Proyecto_3_CUDA_Corta.pptx'
NS={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','p':'http://schemas.openxmlformats.org/presentationml/2006/main',
 'c':'http://schemas.openxmlformats.org/drawingml/2006/chart','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--revision',default='revision-01');args=ap.parse_args()
    out=ROOT/'resultados/preparacion/presentacion-corta-v01'/args.revision
    out.mkdir(parents=True,exist_ok=False)
    issues=[];files=[]
    before=sha(TARGET)
    with zipfile.ZipFile(TARGET) as z:
        assert z.testzip() is None
        names=set(z.namelist());charts=[];slides=[];notes=[]
        for name in sorted(names):
            if name.endswith('.xml') or name.endswith('.rels'):
                try:root=etree.fromstring(z.read(name))
                except etree.XMLSyntaxError as e:issues.append([name,str(e)]);continue
                if name.endswith('.rels'):
                    base=posixpath.dirname(posixpath.dirname(name))
                    for rel in root:
                        if rel.get('TargetMode')=='External':continue
                        target=unquote(rel.get('Target','')).replace('\\','/')
                        resolved=target.lstrip('/') if target.startswith('/') else posixpath.normpath(posixpath.join(base,target))
                        if resolved not in names:issues.append([name,'Missing relationship '+resolved])
                for color in root.xpath('//*[local-name()="srgbClr"]/@val'):
                    if not re.fullmatch(r'[0-9A-Fa-f]{6}',color):issues.append([name,'Invalid hexadecimal '+color])
                if re.fullmatch(r'ppt/charts/chart\d+\.xml',name):
                    charts.append(name)
                    axisdefs=set(root.xpath('//c:catAx/c:axId/@val | //c:valAx/c:axId/@val | //c:dateAx/c:axId/@val | //c:serAx/c:axId/@val',namespaces=NS))
                    refs=set(root.xpath('//c:barChart/c:axId/@val | //c:lineChart/c:axId/@val | //c:scatterChart/c:axId/@val',namespaces=NS))
                    if not refs.issubset(axisdefs):issues.append([name,'Undeclared chart axes'])
                    stacked=root.xpath('//c:barChart/c:grouping/@val',namespaces=NS)
                    if any(x in ['stacked','percentStacked'] for x in stacked):
                        if root.xpath('//c:dLblPos[@val="outEnd"]',namespaces=NS):issues.append([name,'Invalid stacked label position'])
                if re.fullmatch(r'ppt/slides/slide\d+\.xml',name):slides.append(name)
                if re.fullmatch(r'ppt/notesSlides/notesSlide\d+\.xml',name):notes.append(name)
    assert not issues,issues
    deck=Presentation(TARGET)
    extracted=[];offslide=[]
    for i,slide in enumerate(deck.slides,1):
        text=[]
        for shape in slide.shapes:
            if shape.has_text_frame:text.append(shape.text)
            if shape.has_table:
                text.extend(' | '.join(c.text for c in row.cells) for row in shape.table.rows)
            if shape.has_chart:
                text.append('GRÁFICA NATIVA: '+str(shape.chart.chart_type))
                for series in shape.chart.series:text.append(series.name+': '+repr(list(series.values)))
            if shape.left < -9144 or shape.top < -9144 or shape.left+shape.width>deck.slide_width+18288 or shape.top+shape.height>deck.slide_height+18288:offslide.append(dict(slide=i,shape=shape.name))
        text.append('NOTAS: '+slide.notes_slide.notes_text_frame.text)
        extracted.append('<!-- Slide number: '+str(i)+' -->\n'+'\n'.join(text))
    assert len(deck.slides)==15 and len(notes)==15 and len(charts)>=7
    assert not offslide,offslide
    (out/'texto-diapositivas.md').write_text('\n\n'.join(extracted),encoding='utf8')
    # The user has edited DOCX/PDF independently: compare against THIS task's identities.
    initial=json.loads((out.parent/'identidades-previas.json').read_text(encoding='utf8'))
    assert all(sha(ROOT/p)==v['sha256'] for p,v in initial.items()),'Prior user documents changed'
    app=win32com.client.DispatchEx('PowerPoint.Application');opened=None
    overflows=[];native=[]
    try:
        app.DisplayAlerts=1
        opened=app.Presentations.Open(str(TARGET.resolve()),ReadOnly=True,Untitled=False,WithWindow=False)
        assert opened.Slides.Count==15
        for i in range(1,opened.Slides.Count+1):
            s=opened.Slides(i)
            s.Export(str((out/f'diapositiva-{i:02d}.png').resolve()),'PNG',1600,900)
            for shape in s.Shapes:
                try:
                    if shape.HasTextFrame and shape.TextFrame.HasText:
                        tf=shape.TextFrame2;tr=tf.TextRange
                        available=shape.Height-tf.MarginTop-tf.MarginBottom
                        if tr.BoundHeight>available+3:overflows.append(dict(slide=i,text=shape.TextFrame.TextRange.Text[:90],bound_pt=tr.BoundHeight,available_pt=available))
                except Exception:pass
            native.append(dict(numero=i,formas=s.Shapes.Count))
    finally:
        if opened is not None:opened.Close()
        if app.Presentations.Count==0:app.Quit()
    assert sha(TARGET)==before
    for first in range(1,16,8):
        subset=list(range(first,min(first+8,16)))
        canvas=Image.new('RGB',(1600,2*480),(222,229,234));draw=ImageDraw.Draw(canvas)
        for j,i in enumerate(subset):
            im=Image.open(out/f'diapositiva-{i:02d}.png');im.thumbnail((390,440))
            x=(j%4)*400;y=(j//4)*480+28;canvas.paste(im,(x,y));draw.text((x+8,y-22),str(i),fill='black')
        canvas.save(out/f'contacto-{first:02d}.png')
    report=dict(estado='aprobado' if not overflows else 'revisar_texto',fecha_utc=datetime.now(timezone.utc).isoformat(),
        comando=[sys.executable,*sys.argv],pptx_sha256=before,diapositivas=15,graficas_nativas=len(charts),notas=len(notes),
        relaciones_xml_validas=True,colores_validos=True,ejes_validos=True,sin_objetos_fuera=True,
        powerpoint_abierto_y_renderizado=True,diapositivas_renderizadas=len(native),posibles_desbordes=overflows,
        documentos_previos_inalterados=True,pdf_generado=False)
    (out/'validacion.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
