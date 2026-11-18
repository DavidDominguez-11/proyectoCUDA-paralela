"""Render an existing DOCX using local Word/XPS/PNG. Never create a PDF."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import win32com.client
import pymupdf


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--entrada',type=Path,required=True)
    parser.add_argument('--destino',type=Path,required=True)
    args=parser.parse_args()
    source=args.entrada.resolve()
    folder=args.destino.resolve()
    folder.mkdir(parents=True,exist_ok=False)
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    word=win32com.client.DispatchEx('Word.Application')
    word.Visible=False
    word.DisplayAlerts=0
    opened=None
    try:
        opened=word.Documents.Open(str(source),ReadOnly=True,AddToRecentFiles=False,Visible=False)
        opened.Repaginate()
        headings=[]
        for paragraph in opened.Paragraphs:
            text=paragraph.Range.Text.strip('\r\x07\x0c')
            if paragraph.OutlineLevel<=2 and text:
                headings.append(dict(texto=text,pagina=paragraph.Range.Information(3)))
        tables=[]
        for i,table in enumerate(opened.Tables,1):
            start=opened.Range(table.Range.Start,table.Range.Start+1).Information(3)
            end=opened.Range(table.Range.End-1,table.Range.End).Information(3)
            tables.append(dict(tabla=i,pagina_inicio=start,pagina_fin=end))
        xps=folder/'revision.xps'
        opened.ExportAsFixedFormat(str(xps),18,OpenAfterExport=False)
        info=dict(fecha_utc=datetime.now(timezone.utc).isoformat(),comando=[sys.executable,*sys.argv],
                  entrada=str(source),sha256=before,word_version=word.Version,
                  paginas=opened.ComputeStatistics(2),encabezados=headings,tablas=tables,pdf_generado=False)
    finally:
        if opened is not None:opened.Close(SaveChanges=False)
        word.Quit()
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before,'Read-only source changed'
    rendered=pymupdf.open(folder/'revision.xps')
    extracted=[]
    for i,page in enumerate(rendered,1):
        pix=page.get_pixmap(matrix=pymupdf.Matrix(1.2,1.2),alpha=False)
        pix.save(folder/f'pagina-{i:02}.png')
        extracted.append(f'PAGINA {i}\n'+page.get_text())
    info['paginas_renderizadas']=len(rendered)
    (folder/'texto-maquetado.txt').write_text('\n\n'.join(extracted),encoding='utf8')
    (folder/'revision.json').write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps(info,ensure_ascii=False))


if __name__=='__main__':main()
