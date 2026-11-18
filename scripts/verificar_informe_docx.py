"""Validate report structure, embedded evidence and Word layout without a model."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile
from docx import Document

ROOT=Path(__file__).resolve().parents[1]
TITLES=[
    '1. Análisis de la solución secuencial existente',
    '2. Medición del tiempo de ejecución secuencial',
    '3. Identificación de oportunidades de paralelización',
    '4. Diseño del algoritmo paralelo',
    '5. Implementación con CUDA Python',
    '6. Análisis y conclusiones',
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--carpeta',type=Path,required=True)
    parser.add_argument('--revision',type=Path,required=True)
    parser.add_argument('--salida',type=Path,required=True)
    args=parser.parse_args()
    if args.salida.exists():raise FileExistsError(args.salida)
    folder=args.carpeta.resolve()
    file=folder/'Informe_Proyecto_3_CUDA.docx'
    doc=Document(file)
    text='\n'.join(p.text for p in doc.paragraphs)
    headings=[p.text for p in doc.paragraphs if p.style.name=='Heading 1']
    assert headings==TITLES,headings
    assert len(doc.tables)==12 and len(doc.inline_shapes)==7
    for part in ('David Domínguez · 23712','Jonathan Diaz · 23873','Luis Padilla · 23663'):
        assert part in text
    assert not any(term in text for term in ('docs/','docs\\','resultados/','.md','C:\\Users'))
    questions=['¿Por qué procesar imágenes/archivos con varios núcleos CPU y delegar la matemática a GPU?',
               '¿Cómo CUDA permitió reducir o cambiar el tiempo de procesamiento?',
               '¿Qué factores justifican o limitan la utilización de CUDA?']
    assert all(question in text for question in questions)
    assert all(value in text for value in ('1.924593739','387.042','48.041','saturación total','no se midió'))
    with zipfile.ZipFile(file) as z:
        assert z.testzip() is None
        embedded={hashlib.sha256(z.read(name)).hexdigest() for name in z.namelist() if name.startswith('word/media/')}
    figures=sorted((folder/'figuras').glob('*.png'))
    assert {digest(p) for p in figures}==embedded and len(embedded)==7
    count=0
    with (folder/'fuentes.csv').open(encoding='utf8',newline='') as source:
        for row in csv.DictReader(source):
            path=ROOT/row['ruta_relativa']
            assert path.stat().st_size==int(row['bytes']) and digest(path)==row['sha256'],str(path)
            count+=1
    revision=json.loads(args.revision.read_text(encoding='utf8'))
    assert revision['sha256']==digest(file)
    assert revision['paginas']==revision['paginas_renderizadas']==13
    assert all(t['pagina_inicio']==t['pagina_fin'] for t in revision['tablas'])
    for label in ('cpu','gpu'):
        suffix='20261117T233947Z-cpu.png' if label=='cpu' else '20261117T233706Z-gpu.png'
        original=ROOT/'resultados/evidencias/EXP-003'/suffix
        assert digest(original) in embedded
    assert not list(folder.rglob('*.pdf'))
    outcome=dict(estado='aprobado',fecha_utc=datetime.now(timezone.utc).isoformat(),
                 docx_sha256=digest(file),secciones_ordenadas=headings,tablas=12,figuras_incrustadas=7,
                 fuentes_verificadas=count,paginas_word=13,tablas_cortadas=0,
                 capturas_originales_sin_alterar=True,sin_rutas_internas_en_cuerpo=True,
                 autores_confirmados=True,tres_preguntas_tecnicas=True,pdf_generado=False,
                 entrenamiento_ejecutado=False,
                 limites='Uso real acreditado; saturacion literal/ocupacion SM no demostradas. CPU historica y RNG diferentes.',
                 script_sha256=digest(Path(__file__)))
    args.salida.parent.mkdir(parents=True,exist_ok=True)
    args.salida.write_text(json.dumps(outcome,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps(outcome,ensure_ascii=False))


if __name__=='__main__':main()
