"""Static check of the guide's source anchors, notebook map and untouched files."""
from pathlib import Path
import ast,csv,hashlib,json,re,textwrap
from urllib.parse import unquote
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'entregables/Guia_Codigo_CUDA.md'
QA=ROOT/'resultados/preparacion/guia-codigo-cuda-v01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    import sys
    if '--pulir' in sys.argv:
        # This module only reads/parses source; it never imports training code.
        from crear_guia_codigo_cuda import format_guide
        content=OUT.read_text(encoding='utf8')
        content=content.replace('```python\ndef dropout','```python\n@cuda.jit(cache=True)\ndef dropout')
        content=format_guide(content)
        lines=content.splitlines()
        for i,line in enumerate(lines):
            if line.startswith('> «La contribución no es únicamente'):
                lines[i]='> «La contribución no es únicamente activar una GPU. Separamos preparación CPU y entrenamiento GPU, definimos kernels y su jerarquía, controlamos memoria y dependencias, unificamos el stream con cuBLAS y medimos el coste completo, incluidas transferencias y validación. La espera por lote hace que los resultados y los tiempos se lean después de terminar el trabajo. Es una implementación verificable para tres redes, con los límites numéricos y de calidad declarados».'
        OUT.write_text('\n'.join(lines)+'\n',encoding='utf8')
    text=OUT.read_text(encoding='utf8')
    plain=re.sub(r'```[\s\S]*?```','',text)
    links=[];anchors=[]
    for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',text):
        if '://' in target or target.startswith('#'):continue
        rel,_,fragment=unquote(target).partition('#')
        path=(OUT.parent/rel).resolve();assert path.is_file(),target
        links.append(target)
        if re.fullmatch(r'L\d+',fragment):
            line=int(fragment[1:]);src=path.read_text(encoding='utf8').splitlines()
            assert 1<=line<=len(src),(target,len(src))
            anchors.append(dict(archivo=path.relative_to(ROOT).as_posix(),linea=line,contenido=src[line-1]))
    for rel,start,end in re.findall(r'`([^`]+\.py):\d+`, líneas (\d+)–(\d+)',text):
        tree=ast.parse((ROOT/rel).read_text(encoding='utf8'))
        assert any(isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.lineno==int(start) and n.end_lineno==int(end) for n in ast.walk(tree)),(rel,start,end)
    examples=re.findall(r'```python\n(.*?)```',text,flags=re.S)
    for sample in examples:ast.parse(textwrap.dedent(sample))
    required={('scripts/benchmark_cuda.py',178):'backend="cublas"',('src/asl_cuda/fc.py',110):'momentum=float',
        ('src/asl_cuda/model.py',82):'cuda.stream()',('src/asl_cuda/blas.py',28):'cublasSetStream_v2'}
    for (rel,line),term in required.items():
        assert term in (ROOT/rel).read_text(encoding='utf8').splitlines()[line-1]
        assert any(r['archivo']==rel and r['linea']==line for r in anchors)
    for row in csv.DictReader((QA/'fuentes.csv').open(encoding='utf8',newline='')):assert sha(ROOT/row['ruta_relativa'])==row['sha256']
    prior=json.loads((QA/'identidades-previas.json').read_text(encoding='utf8'))
    assert all(sha(ROOT/p)==row['sha256'] for p,row in prior.items())
    nb=json.loads((ROOT/'entregables/notebooks/ASL_CUDA.ipynb').read_text(encoding='utf8'))
    cells=json.loads((QA/'indice-notebook.json').read_text(encoding='utf8'))
    for row in cells:
        c=nb['cells'][row['celda_visual']-1]
        assert c.get('id')==row['id'] and c.get('execution_count')==row['execution_count']
        assert ''.join(c['source']).splitlines()[0]==row['primera_linea']
    assert '|\n\n|' not in text,'Markdown table interrupted by empty rows'
    assert all(term in text for term in ['cuda.syncthreads()','self.stream.synchronize()','cublasSetStream_v2','backend="cublas"','FCNetwork.reduce'])
    report=dict(estado='aprobado',fecha_utc=datetime.now(timezone.utc).isoformat(),guia_sha256=sha(OUT),
        enlaces_locales=len(links),anclas_linea=len(anchors),bloques_python_sintaxis_valida=len(examples),
        ubicaciones_notebook_verificadas=len(cells),fuentes_sha256_sin_cambios=True,
        presentaciones_informes_notebooks_inalterados=True,entrenamientos_ejecutados=0,
        limite='Inspección estática y contraste de documentación API;no certifica una nueva ejecución GPU.')
    (QA/'validacion.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
