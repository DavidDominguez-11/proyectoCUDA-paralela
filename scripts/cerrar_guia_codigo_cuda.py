"""Register the verified static walkthrough and seal its artifacts."""
from pathlib import Path
from datetime import datetime,timezone
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[1]
QA=ROOT/'resultados/preparacion/guia-codigo-cuda-v01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def append(p,text):
    with (ROOT/p).open('a',encoding='utf8') as f:f.write('\n\n'+text.strip()+'\n')
def main():
    v=json.loads((QA/'validacion.json').read_text(encoding='utf8'));assert v['estado']=='aprobado'
    assert sha(ROOT/'entregables/Guia_Codigo_CUDA.md')==v['guia_sha256']
    initial=json.loads((QA/'identidades-previas.json').read_text(encoding='utf8'))
    assert all(sha(ROOT/p)==r['sha256'] for p,r in initial.items())
    doc='''# Guía para demostrar el código CUDA

Entrega: [Guia_Codigo_CUDA.md](../../entregables/Guia_Codigo_CUDA.md). DEC-030, CAM-015, BIT-017.

Documento técnico para exposición: ruta rápida de seis paradas,17apartados,guion literal,archivo/función/líneas,búsquedas del notebook ypreguntas frecuentes. La sección6distingue barrera de bloque,orden de operaciones de un stream yespera del host. Se explica memoria pinned/device/shared,grid-stride,BN,softmax,backward,Adam,banderas atómicas,cuBLAS/im2col,FC-A/FC-B,warmup ymedición. Se identifica la ruta cuBLAS realmente medida frente albackend explícito alternativo.

La información se obtuvo leyendo las fuentes actuales,analizando AST yleyendo el JSON del notebook. Se consultaron páginas oficiales de NVIDIA Numba-CUDA para significado de API ycuBLAS13.1 para stream ySGEMM. No se ejecutaron kernels,modelos,benchmarks ni celdas. El documento separa comportamiento observado en código,mediciones históricas,derivaciones ypropuestas futuras;no convierte la inspección en certificación de nueva ejecución GPU.

Validación:104enlaces locales,82anclas de línea,45ubicaciones de celdas y5bloquesPython de sintaxis válida. Funciones/clases contrastadas con rangos AST;hashes de fuentes yentregables previos iguales. Se verificaron específicamente la activación `backend="cublas"` del controlador,la captura de momentum enFC,la creación delstream yasociación cuBLAS. Se corrigieron dos referencias puntuales antes del cierre:controlador178ymomentumFC110. El índice delnotebook cuenta celdas desde1,incluyendoMarkdown;contadores deejecución reiniciados por sesión se muestran aparte.

Artefactos: [fuentes ySHA-256](../../resultados/preparacion/guia-codigo-cuda-v01/fuentes.csv),[símbolos](../../resultados/preparacion/guia-codigo-cuda-v01/simbolos.json),[índice de notebook](../../resultados/preparacion/guia-codigo-cuda-v01/indice-notebook.json),[validación](../../resultados/preparacion/guia-codigo-cuda-v01/validacion.json),[identidades previas](../../resultados/preparacion/guia-codigo-cuda-v01/identidades-previas.json).

Comandos de documentación,desde la raíz:

```powershell
.\PCUDA\Scripts\python.exe -X utf8 scripts/crear_guia_codigo_cuda.py
.\PCUDA\Scripts\python.exe -X utf8 scripts/verificar_guia_codigo_cuda.py --pulir
.\PCUDA\Scripts\python.exe -X utf8 scripts/cerrar_guia_codigo_cuda.py
.\PCUDA\Scripts\python.exe -X utf8 scripts/verificar_preparacion.py
```

El creador rechaza destinos existentes;no repetirlo sobre la misma carpeta. El verificador hace análisis estático,comprueba hashes ypuede ajustar el formato del documento con --pulir;no importa la implementación CUDA. Los snippets solo añaden espaciado yel decorador visible para exponer;no se modifican fuentes. No se tocaron losPPTX,PDF,DOCX,notebooks niproyectos experimentales.
'''
    (ROOT/'docs/trazabilidad/guia-codigo-cuda.md').write_text(doc,encoding='utf8')
    now=datetime.now(timezone.utc).isoformat()
    append('docs/trazabilidad/bitacora.md',f'''## Cierre BIT-017 — {now}

Guía de código CUDA creada con rutas,funciones ylíneas,guion literal,ruta rápida,sincronización yFAQ. Validación estática aprobada:104enlaces,82anclas,45celdas y5snippetsPython. Fuentes ytodoslosentregables previos intactos porSHA-256. Sin GPU/entrenamientos nuevos. [Detalle](guia-codigo-cuda.md).''')
    append('docs/trazabilidad/cambios.md','## Cierre CAM-015\n\nAñadida guíaMD eíndices/validación estática. No se modificó códigoCUDA,notebooks,informes nipresentaciones. [Detalle](guia-codigo-cuda.md).')
    append('docs/trazabilidad/decisiones.md','## Cierre DEC-030\n\nGuía basada en fuentes actuales,con distinción entrebarreras/stream/espera host,ruta medida yalternativa,mapeo de notebook yguionparaexponer. [Entrega](guia-codigo-cuda.md).')
    append('docs/trazabilidad/README.md','## Guía para defender el código\n\n[Qué mostrar yqué decir sobre CUDA](guia-codigo-cuda.md):ubicaciones exactas,sincronización,memoria,kernels,cuBLAS,FC ymedición;sin ejecutar modelos.')
    append('README.md','[Guía de exposición del código CUDA](entregables/Guia_Codigo_CUDA.md):qué enseñar,qué decir yarchivo/función/línea exactos.')
    append('entregables/README.md','## Guía para mostrar código\n\n[Guia_Codigo_CUDA.md](Guia_Codigo_CUDA.md):ruta rápida,ubicaciones exactas yguion literal,con atención especial asincronización,streams,memoria,kernels,cuBLAS yFC.')
    paths={ROOT/'entregables/Guia_Codigo_CUDA.md',ROOT/'docs/trazabilidad/guia-codigo-cuda.md'}
    paths.update(p for p in QA.rglob('*') if p.is_file())
    paths.update([ROOT/'scripts/crear_guia_codigo_cuda.py',ROOT/'scripts/verificar_guia_codigo_cuda.py',Path(__file__).resolve()])
    seal=QA/'artefactos.csv';paths.discard(seal)
    with seal.open('x',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['ruta_relativa','bytes','sha256']);writer.writeheader()
        for p in sorted(paths):writer.writerow(dict(ruta_relativa=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    print('CIERRE',len(paths),'artefactos; guía yreferencias verificadas;sinentrenamientos.')
if __name__=='__main__':main()
