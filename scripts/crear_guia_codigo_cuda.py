"""Static source walkthrough and speaking script. Never import CUDA/model code."""
from pathlib import Path
from datetime import datetime,timezone
import ast,json,hashlib,csv,re

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'entregables/Guia_Codigo_CUDA.md'
QA=ROOT/'resultados/preparacion/guia-codigo-cuda-v01'
MODEL='src/asl_cuda/model.py';KERNEL='src/asl_cuda/kernels.py';FC='src/asl_cuda/fc.py';BLAS='src/asl_cuda/blas.py';RUN='scripts/benchmark_cuda.py';RUNTIME='src/asl_cuda/runtime.py'
SOURCES={p.relative_to(ROOT).as_posix():p.read_text(encoding='utf8') for p in [*sorted((ROOT/'src/asl_cuda').glob('*.py')),ROOT/RUN,ROOT/'scripts/validar_cuda_fc.py']}
INDEX={}
for path,source in SOURCES.items():
    tree=ast.parse(source)
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
            INDEX[(path,node.name)]=(node.lineno,node.end_lineno)
            if isinstance(node,ast.ClassDef):
                for child in node.body:
                    if isinstance(child,ast.FunctionDef):INDEX[(path,node.name+'.'+child.name)]=(child.lineno,child.end_lineno)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def loc(path,name,label=None):
    start,end=INDEX[(path,name)]
    return f'[{label or name}](../{path}#L{start}) — `{path}:{start}`, líneas {start}–{end}'
def spot(path,line):return f'[{path}:{line}](../{path}#L{line})'
def code(path,name,start_offset=0,max_lines=None):
    start,end=INDEX[(path,name)];start+=start_offset
    if start_offset==0 and start>1 and SOURCES[path].splitlines()[start-2].lstrip().startswith('@'):
        start-=1
    if max_lines is not None:end=min(end,start+max_lines-1)
    text='\n'.join(SOURCES[path].splitlines()[start-1:end])
    # Whitespace before argument parentheses is valid Python. It also avoids
    # treating kernel launch indexing as a Markdown link in the repo checker.
    return '```python\n'+text.replace('](','] (')+'\n```'
def format_guide(content):
    content=re.sub(r'\|\n\n(?=\|)','|\n',content)
    # Preserve code, links and identifiers; improve spacing only in prose.
    protected=r'(```[\s\S]*?```|`[^`\n]*`|\[[^\]]*\]\([^)]+\)|https?://[^\s)]+|\b(?:float32|int32|int64|uint8|im2col|col2im|H2D|D2H|FP32|SGEMM|PCG64)\b)'
    parts=re.split(protected,content)
    for i in range(0,len(parts),2):
        text=parts[i]
        text=re.sub(r'(?<=\d)(?=[a-záéíóúñ])',' ',text)
        text=re.sub(r'(?<=[a-záéíóúñ])(?=\d)',' ',text)
        for a,b in [('MiB',' MiB'),('KiB',' KiB'),('CPUhistórica','CPU histórica'),('CPU/GPUidéntica','CPU/GPU idéntica'),
            ('enGPU','en GPU'),('enCPU','en CPU'),('lasFC','las FC'),('cuBLASes','cuBLAS es'),('delCPU','del CPU'),
            ('laGPU','la GPU'),('toda laGPU','toda la GPU'),('porSM','por SM'),('RNGdistinto','RNG distinto')]:text=text.replace(a,b)
        for _ in range(3):
            text=re.sub(r'(?<=[a-záéíóúñ])(?=(?:CPU|GPU|CUDA|BN|Adam|SGEMM|cuBLAS|TensorFlow)\b)',' ',text)
            text=re.sub(r'\b(CPU|GPU|CUDA|BN|SGEMM|cuBLAS)(?=[a-záéíóúñ])',r'\1 ',text)
            text=re.sub(r'\b(y|de|la|el|los|las|en|del|al|sin|con|para|una|un|por)(?=(?:CPU|GPU|cuBLAS|CUDA|stream|kernels?|hilos|bloques|máscaras|gradientes|pesos|contador|primera|precisión|calidad|validación|lectura|matemática|transferencias|trayectorias|datos|tamaño|último|óptimo|cuello|memoria|bandera|código|función|rutas|trabajo|speedup|modelos|capas|arreglos|saltos|device|host|mismo|siguiente|reporte|notebook|tiempos|lotes?|su|sus|ese|esas|nuestra)\b)',r'\1 ',text)
        for a,b in [('ycontador','y contador'),('ydtype','y dtype'),('yprimera','y primera'),('yprecisión','y precisión'),
            ('ylas','y las'),('yla','y la'),('ysu','y su'),('yvalidación','y validación'),('ygráficas','y gráficas'),
            ('ydocumentación','y documentación'),('resultadosguardados','resultados guardados'),('paraerrores','para errores'),
            ('sinatomics','sin atomics'),('laaccuracy','la accuracy'),('LaGPUestuvo','La GPU estuvo'),
            ('deactividad','de actividad'),('enesa','en esa'),('GPUestuvo','GPU estuvo'),('testCPU','test CPU'),
            ('CPUhost','CPU host'),('porlote','por lote'),('eltamaño','el tamaño'),('elequipo','el equipo'),
            ('losbloques','los bloques'),('loshilos','los hilos'),('deun','de un'),('enel','en el'),('dela','de la')]:text=text.replace(a,b)
        text=re.sub(r'([,;:])(?=[A-Za-zÁÉÍÓÚÑáéíóúñ])',r'\1 ',text)
        for a,b in [('float 32','float32'),('int 32','int32'),('int 64','int64'),('uint 8','uint8'),('im 2 col','im2col'),('col 2 im','col2im'),('  MiB',' MiB'),('  KiB',' KiB')]:text=text.replace(a,b)
        parts[i]=text
    return ''.join(parts)
def main():
    if OUT.exists():raise FileExistsError(OUT)
    QA.mkdir(parents=True,exist_ok=False)
    prior={}
    for rel in ['entregables/Presentacion_Proyecto_3_CUDA.pptx','entregables/Presentacion_Proyecto_3_CUDA_Corta.pptx',
        'entregables/Informe_Proyecto_3_CUDA.docx','entregables/InformeFinal_Proyecto_3_CUDA.pdf',
        'entregables/notebooks/ASL_Secuencial.ipynb','entregables/notebooks/ASL_CUDA.ipynb']:
        p=ROOT/rel
        if p.exists():prior[rel]=dict(bytes=p.stat().st_size,sha256=sha(p))
    (QA/'identidades-previas.json').write_text(json.dumps(prior,indent=2),encoding='utf8')
    nb=json.loads((ROOT/'entregables/notebooks/ASL_CUDA.ipynb').read_text(encoding='utf8'))
    cells=[]
    for i,c in enumerate(nb['cells'],1):
        text=''.join(c.get('source',[]))
        if c['cell_type']!='code':continue
        tags=[]
        for term,label in [('%%modulo asl_cuda.runtime','Módulo runtime CNN-B'),('%%modulo asl_cuda.kernels','Kernels CNN-B'),
            ('%%modulo asl_cuda.blas','Adaptador cuBLAS CNN-B'),('%%modulo asl_cuda.model','Clase Network CNN-B'),
            ('%%modulo_fc asl_cuda.fc','Extensión FCNetwork'),('%%modulo_fc asl_cuda.kernels','Kernels reutilizados FC'),
            ('%%modulo_fc asl_cuda.model','Network reutilizada FC'),('%%modulo_fc asl_cuda.blas','cuBLAS reutilizado FC')]:
            if text.startswith(term):tags.append(label)
        if 'train_start' in text and 'for epoch in range(15)' in text:tags.append('Bucle real de entrenamiento')
        if "state('preparacion_cpu'" in text or 'state("preparacion_cpu"' in text:tags.append('Preparación CPU')
        if 'net=Network(reference,backend=' in text or 'net = Network(reference, backend=' in text:tags.append('Creación Network, backend medido')
        if 'net=FCNetwork(reference,seed)' in text:tags.append('Creación FCNetwork / warmup')
        if tags:cells.append(dict(celda_visual=i,execution_count=c.get('execution_count'),id=c.get('id'),primera_linea=text.splitlines()[0],temas=tags))
    (QA/'indice-notebook.json').write_text(json.dumps(cells,indent=2,ensure_ascii=False),encoding='utf8')
    now=datetime.now(timezone.utc).isoformat()
    for rel,title,body in [
        ('decisiones.md','DEC-030 — Guía técnica para defender el código CUDA','Crear Markdown con localización exacta por archivo/función/líneas ycelda delnotebook;guionliteral para exponer,prioridad asincronización,streams,transferencias,memoria,jerarquía ybackend realmente medido. Inspección estática ycontraste condocumentación oficial;sinentrenar ni modificarel código.'),
        ('cambios.md','CAM-015 — Guía de exposición del código','Añadir entregables/Guia_Codigo_CUDA.md yfuentes/validación en resultados/preparacion/guia-codigo-cuda-v01. Código,notebooks,informes ypresentaciones permanecenintactos;identidades previas conservadas.'),
        ('bitacora.md','BIT-017 — Inspección técnica de CUDA','InicioUTC:'+now+'. Responsable:Codexporpetición delusuario. Leídosruntime,kernels,blas,model,fc ycontrolador medido. Se consultaron las APIoficialesNumba-CUDA ycuBLAS13.1. No seimportaron módulos de entrenamiento ni se ejecutó códigoGPU. Se generará índice estáticoAST ymapa deceldas visuales.')]:
        with (ROOT/'docs/trazabilidad'/rel).open('a',encoding='utf8') as f:f.write('\n\n## '+title+'\n\n'+body+'\n')
    sections=[]
    def add(t):sections.append(t.strip())
    add(f'''# Guía para enseñar y defender el código CUDA

Guion de consulta para la exposición. Basado en **el código real del proyecto**, inspeccionado el {now[:10]}. Las ubicaciones incluyen archivo, función y línea de inicio para abrir con precisión. Las presentaciones y los entrenamientos no se modificaron.

## Cómo usar esta guía frente al profesor

1. Abre el repositorio en VS Code. Presiona **Ctrl+P**, pega una ruta como `src/asl_cuda/model.py:333` y pulsa Enter. Una vez dentro, **Ctrl+G** permite saltar a otra línea.
2. Usa las frases **«Di exactamente»** como guion. No necesitas explicar todos los apartados: empieza por la ruta rápida y usa los demás para preguntas.
3. Abre [el notebook CUDA con sus salidas](notebooks/ASL_CUDA.ipynb) como respaldo. Para esta demostración basta leer código y resultados guardados; **no hace falta pulsar “Run All”** ni repetir entrenamiento.
4. En el notebook, cuenta las celdas desde **1 incluyendo Markdown**. Los números `In [n]` se reinician por sesión y no son una dirección única. Al final se incluye un mapa de celdas reales de esta entrega.
5. Los enlaces a código usan anclas de línea de GitHub; para una navegación local segura usa la ruta `archivo.py:número` con Ctrl+P. Las líneas corresponden a las identidades SHA-256 registradas junto a esta guía.

Los bloques son extractos del proyecto; se añade espacio entre `]` y `(` en algunos lanzamientos para mostrarlos como código y no como enlaces Markdown. Ese espacio no cambia la sintaxis Python ni los archivos originales.

## Ruta rápida: seis paradas para una demostración de unos 5 minutos

| Orden | Abre exactamente | Qué mostrar | Qué decir en una frase |
|---|---|---|---|
|1 |{loc(MODEL,'Network.train_step')} |El lote completo |«Aquí se ordenan las copias, forward, backward, Adam y espera final del lote». |
|2 |{loc(MODEL,'Network.upload')} |Host pinned y H2D |«El CPU prepara; envío únicamente el lote y sus máscaras a GPU». |
|3 |{loc(KERNEL,'dropout')} |Índice global y recorrido con salto |«Este kernel distribuye elementos entre hilos CUDA». |
|4 |{loc(KERNEL,'reduce_channels')} |Shared y `cuda.syncthreads()` |«Aquí cooperan los hilos del mismo bloque para una reducción». |
|5 |{loc(MODEL,'Network.drain_events')} y {loc(BLAS,'Blas.__init__')} |Espera CPU y mismo stream en cuBLAS |«Las dependencias se ordenan en un stream; el host espera al final del lote». |
|6 |{loc(FC,'FCNetwork.reduce')} |Herencia y BN de las FC |«FC-A y FC-B reutilizan el motor y conservan su propia normalización». |

Si dispones de 10–15 minutos, añade los apartados de cuBLAS, backward, softmax, medición y evidencias. Si el profesor pide “solo sincronización”, ve directamente al apartado **6**.
''')
    add(f'''## 1. Mapa del proyecto: qué archivo tiene cada responsabilidad

| Archivo | Qué contiene | Prioridad para enseñar |
|---|---|---|
|[model.py](../{MODEL}) |Orquestación del entrenamiento, buffers, stream, lanzamientos, copias, medición y exportación. |Alta: empieza aquí. |
|[kernels.py](../{KERNEL}) |Funciones que ejecutan los hilos GPU: BN, Dropout, pooling, pérdida, Adam, etc. |Alta: demostrar CUDA explícito. |
|[blas.py](../{BLAS}) |Adaptador `ctypes` para SGEMM en GPU y asociación de cuBLAS al mismo stream. |Alta: explicar el backend real. |
|[fc.py](../{FC}) |Extensión FC-A/FC-B: buffers, semillas y finalización BN con momentum original. |Alta si preguntan por las redes adicionales. |
|[runtime.py](../{RUNTIME}) |Configuración de DLL y detección de dispositivo CUDA real. |Respaldo. |
|[benchmark_cuda.py](../{RUN}) |Protocolo de las cinco corridas CNN-B y preparación CPU. |Alta para justificar las mediciones. |
|[ASL_CUDA.ipynb](notebooks/ASL_CUDA.ipynb) |Código por fases y salidas reales CNN-B y FC. |Respaldo visual de ejecución. |

**Di exactamente:**

> «El código Python del host organiza el trabajo. Los métodos `forward` y `backward` del host lanzan los kernels; no son por sí solos un único kernel GPU. Las funciones `@cuda.jit` y las llamadas cuBLAS realizan la matemática en el dispositivo. TensorFlow se utiliza en CPU para el modelo de referencia, la validación y el guardado». 

**Aclaración importante del backend:** `Network` tiene `backend="explicit"` como valor predeterminado en su constructor. Eso **no significa que esa ruta produjo los tiempos finales**. En {spot(RUN,178)} se crea `Network(reference, backend="cublas")`; en {spot(FC,78)} FC fija el backend a `cublas`. La implementación matricial tiled explícita se conserva como alternativa y antecedente.
''')
    add(f'''## 2. CPU y GPU separados: dónde empieza realmente cada fase

**Abre:** {loc(RUN,'load_images')}.

**Señala:** `Image.open`, conversión RGB, resize64×64, `float32/255`, `ThreadPoolExecutor`, `pending` y `popleft().result()`.

**Di exactamente:**

> «La lectura y preparación de archivos permanece en CPU. Uso seis workers y mantengo como máximo doce tareas pendientes. Cada imagen puede prepararse independientemente; el coordinador coloca los resultados en el orden del manifiesto para conservar la correspondencia con las etiquetas. Esta mejora de preparación es paralelismo CPU, no aceleración CUDA». 

`pending.popleft().result()` espera el resultado de una tarea CPU cuando hace falta. **No es una sincronización CUDA.** Este punto permite distinguir el control de tareas CPU del control del stream GPU.

**Para demostrar que TensorFlow no entrena en GPU:** abre {loc(MODEL,'cpu_reference')} y {loc(FC,'fc_reference')}. Muestra `tf.config.set_visible_devices([], "GPU")` y la política `float32`.

> «Oculto la GPU únicamente para TensorFlow. Eso no desactiva la GPU para Numba-CUDA: son interfaces distintas. El entrenamiento de nuestra versión CUDA se hace con kernels y cuBLAS; la copia Keras evalúa en CPU». 

**Para mostrar admisión de GPU:** {loc(RUNTIME,'configure')} rechaza `NUMBA_ENABLE_CUDASIM`; {loc(RUNTIME,'device_info')} usa `cuda.is_available()`, obtiene el dispositivo y registra capacidad y memoria. El alias de compatibilidad `np.row_stack` en ese archivo no es una optimización matemática ni una transferencia GPU.
''')
    add(f'''## 3. Un lote de entrenamiento, de principio a fin

**Abre y muestra completo:** {loc(MODEL,'Network.train_step')}.

{code(MODEL,'Network.train_step')}

**Señala en este orden:**

| Instrucción | Dónde sucede la matemática | Qué aporta |
|---|---|---|
|`self.upload(...)` |CPU prepara; CUDA copia |Lote, etiquetas y máscaras. |
|`self.forward(len(x))` |Kernels GPU y cuBLAS |Activaciones, logits, pérdida y gradiente de salida. |
|`self.backward(len(x))` |Kernels GPU y cuBLAS |Gradientes de todas las capas. |
|`self.update(len(x))` |Kernels GPU |Adam y métricas. |
|`self.error.copy_to_host(...)` |D2H |Indicador de error del lote. |
|`self.drain_events()` |CPU espera el stream |Termina el trabajo encolado y obtiene tiempos. |

**Di exactamente:**

> «Este método es el recorrido real de cada lote. No estamos usando `model.fit` de TensorFlow sobre GPU. Enviamos el lote, calculamos forward, backward y Adam, copiamos el indicador de error y esperamos a que el stream termine antes de devolver el control al bucle de entrenamiento». 

El argumento `b` se obtiene de `len(x)`. Hay96lotes por época:95de128 y1de20. Se usa el tamaño real, no siempre128. Quince épocas producen1440actualizaciones por modelo y corrida.
''')
    add(f'''## 4. Memoria GPU, memoria pinned y transferencias

**Abre:** {loc(MODEL,'Network.__init__')} para los buffers y {loc(MODEL,'Network.alloc')} / {loc(MODEL,'Network.param')} para la reserva.

**Señala:** `cuda.device_array`, `cuda.pinned_array`, `value`, `grad`, `m`, `v`, `g1`, `g2`, `host_x`, `host_y`.

| Memoria / dato | Ubicación | Uso real |
|---|---|---|
|`value`, `grad`, `m`, `v` |Global GPU |Pesos, gradientes y estados Adam persistentes. |
|`input`, salidas y `g1/g2` |Global GPU |Activaciones y buffers de gradiente. |
|`host_x`, `host_y`, máscaras host |RAM pinned del CPU |Staging para las transferencias de lotes. |
|`cuda.shared.array(...)` |Compartida por bloque GPU |Reducciones y colaboración de hilos. |
|Acumuladores escalares |Privados de cada hilo |Sumas locales; la asignación física de registros/spills depende del compilador. |

**Di exactamente:**

> «No copio todo el dataset a VRAM. Conservo los pesos y estados en GPU y envío cada lote. Los buffers host pinned siguen siendo RAM del CPU; no son VRAM, memoria unificada ni una copia cero. Su objetivo es facilitar las transferencias explícitas». 

El staging de128imágenes RGB64×64float32 ocupa6MiB de datos de imagen. El último lote utiliza solo su porción activa.

**Abre H2D:** {loc(MODEL,'Network.upload')} y para FC {loc(FC,'FCNetwork.upload')}.

**Señala:** `np.copyto` prepara el staging en CPU; `copy_to_device(..., stream=self.stream)` transfiere; las máscaras Bernoulli se generan en CPU y también se copian. En GPU, `dropout` aplica la máscara y el escalado.

**Abre D2H:** {loc(MODEL,'Network.export')}.

> «Cada época exporto pesos y estadísticas de normalización para validar en CPU. Primero encolo las copias D2H y espero; solo después asigno los valores a las variables Keras. Al final también exporto los estados Adam y compruebo el contador». 

No afirmes que **todas** las copias del archivo usan buffers pinned ni que todas se solapan: `export()` también crea destinos host al copiar pesos. Se especifica el stream, pero el protocolo no implementa un pipeline de solapamiento H2D/cómputo/D2H.

Referencia conceptual breve: [memoria pinned y gestión de memoria de Numba-CUDA](https://nvidia.github.io/numba-cuda/user/memory.html).
''')
    add(f'''## 5. Cómo enseñar un kernel y la jerarquía CUDA

**Abre:** {loc(KERNEL,'dropout')}. Para ver dónde se lanza, {loc(MODEL,'Network.forward')} y busca `k.dropout`.

{code(KERNEL,'dropout')}

**Di exactamente:**

> «`@cuda.jit` compila esta función para la GPU. `cuda.grid(1)` da el índice global del hilo; `cuda.gridsize(1)` es el salto entre los elementos que procesa ese hilo. Cada salida tiene un escritor: multiplico la activación por su máscara y por el factor de escalado. Aquí los elementos son independientes y no necesito una barrera de bloque». 

En el lanzamiento, muestra la parte `k.dropout[grid(E), 256, self.stream]`: **grid, threads por block y stream**, en ese orden. Los argumentos de datos van después, entre paréntesis.

**Abre:** {loc(MODEL,'grid')}.

{code(MODEL,'grid')}

| Concepto | En este código |
|---|---|
|Thread |Ejecuta el cuerpo del kernel sobre índices asignados. |
|Block |256hilos en kernels vectoriales;256=8warps de32. Eso no son8SM. |
|Grid |Hasta4096bloques para estos kernels vectoriales. |
|Grid-stride loop |Permite cubrir más elementos que hilos lanzados; evita depender solo del número de bloques. |
|Último lote |Se calcula `E` con20muestras, aunque la capacidad de memoria sea128. |

Ejemplo para explicar sin ejecutar: si `E=1025`, `grid(E)=5`. Se lanzan1280hilos; solo los índices válidos trabajan en ese primer recorrido. Si el tamaño supera el tope de hilos lanzados, el salto de `gridsize` permite continuar.

Referencia de terminología y lanzamiento: [Writing CUDA Kernels](https://nvidia.github.io/numba-cuda/user/kernels.html).
''')
    add(f'''## 6. Sincronización: la parte que debes dominar

### 6.1 Tres mecanismos distintos

| Mecanismo | Dónde se escribe | A quién coordina / espera | Dónde enseñarlo |
|---|---|---|---|
|`cuda.syncthreads()` |Dentro de un kernel |Los hilos del **mismo bloque** llegan a la barrera. |{loc(KERNEL,'reduce_channels')} |
|Orden del mismo stream |Lanzamientos desde el host |Las operaciones dependientes se encolan en orden; no exige detener al CPU entre cada kernel. |{loc(MODEL,'Network.forward')} y {loc(BLAS,'Blas.__init__')} |
|`self.stream.synchronize()` |Código host Python |El CPU espera el trabajo previo de **ese stream**. |{loc(MODEL,'Network.drain_events')} |

**Frase corta para memorizar:**

> «La barrera coordina hilos de un bloque; el stream ordena operaciones GPU; `stream.synchronize` hace esperar al CPU». 

`cuda.synchronize()` es otra API, de alcance del contexto actual. **No es la llamada que usa esta ruta para terminar cada lote**; aquí se usa `self.stream.synchronize()`. Tampoco se implementó una barrera global de todos los bloques dentro de un kernel.

### 6.2 Barrera dentro de un bloque: reducción BN

**Abre:** {loc(KERNEL,'reduce_channels')}. Muestra especialmente las líneas169–175.

**Señala:** `s1` y `s2` compartidas, `tx/ty`, acumulaciones locales, escritura a shared y barreras.

**Di exactamente:**

> «Cada hilo calcula primero una suma parcial. Después la escribe en memoria compartida. La primera barrera evita que otro hilo lea antes de que todos hayan terminado de escribir. En la reducción, unos hilos suman los valores de otros y al final de cada nivel todos vuelven a esperar. Así el siguiente nivel no usa una suma que todavía está incompleta». 

Los bloques son `(32,8)`: `tx` identifica una de32características y `ty` divide el trabajo de reducción entre8filas de hilos. `s1` y `s2` tienen8×32float32 cada una:2KiB de shared en el diseño del código. El patrón de índices hace que valores vecinos de `c` estén contiguos en el arreglo; es una elección de acceso, no una medición de ancho de banda real.

**Si pregunta por qué la barrera está fuera de `if ty < stride`:**

> «Solo algunos hilos suman en un nivel, pero todos los hilos del bloque deben pasar por la barrera. Si la pusiera dentro de una condición que no cumplen todos, no tendría una sincronización válida». 

Lo mismo ocurre con `if c < C`: los hilos fuera de rango aportan cero, pero no se saltan las barreras. No sustituirla por `syncwarp`: aquí hay colaboración entre varias filas de hilos del bloque, no solo dentro de un warp.

### 6.3 ¿Cómo cooperan distintos bloques sin una barrera global?

**Abre la pareja:** {loc(MODEL,'Network.reduce')} y {loc(KERNEL,'finish_channels')}.

> «Cada bloque escribe su parcial en una posición propia de memoria global. Cuando termina ese kernel, un segundo kernel combina los parciales. Los dos están en el mismo stream, por eso el segundo no consume parciales antes de que el primero haya terminado». 

Primera etapa: `parts[blockIdx.x*C+c]`. Segunda: `finish_channels`. Para obtener varianza, el `forward` primero completa la media, después calcula la suma de desviaciones y finalmente normaliza. Las dependencias entre etapas se resuelven por **límites de kernel y orden del stream**, no con `syncthreads` entre bloques.

### 6.4 Un solo stream, incluso dentro de cuBLAS

**Muestra:** {spot(MODEL,82)} (`cuda.stream()`), {spot(FC,83)} y {spot(BLAS,28)} (`cublasSetStream_v2`).

> «Se crea un stream por instancia de red y se pasa a los kernels, a las copias y al handle cuBLAS. Esto permite mantener el orden copia, cálculo, normalización y siguiente operación sin sincronizar al CPU después de cada instrucción». 

**Importante:** un stream no significa “un hilo GPU” ni elimina el paralelismo de los hilos dentro de cada kernel. Tampoco significa que diferentes capas se ejecuten simultáneamente. Aquí se prioriza un orden explícito, con un solo stream y sin solapamiento programado.

### 6.5 Espera del host: al final de cada lote

**Abre:** {loc(MODEL,'Network.drain_events')} y vuelve a {loc(MODEL,'Network.train_step')}.

{code(MODEL,'Network.drain_events')}

**Di exactamente:**

> «Los lanzamientos pueden regresar al host antes de completar el cálculo. Antes de leer el error y los tiempos, espero el stream. Por eso nuestro `train_step` termina de forma sincronizada: cuando devuelve el control, ese lote ya finalizó en GPU. También evita reutilizar el staging pinned del siguiente lote mientras el anterior aún lo necesita». 

Esta espera **sí limita el solapamiento** y añade coste real. No se debe decir que se procesan simultáneamente varios lotes mediante múltiples streams.

### 6.6 Otras esperas que puedes encontrar

| Lugar | Motivo |
|---|---|
|{loc(MODEL,'Network.restore')} |Completar restauración antes de usar el estado inicial. |
|{loc(MODEL,'Network.export')} |Completar D2H antes de asignar pesos/BN en CPU o comprobar el contador. |
|{spot(MODEL,318)} |Espera adicional si se activa `audit_hook` para inspección de gradientes; no se activa en el recorrido ordinario. |
|{spot(RUN,328)} |Cierre del trabajo GPU al finalizar la corrida. |

**Atomic no significa barrera.** Se explica en el apartado8.

Apoyo conceptual: [barreras de bloque de Numba-CUDA](https://nvidia.github.io/numba-cuda/reference/kernel.html#synchronization-and-atomic-operations), [streams](https://nvidia.github.io/numba-cuda/user/memory.html#streams) y [cublasSetStream](https://docs.nvidia.com/cuda/archive/13.1.0/cublas/index.html#cublassetstream).
''')
    add(f'''## 7. Softmax: un ejemplo pequeño y completo de cooperación

**Abre:** {loc(KERNEL,'softmax_loss')}. Busca su lanzamiento en {spot(MODEL,272)}.

**Qué enseñar:** un bloque por imagen,32hilos por bloque y29clases. `n=blockIdx.x` selecciona la muestra; `t=threadIdx.x` selecciona la clase cuando `t<29`.

**Di exactamente:**

> «Dedico un bloque de32hilos a cada imagen. Los primeros29representan las clases. Primero encuentro el máximo, lo resto para estabilizar las exponenciales y después reduzco la suma para normalizar. Los tres hilos sobrantes no representan clases, pero participan en las barreras aportando valores neutros». 

**Recorre:** máximo con `-inf`, barrera, exponenciales, suma compartida, probabilidades, `dlogits`, pérdida y predicción.

El gradiente divide entre `b`, el tamaño real del lote. El lote final de20no se normaliza como si tuviera128. Restar el máximo es estabilidad numérica; no equivale a cambiar la tarea de clasificación.

**Si pregunta por qué no usar solo una imagen por hilo:**

> «Aquí hay una reducción entre las29clases de la misma imagen. Un bloque permite que esos hilos cooperen y compartan el denominador». 
''')
    add(f'''## 8. Condiciones de carrera, atomics y actualización de pesos

### 8.1 La política general: un escritor por salida

**Abre:** {loc(KERNEL,'col2im_gather')} y {loc(KERNEL,'pool_backward')}.

> «En vez de que muchas contribuciones compitan escribiendo el mismo gradiente, cada hilo toma la responsabilidad de una salida y reúne sus contribuciones. Es un patrón gather. Evitamos atomics flotantes para estas acumulaciones». 

En pooling2×2stride2, el backward usa `arg` para enviar el gradiente únicamente al elemento elegido. Los empates se resuelven según el primer máximo encontrado por el forward. El redondeo puede modificar rutas cerca de empates: por eso no se promete trayectoria CPU/GPU idéntica.

### 8.2 Sí hay atomics, pero sobre una bandera entera

**Abre:** {loc(KERNEL,'finite_check')} y {loc(KERNEL,'adam')}.

> «Sí utilizamos `cuda.atomic.max`, pero sobre un indicador `int32` de error, no para acumular los gradientes de la red. Si varios hilos detectan un valor no finito, actualizan esa misma bandera de forma atómica. El host la lee después de esperar el stream». 

No afirmar “no hay atomics en todo el proyecto”. Tampoco que un atomic sincroniza todos los hilos ni que revierte una actualización: la bandera detecta el fallo y `train_step` lanza `FloatingPointError`; no constituye una transacción con rollback.

### 8.3 Backward completo antes de Adam

**Abre:** {loc(MODEL,'Network.backward')}, {loc(MODEL,'Network.update')} y {loc(KERNEL,'adam')}.

> «Primero calculo todos los gradientes con el estado de pesos del lote. Después actualizo los parámetros con Adam. Los buffers `g1` y `g2` se alternan para propagar gradientes sin reservar un tensor nuevo en cada capa». 

`current, other = other, current` intercambia referencias host a buffers GPU; no es una copia D2H. El orden del stream conserva las dependencias aunque los lanzamientos sean asíncronos.

Cada hilo Adam es dueño de un índice de `theta`, `grad`, `moment` y `velocity`. `alpha` incorpora corrección de sesgo con el número de paso. `advance_step` se lanza con1bloque/1hilo para el contador GPU; por eso ese contador no necesita un atomic con múltiples escritores.
''')
    add(f'''## 9. Dónde ocurre la multiplicación: cuBLAS e im2col

**Abre primero:** {loc(MODEL,'Network.gemm')}. Después {loc(BLAS,'Blas.gemm')}.

{code(BLAS,'Blas.gemm')}

**Di exactamente:**

> «Los productos grandes de la versión medida se ejecutan con `cublasSgemm_v2`: multiplicación matricial en precisión simple en GPU. Python prepara los tamaños y pasa las direcciones de los buffers GPU mediante `ctypes`; no está calculando el producto con NumPy en CPU». 

**Qué señalar en `Blas.__init__`:** DLL del toolkit, firmas `ctypes`, `cublasSetStream_v2`, `CUBLAS_PEDANTIC_MATH` y `CUBLAS_ATOMICS_NOT_ALLOWED`. Este último es una política de cuBLAS, distinta de los atomics enteros propios de control de errores.

**Sobre row-major y column-major:**

> «Nuestros arreglos se interpretan en orden fila. El adaptador invierte el orden de operandos y dimensiones para usar la interfaz column-major de cuBLAS, de forma equivalente a transponer el producto». 

No es necesario afirmar que se crea una copia CPU transpuesta: el adaptador cambia la interpretación y los parámetros de la llamada.

**Para las convoluciones:** {loc(KERNEL,'im2col')}, {loc(KERNEL,'col2im_gather')} y la rama `mode in (0,2)` de `Network.gemm`.

> «Transformo las ventanas3×3en una matriz con im2col, uso SGEMM y reorganizo los gradientes de entrada con gather cuando hace falta. Esto reutiliza un producto matricial optimizado a cambio de memoria temporal». 

CNN-B reserva un workspace de columnas de576MiB. Es un coste de memoria de esta ruta, no tamaño del dataset ni “memoria shared”. Los modos Dense3/4/5 corresponden a forward, gradiente de entrada y gradiente de pesos.

### Qué no presentar como backend medido

{loc(KERNEL,'_make_gemm')} contiene el kernel tiled16×16con `sa/sb` shared y dos barreras por tile: una tras cargar y otra antes de reutilizar shared. Es útil si preguntan por multiplicación matricial manual, **pero es la alternativa explícita conservada, no el producto grande usado para los promedios finales**. Para demostrar la ruta medida vuelve al constructor con `backend="cublas"`.

La selección interna de kernels, grids y bloques de SGEMM pertenece a NVIDIA. Nuestro código **no fija** la geometría interna de cuBLAS. Referencia: [cuBLAS13.1](https://docs.nvidia.com/cuda/archive/13.1.0/cublas/index.html).
''')
    add(f'''## 10. Qué cambió para FC-A y FC-B

**Abre:** {loc(FC,'FCNetwork')} y luego {loc(FC,'FCNetwork.reduce')}.

**Di exactamente:**

> «FC significa totalmente conectada. No es una sola operación: son redes completas con forward, normalización, Dropout, backward y Adam. Heredan el motor CUDA de `Network`; la extensión adapta los buffers, la semilla y la reducción de BatchNormalization». 

| Diferencia que señalar | Dónde |
|---|---|
|FC-A128/64 yDropout0.3;FC-B512/256/128 yDropout0.4 |{loc(FC,'fc_reference')} |
|`backend="cublas"`,stream propio ymemoria ajustada |{loc(FC,'FCNetwork.__init__')} |
|Captura `layer.momentum` |{spot(FC,110)} |
|Sin workspace im2col: `self.columns=None` |{spot(FC,148)} |
|Semilla por modelo para las máscaras |{loc(FC,'FCNetwork.upload')} |
|Reducción BN que respeta momentum |{loc(FC,'FCNetwork.reduce')} y {loc(FC,'finish_fc')} |

FC-A yFC-B tienen BN sobre12,288características después de Flatten. Su momentum es0.99; CNN-B usa0.9. **No se copió el0.9de CNN-B a las FC.** Epsilon se conserva0.001.

**Si pregunta por ausencia de `syncthreads` en `finish_fc`:**

> «En esa función cada hilo termina una característica distinta y usa sus propios acumuladores. No intercambia datos shared con otro hilo, por eso no necesita una barrera interna. La dependencia con el kernel que produce los parciales sigue garantizada por el mismo stream». 

Con `C=12288` y`L=128` o20, la reducción parcial usa grid `(1,384)`,block `(32,8)`. La final usa48bloques de256hilos. La implementación FC hereda `train_step`, `forward`, `backward`, `update`, `restore` y `export`; encontrar esos nombres solo en `model.py` es correcto.

Buffers propios FC-A≈49.46MiB yFC-B≈124.65MiB. No son iguales a la VRAM observada del proceso/contexto ni acreditan saturación.
''')
    add(f'''## 11. Cómo se midió el tiempo sin medir solo el envío de trabajo

**Abre eventos:** {loc(MODEL,'Network.timed')} y {loc(MODEL,'Network.drain_events')}.

> «Registro eventos de inicio y fin en el mismo stream, espero a que terminen y obtengo el intervalo. La API devuelve milisegundos; divido entre1000para guardar segundos». 

**Abre tiempo principal:** {spot(RUN,233)} hasta la línea282del controlador CNN-B; para las FC, usa las celdas de entrenamiento del notebook indicadas más abajo.

**Señala:** `train_start = perf_counter_ns()`, restauración inicial incluida,15épocas, `net.train_step`, `net.export`, `reference.evaluate` en cada época, exportación final Adam y `train_end`.

> «La cifra principal es tiempo pared del entrenamiento completo, no la suma de tiempos de kernels. Incluye preparación de lotes y máscaras, transferencias, esperas, forward, backward, Adam y validación CPU. El test final se mide aparte. Warmup, preparación inicial de imágenes, guardado y elaboración de gráficas no están dentro de ese fit». 

Las llamadas `train_step` y `export` esperan el stream antes de devolver el control. Por eso el final del intervalo no deja cálculo del último lote pendiente.

**Dos aclaraciones para evitar errores:**

- Los eventos agrupan operaciones y pueden incluir huecos mientras el host envía trabajo. No debes llamarlos «tiempo activo puro de los núcleos GPU».
- Validación y eventos son subintervalos del fit. No se vuelven a sumar al total de entrenamiento. La exportación final copia el modelo una vez adicional; ese coste se incluyó, no se restó.

Referencia de eventos/unidades: [CUDA Host API](https://nvidia.github.io/numba-cuda/reference/host.html#events).
''')
    add(f'''## 12. Warmup, restauración y pruebas: qué mostrar si preguntan por validez

**Abre protocolo CNN-B:** {spot(RUN,175)} hasta la línea209. **Pruebas nuevas FC:** {loc('scripts/validar_cuda_fc.py','main')}.

> «Antes de cronometrar ejercito lotes128y20y las rutas de validación/testCPU. Después restauro pesos, estadísticasBN, momentos/velocidadesAdam ycontador. Verifico la restauración; no inicio la medición con un modelo ya entrenado por el calentamiento». 

**Datos y estado:** muestra las comprobaciones SHA-256 y de tensores en el controlador antes de entrenar. En `cpu_reference`/`fc_reference` se restauran los archivos NPZ históricos por orden de variable, forma ydtype. No dependemos solo de volver a usar la misma semilla.

**Pruebas FC:** `GradientTape`, máscaras compartidas con el oráculo, comparación de probabilidades/gradientes, BN yprimera actualizaciónAdam; lotes20y128para ambas redes.

**Di exactamente sobre el límite de comparación:**

> «Reutilizamos la CPU histórica para no repetir horas de ejecución. Coinciden imágenes, particiones, pesos iniciales, arquitectura, número de pasos yprecisión. El orden interno de lotes ylas máscaras históricas no son idénticos, y el estado térmico no se alternó por pares. Por eso reportamos el speedup temporal yla calidad por separado». 

No etiquetar las pruebas locales como prueba de igualdad de toda la trayectoria de15épocas. Tampoco llamar determinismo general a cinco repeticiones GPU con el mismo estado y RNG.
''')
    add('''## 13. Dónde enseñar las salidas reales dentro del notebook

Abre [entregables/notebooks/ASL_CUDA.ipynb](notebooks/ASL_CUDA.ipynb). Los módulos también aparecen como código visible en celdas `%%modulo` / `%%modulo_fc`. Las distintas sesiones están identificadas; los contadores de ejecución se reinician legítimamente.

**Para llegar rápido:** Ctrl+F con `%%modulo asl_cuda.model`, `%%modulo_fc asl_cuda.fc`, `train_start`, `EPOCH FC-A` o `EPOCH FC-B`. Los primeros encuentran código; los últimos, salidas guardadas.

| Celda visual (desde1) | In [n] de esa sesión | Qué enseñar | Primera línea / búsqueda |
|---|---|---|---|''')
    for c in cells:
        add('|'+str(c['celda_visual'])+'|'+str(c['execution_count'])+'|'+'; '.join(c['temas'])+'|`'+c['primera_linea'].replace('|','/')+'`|')
    add('''La numeración anterior cuenta también las celdas Markdown y corresponde al hash del notebook registrado al crear esta guía. Si se añaden celdas después, las búsquedas por función/magic siguen siendo la referencia más estable. Las celdas de recuperación/presentación de resultados se identifican como lectura de archivos; no son un entrenamiento nuevo.

**Di exactamente al mostrar una salida:**

> «Estas salidas quedaron registradas durante las ejecuciones reales. Aquí podemos comprobar las15épocas, los1440pasos, las métricas ylas capturas. Para revisar evidencia no necesitamos repetir el entrenamiento». 
''')
    add('''## 14. Resultados y evidencias: archivos exactos para responder con pruebas

| Si pregunta… | Abre |
|---|---|
|¿Cuánto aceleró cada red? |[comparacion.csv](../resultados/comparacion/ampliacion-fc/comparacion.csv):filas `entrenamiento_*`. |
|¿La exactitud se mantuvo? |[calidad.csv](../resultados/comparacion/ampliacion-fc/calidad.csv),y[calidad CNN-B](../resultados/paralelo/EXP-003/calidad_comparada.csv). |
|¿Hay varias repeticiones yDE? |[estadisticas.csv](../resultados/comparacion/ampliacion-fc/estadisticas.csv) y[observaciones.csv](../resultados/comparacion/ampliacion-fc/observaciones.csv). |
|¿Se verificaron checkpoints ypasos? |[validacion-final.json](../resultados/preparacion/ampliacion-fc/validacion-final.json). |
|¿Se comprobaron gradientes FC? |[validación del oráculo FC](../resultados/preparacion/ampliacion-fc/validacion-01/validacion.json). |
|¿La GPU trabajó de verdad? |[captura CNN-B](../resultados/paralelo/EXP-009/RUN-20261118T041129Z-cuda-notebook-01/captura-cuda.png),y su[JSON](../resultados/paralelo/EXP-009/RUN-20261118T041129Z-cuda-notebook-01/captura-cuda.json). |
|¿También trabajaron FC-A yFC-B? |[captura FC-A](../resultados/paralelo/EXP-011/RUN-20261118T134323Z-fc-a-01/captura-fc-a.png) y[captura FC-B](../resultados/paralelo/EXP-012/RUN-20261118T134514Z-fc-b-01/captura-fc-b.png). |
|¿Qué se copió por lote? |[transferencias FC-B](../resultados/paralelo/EXP-012/RUN-20261118T134514Z-fc-b-01/transferencias.json). |
|¿Qué geometrías yrecursos se registraron? |[lanzamientos FC-B](../resultados/paralelo/EXP-012/RUN-20261118T134514Z-fc-b-01/lanzamientos.json). |

**Resumen para decir:**

> «Con cinco corridas por modelo, CNN-B pasó de805.65a418.61segundos; FC-A,de50.72a15.12; yFC-B,de102.44a21.06. Los speedups son1.92,3.35y4.86veces. Incluyen transferencias yvalidaciónCPU. Esto no es el speedup de todo el notebook original». 

> «Las FC tuvieron menor exactitud que la referenciaCPU:lo reportamos,no lo ocultamos. La mejora demostrada es temporal. Tampoco afirmamos saturación total deVRAM ni ocupación porSM:los monitores muestran actividad agregada ymemoria usada». 

La capturaCNN-Bdel91% pertenece a una corrida adicional n=1delnotebook; no reemplaza los cinco RUN del promedio. Registros compilados de registros/shared/local no equivalen a medir ocupaciónSM ni prueban que256hilos sea el tamaño óptimo global.
''')
    add(f'''## 15. Preguntas técnicas probables y respuestas listas

| Pregunta | Respuesta que puedes decir | Dónde apuntar |
|---|---|---|
|¿Dónde está CUDA si todo esPython? |«En las funciones `@cuda.jit` ylos lanzamientos explícitos. Numba-CUDA compila el kernel; Pythonhost organiza el flujo». |{loc(KERNEL,'dropout')} |
|¿Por qué algunas funciones no llevan `@cuda.jit`? |«Son controladores delhost. Dentro llaman kernels ycuBLAS; no toda líneaPython se ejecuta enGPU». |{loc(MODEL,'Network.forward')} |
|¿`syncthreads` espera toda laGPU? |«No. Solo los hilos delmismo bloque». |{loc(KERNEL,'reduce_channels')} |
|¿El stream hace esperar alCPU? |«Encolar en elstream no implica esperar; `stream.synchronize` sí hace esa espera». |{loc(MODEL,'Network.drain_events')} |
|¿Usaron varios streams? |«No. Uno por red yel mismo en cuBLAS,para conservar dependencias. Sin pipeline de lotes solapados». |{spot(BLAS,28)} |
|¿Usaron shared? |«Sí,en reduccionesBN,softmax ymétricas. ElGEMMtiled también tiene shared,pero esalternativo albackendcuBLASmedido». |{loc(KERNEL,'reduce_channels')} |
|¿Por qué256hilos? |«Es la configuración aplicada:8warps yrecorrido con salto. No afirmamos haber demostrado el óptimo de ocupación». |{loc(MODEL,'grid')} |
|¿Por qué32hilos ensoftmax sihay29clases? |«Un bloque cubre las clases deuna imagen;los hilos sobrantes aportan neutros yparticipan enbarreras». |{loc(KERNEL,'softmax_loss')} |
|¿Todas las operaciones están paralelizadas? |«La matemática delentrenamiento deCNN-B,FC-AyFC-B está enGPU. Preparación,validación,test yotras ramas permanecenCPU». |{loc(RUN,'load_images')} |
|¿cuBLASes hacer trampa o usarCPU? |«Es una bibliotecaCUDA deNVIDIA que ejecutaSGEMMenGPU. Nuestroskernels manejan otras operaciones;separamos ambasresponsabilidades». |{loc(BLAS,'Blas.gemm')} |
|¿Se perdió la sincronización con cuBLAS? |«No;su handle recibe exactamenteel mismo stream que los kernels». |{spot(BLAS,28)} |
|¿Hay atomics? |«Sí,sobrela banderaentera deerror. No acumulamos gradientes conatomicsflotantes ennuestroskernels». |{loc(KERNEL,'finite_check')} |
|¿Cómo procesan el último lote? |«Su tamaño real es20;la pérdida ylas dimensiones se calculan conese tamaño». |{loc(MODEL,'Network.train_step')} |
|¿Por qué exportan cada época? |«Para validar enCPU manteniendoCUDA enentrenamiento;esecosteestá incluido». |{loc(MODEL,'Network.export')} |
|¿Por qué las FC usanmenosGPU yaceleran más? |«Su referenciaCPU ysu carga son distintas;el speedup es un cociente detiempos,no un porcentaje deocupación». |Desglose ycomparación del apartado14. |
|¿Qué optimización futura harían? |«Evaluaría fusión dekernels,reducir esperas porlote ysolapar transferencias donde hayaindependencia. Son propuestas,no resultados medidos aquí». |Final de `train_step`,no prometer speedup futuro. |
''')
    add('''## 16. Frases que conviene evitar y su versión correcta

| Evita afirmar | Di en su lugar |
|---|---|
|«Todo el Python corre enGPU». |«El host lanza kernels CUDA ySGEMM;preparación yvalidación siguen enCPU». |
|«`syncthreads` sincroniza todos losbloques». |«Sincroniza hilos delmismo bloque;entre etapas usamos kernels separados en el mismo stream». |
|«Como hay stream,cálculo ycopias se solapan». |«Hay un solo stream yuna espera porlote;no se implementó solapamiento». |
|«Los eventos miden tiempo puro denúcleos». |«Miden intervalos delstream;el principal es tiempo pared completo». |
|«El kernel tiled16×16produjo el speedup final». |«La serie final utiliza cuBLAS;el tiled es unbackend alternativo conservado». |
|«cuBLAS usa nuestros mismos256hilos». |«La biblioteca gestiona su propia jerarquía interna». |
|«No hay atomics». |«Hay atomics enteros paraerrores;sinatomics flotantes propios para acumular gradientes». |
|«Pinned significa memoriaGPU». |«Es memoria host fijada;se copia explícitamente a buffers device». |
|«LaGPUestuvo91%ocupada porSM». |«nvidia-smi mostró91%deactividad agregada enesa muestra». |
|«Se mantuvo exactamente laaccuracy». |«Se mantuvo la carga;calidad ytrayectorias se reportan por separado». |
|«Las presentaciones,guías ygráficas requieren entrenar otra vez». |«Son lecturas ydocumentación de resultadosguardados». |

## 17. Cierre literal de tu demostración

> «La contribución no es únicamente activar una GPU. Separamos preparaciónCPU yentrenamientoGPU,definimos kernels ysu jerarquía,controlamos memoria ydependencias,unificamos elstreamconcuBLAS ymedimos el coste completo,incluidastransferencias yvalidación. Laespera porlote hace que los resultados ylos tiempos se lean después determinar eltrabajo. Es unaimplementación verificable para tresredes,con loslímites numéricos ydecalidad declarados». 

## Documentación técnica de apoyo

Las ubicaciones de código anteriores son la evidencia principal de **lo implementado**. Las referencias oficiales solo respaldan el significado de las API; no sustituyen las mediciones locales.

- NVIDIA. *Writing CUDA Kernels*, Numba-CUDA: [modelo de ejecución y lanzamientos](https://nvidia.github.io/numba-cuda/user/kernels.html).
- NVIDIA. *CUDA Kernel API*, Numba-CUDA: [sincronización y atomics](https://nvidia.github.io/numba-cuda/reference/kernel.html).
- NVIDIA. *Memory management*, Numba-CUDA: [pinned, shared y streams](https://nvidia.github.io/numba-cuda/user/memory.html).
- NVIDIA. *CUDA Host API*, Numba-CUDA: [dispositivo yeventos](https://nvidia.github.io/numba-cuda/reference/host.html).
- NVIDIA. *cuBLAS Library, CUDA Toolkit13.1*: [stream,SGEMM ydisposición de matrices](https://docs.nvidia.com/cuda/archive/13.1.0/cublas/index.html).

Consulta de apoyo realizada durante esta inspección documental. No se ejecutó ningún kernel para crear la guía.
''')
    OUT.write_text(format_guide('\n\n'.join(sections)+'\n'),encoding='utf8')
    anchors=[dict(archivo=p,simbolo=n,inicio=se,fin=en) for (p,n),(se,en) in INDEX.items()]
    (QA/'simbolos.json').write_text(json.dumps(anchors,indent=2,ensure_ascii=False),encoding='utf8')
    with (QA/'fuentes.csv').open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['ruta_relativa','bytes','sha256']);w.writeheader()
        for rel in [*SOURCES,'entregables/notebooks/ASL_CUDA.ipynb','configuraciones/EXP-003.json','configuraciones/EXP-012.json']:
            p=ROOT/rel;w.writerow(dict(ruta_relativa=rel,bytes=p.stat().st_size,sha256=sha(p)))
    print('GUÍA CREADA:',len(OUT.read_text(encoding='utf8').splitlines()),'líneas;',len(cells),'ubicaciones de notebook;',len(INDEX),'símbolos AST. Sin ejecutar modelos.')
if __name__=='__main__':main()
