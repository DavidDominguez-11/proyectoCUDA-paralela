# Guía para enseñar y defender el código CUDA

Guion de consulta para la exposición. Basado en **el código real del proyecto**, inspeccionado el 2026-11-18. Las ubicaciones incluyen archivo, función y línea de inicio para abrir con precisión. Las presentaciones y los entrenamientos no se modificaron.

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
|1 |[Network.train_step](../src/asl_cuda/model.py#L333) — `src/asl_cuda/model.py:333`, líneas 333–346 |El lote completo |«Aquí se ordenan las copias, forward, backward, Adam y espera final del lote». |
|2 |[Network.upload](../src/asl_cuda/model.py#L233) — `src/asl_cuda/model.py:233`, líneas 233–249 |Host pinned y H2D |«El CPU prepara; envío únicamente el lote y sus máscaras a GPU». |
|3 |[dropout](../src/asl_cuda/kernels.py#L113) — `src/asl_cuda/kernels.py:113`, líneas 113–115 |Índice global y recorrido con salto |«Este kernel distribuye elementos entre hilos CUDA». |
|4 |[reduce_channels](../src/asl_cuda/kernels.py#L147) — `src/asl_cuda/kernels.py:147`, líneas 147–179 |Shared y `cuda.syncthreads()` |«Aquí cooperan los hilos del mismo bloque para una reducción». |
|5 |[Network.drain_events](../src/asl_cuda/model.py#L191) — `src/asl_cuda/model.py:191`, líneas 191–197 y [Blas.__init__](../src/asl_cuda/blas.py#L12) — `src/asl_cuda/blas.py:12`, líneas 12–33 |Espera CPU y mismo stream en cuBLAS |«Las dependencias se ordenan en un stream; el host espera al final del lote». |
|6 |[FCNetwork.reduce](../src/asl_cuda/fc.py#L178) — `src/asl_cuda/fc.py:178`, líneas 178–186 |Herencia y BN de las FC |«FC-A y FC-B reutilizan el motor y conservan su propia normalización». |

Si dispones de 10–15 minutos, añade los apartados de cuBLAS, backward, softmax, medición y evidencias. Si el profesor pide “solo sincronización”, ve directamente al apartado **6**.

## 1. Mapa del proyecto: qué archivo tiene cada responsabilidad

| Archivo | Qué contiene | Prioridad para enseñar |
|---|---|---|
|[model.py](../src/asl_cuda/model.py) |Orquestación del entrenamiento, buffers, stream, lanzamientos, copias, medición y exportación. |Alta: empieza aquí. |
|[kernels.py](../src/asl_cuda/kernels.py) |Funciones que ejecutan los hilos GPU: BN, Dropout, pooling, pérdida, Adam, etc. |Alta: demostrar CUDA explícito. |
|[blas.py](../src/asl_cuda/blas.py) |Adaptador `ctypes` para SGEMM en GPU y asociación de cuBLAS al mismo stream. |Alta: explicar el backend real. |
|[fc.py](../src/asl_cuda/fc.py) |Extensión FC-A/FC-B: buffers, semillas y finalización BN con momentum original. |Alta si preguntan por las redes adicionales. |
|[runtime.py](../src/asl_cuda/runtime.py) |Configuración de DLL y detección de dispositivo CUDA real. |Respaldo. |
|[benchmark_cuda.py](../scripts/benchmark_cuda.py) |Protocolo de las cinco corridas CNN-B y preparación CPU. |Alta para justificar las mediciones. |
|[ASL_CUDA.ipynb](notebooks/ASL_CUDA.ipynb) |Código por fases y salidas reales CNN-B y FC. |Respaldo visual de ejecución. |

**Di exactamente:**

> «El código Python del host organiza el trabajo. Los métodos `forward` y `backward` del host lanzan los kernels; no son por sí solos un único kernel GPU. Las funciones `@cuda.jit` y las llamadas cuBLAS realizan la matemática en el dispositivo. TensorFlow se utiliza en CPU para el modelo de referencia, la validación y el guardado». 

**Aclaración importante del backend:** `Network` tiene `backend="explicit"` como valor predeterminado en su constructor. Eso **no significa que esa ruta produjo los tiempos finales**. En [scripts/benchmark_cuda.py:178](../scripts/benchmark_cuda.py#L178) se crea `Network(reference, backend="cublas")`; en [src/asl_cuda/fc.py:78](../src/asl_cuda/fc.py#L78) FC fija el backend a `cublas`. La implementación matricial tiled explícita se conserva como alternativa y antecedente.

## 2. CPU y GPU separados: dónde empieza realmente cada fase

**Abre:** [load_images](../scripts/benchmark_cuda.py#L102) — `scripts/benchmark_cuda.py:102`, líneas 102–128.

**Señala:** `Image.open`, conversión RGB, resize 64×64, `float32/255`, `ThreadPoolExecutor`, `pending` y `popleft().result()`.

**Di exactamente:**

> «La lectura y preparación de archivos permanece en CPU. Uso seis workers y mantengo como máximo doce tareas pendientes. Cada imagen puede prepararse independientemente; el coordinador coloca los resultados en el orden del manifiesto para conservar la correspondencia con las etiquetas. Esta mejora de preparación es paralelismo CPU, no aceleración CUDA». 

`pending.popleft().result()` espera el resultado de una tarea CPU cuando hace falta. **No es una sincronización CUDA.** Este punto permite distinguir el control de tareas CPU del control del stream GPU.

**Para demostrar que TensorFlow no entrena en GPU:** abre [cpu_reference](../src/asl_cuda/model.py#L20) — `src/asl_cuda/model.py:20`, líneas 20–57 y [fc_reference](../src/asl_cuda/fc.py#L41) — `src/asl_cuda/fc.py:41`, líneas 41–73. Muestra `tf.config.set_visible_devices([], "GPU")` y la política `float32`.

> «Oculto la GPU únicamente para TensorFlow. Eso no desactiva la GPU para Numba-CUDA: son interfaces distintas. El entrenamiento de nuestra versión CUDA se hace con kernels y cuBLAS; la copia Keras evalúa en CPU». 

**Para mostrar admisión de GPU:** [configure](../src/asl_cuda/runtime.py#L8) — `src/asl_cuda/runtime.py:8`, líneas 8–16 rechaza `NUMBA_ENABLE_CUDASIM`; [device_info](../src/asl_cuda/runtime.py#L29) — `src/asl_cuda/runtime.py:29`, líneas 29–38 usa `cuda.is_available()`, obtiene el dispositivo y registra capacidad y memoria. El alias de compatibilidad `np.row_stack` en ese archivo no es una optimización matemática ni una transferencia GPU.

## 3. Un lote de entrenamiento, de principio a fin

**Abre y muestra completo:** [Network.train_step](../src/asl_cuda/model.py#L333) — `src/asl_cuda/model.py:333`, líneas 333–346.

```python
    def train_step(self, x, labels, epoch=0, batch_index=0):
        self.upload(x, labels, epoch, batch_index)
        with self.timed("forward_loss"):
            self.forward(len(x))
        with self.timed("backward"):
            self.backward(len(x))
        with self.timed("adam_metricas"):
            self.update(len(x))
        with self.timed("d2h_control"):
            self.error.copy_to_host(self.host_error, stream=self.stream)
        times = self.drain_events()
        if self.host_error[0]:
            raise FloatingPointError(f"CUDA sticky error {self.host_error[0]}, step {self.steps}")
        return times
```

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

El argumento `b` se obtiene de `len(x)`. Hay 96 lotes por época:95 de 128 y 1 de 20. Se usa el tamaño real, no siempre 128. Quince épocas producen 1440 actualizaciones por modelo y corrida.

## 4. Memoria GPU, memoria pinned y transferencias

**Abre:** [Network.__init__](../src/asl_cuda/model.py#L78) — `src/asl_cuda/model.py:78`, líneas 78–154 para los buffers y [Network.alloc](../src/asl_cuda/model.py#L156) — `src/asl_cuda/model.py:156`, líneas 156–159 / [Network.param](../src/asl_cuda/model.py#L161) — `src/asl_cuda/model.py:161`, líneas 161–166 para la reserva.

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

El staging de 128 imágenes RGB64×64 float32 ocupa 6 MiB de datos de imagen. El último lote utiliza solo su porción activa.

**Abre H2D:** [Network.upload](../src/asl_cuda/model.py#L233) — `src/asl_cuda/model.py:233`, líneas 233–249 y para FC [FCNetwork.upload](../src/asl_cuda/fc.py#L160) — `src/asl_cuda/fc.py:160`, líneas 160–176.

**Señala:** `np.copyto` prepara el staging en CPU; `copy_to_device(..., stream=self.stream)` transfiere; las máscaras Bernoulli se generan en CPU y también se copian. En GPU, `dropout` aplica la máscara y el escalado.

**Abre D2H:** [Network.export](../src/asl_cuda/model.py#L351) — `src/asl_cuda/model.py:351`, líneas 351–377.

> «Cada época exporto pesos y estadísticas de normalización para validar en CPU. Primero encolo las copias D2H y espero; solo después asigno los valores a las variables Keras. Al final también exporto los estados Adam y compruebo el contador». 

No afirmes que **todas** las copias del archivo usan buffers pinned ni que todas se solapan: `export()` también crea destinos host al copiar pesos. Se especifica el stream, pero el protocolo no implementa un pipeline de solapamiento H2D/cómputo/D2H.

Referencia conceptual breve: [memoria pinned y gestión de memoria de Numba-CUDA](https://nvidia.github.io/numba-cuda/user/memory.html).

## 5. Cómo enseñar un kernel y la jerarquía CUDA

**Abre:** [dropout](../src/asl_cuda/kernels.py#L113) — `src/asl_cuda/kernels.py:113`, líneas 113–115. Para ver dónde se lanza, [Network.forward](../src/asl_cuda/model.py#L251) — `src/asl_cuda/model.py:251`, líneas 251–274 y busca `k.dropout`.

```python
@cuda.jit(cache=True)
def dropout(x, mask, out, factor, E):
    for i in range(cuda.grid(1), E, cuda.gridsize(1)):
        out[i] = x[i] * float32(mask[i]) * factor
```

**Di exactamente:**

> «`@cuda.jit` compila esta función para la GPU. `cuda.grid(1)` da el índice global del hilo; `cuda.gridsize(1)` es el salto entre los elementos que procesa ese hilo. Cada salida tiene un escritor: multiplico la activación por su máscara y por el factor de escalado. Aquí los elementos son independientes y no necesito una barrera de bloque». 

En el lanzamiento, muestra la parte `k.dropout[grid(E), 256, self.stream]`: **grid, threads por block y stream**, en ese orden. Los argumentos de datos van después, entre paréntesis.

**Abre:** [grid](../src/asl_cuda/model.py#L60) — `src/asl_cuda/model.py:60`, líneas 60–61.

```python
def grid(E):
    return max(1, min(4096, (int(E) + 255) // 256))
```

| Concepto | En este código |
|---|---|
|Thread |Ejecuta el cuerpo del kernel sobre índices asignados. |
|Block |256 hilos en kernels vectoriales;256=8 warps de 32. Eso no son 8SM. |
|Grid |Hasta 4096 bloques para estos kernels vectoriales. |
|Grid-stride loop |Permite cubrir más elementos que hilos lanzados; evita depender solo del número de bloques. |
|Último lote |Se calcula `E` con 20 muestras, aunque la capacidad de memoria sea 128. |

Ejemplo para explicar sin ejecutar: si `E=1025`, `grid(E)=5`. Se lanzan 1280 hilos; solo los índices válidos trabajan en ese primer recorrido. Si el tamaño supera el tope de hilos lanzados, el salto de `gridsize` permite continuar.

Referencia de terminología y lanzamiento: [Writing CUDA Kernels](https://nvidia.github.io/numba-cuda/user/kernels.html).

## 6. Sincronización: la parte que debes dominar

### 6.1 Tres mecanismos distintos

| Mecanismo | Dónde se escribe | A quién coordina / espera | Dónde enseñarlo |
|---|---|---|---|
|`cuda.syncthreads()` |Dentro de un kernel |Los hilos del **mismo bloque** llegan a la barrera. |[reduce_channels](../src/asl_cuda/kernels.py#L147) — `src/asl_cuda/kernels.py:147`, líneas 147–179 |
|Orden del mismo stream |Lanzamientos desde el host |Las operaciones dependientes se encolan en orden; no exige detener al CPU entre cada kernel. |[Network.forward](../src/asl_cuda/model.py#L251) — `src/asl_cuda/model.py:251`, líneas 251–274 y [Blas.__init__](../src/asl_cuda/blas.py#L12) — `src/asl_cuda/blas.py:12`, líneas 12–33 |
|`self.stream.synchronize()` |Código host Python |El CPU espera el trabajo previo de **ese stream**. |[Network.drain_events](../src/asl_cuda/model.py#L191) — `src/asl_cuda/model.py:191`, líneas 191–197 |

**Frase corta para memorizar:**

> «La barrera coordina hilos de un bloque; el stream ordena operaciones GPU; `stream.synchronize` hace esperar al CPU». 

`cuda.synchronize()` es otra API, de alcance del contexto actual. **No es la llamada que usa esta ruta para terminar cada lote**; aquí se usa `self.stream.synchronize()`. Tampoco se implementó una barrera global de todos los bloques dentro de un kernel.

### 6.2 Barrera dentro de un bloque: reducción BN

**Abre:** [reduce_channels](../src/asl_cuda/kernels.py#L147) — `src/asl_cuda/kernels.py:147`, líneas 147–179. Muestra especialmente las líneas 169–175.

**Señala:** `s1` y `s2` compartidas, `tx/ty`, acumulaciones locales, escritura a shared y barreras.

**Di exactamente:**

> «Cada hilo calcula primero una suma parcial. Después la escribe en memoria compartida. La primera barrera evita que otro hilo lea antes de que todos hayan terminado de escribir. En la reducción, unos hilos suman los valores de otros y al final de cada nivel todos vuelven a esperar. Así el siguiente nivel no usa una suma que todavía está incompleta». 

Los bloques son `(32,8)`: `tx` identifica una de 32 características y `ty` divide el trabajo de reducción entre 8 filas de hilos. `s1` y `s2` tienen 8×32 float32 cada una:2 KiB de shared en el diseño del código. El patrón de índices hace que valores vecinos de `c` estén contiguos en el arreglo; es una elección de acceso, no una medición de ancho de banda real.

**Si pregunta por qué la barrera está fuera de `if ty < stride`:**

> «Solo algunos hilos suman en un nivel, pero todos los hilos del bloque deben pasar por la barrera. Si la pusiera dentro de una condición que no cumplen todos, no tendría una sincronización válida». 

Lo mismo ocurre con `if c < C`: los hilos fuera de rango aportan cero, pero no se saltan las barreras. No sustituirla por `syncwarp`: aquí hay colaboración entre varias filas de hilos del bloque, no solo dentro de un warp.

### 6.3 ¿Cómo cooperan distintos bloques sin una barrera global?

**Abre la pareja:** [Network.reduce](../src/asl_cuda/model.py#L222) — `src/asl_cuda/model.py:222`, líneas 222–231 y [finish_channels](../src/asl_cuda/kernels.py#L183) — `src/asl_cuda/kernels.py:183`, líneas 183–210.

> «Cada bloque escribe su parcial en una posición propia de memoria global. Cuando termina ese kernel, un segundo kernel combina los parciales. Los dos están en el mismo stream, por eso el segundo no consume parciales antes de que el primero haya terminado». 

Primera etapa: `parts[blockIdx.x*C+c]`. Segunda: `finish_channels`. Para obtener varianza, el `forward` primero completa la media, después calcula la suma de desviaciones y finalmente normaliza. Las dependencias entre etapas se resuelven por **límites de kernel y orden del stream**, no con `syncthreads` entre bloques.

### 6.4 Un solo stream, incluso dentro de cuBLAS

**Muestra:** [src/asl_cuda/model.py:82](../src/asl_cuda/model.py#L82) (`cuda.stream()`), [src/asl_cuda/fc.py:83](../src/asl_cuda/fc.py#L83) y [src/asl_cuda/blas.py:28](../src/asl_cuda/blas.py#L28) (`cublasSetStream_v2`).

> «Se crea un stream por instancia de red y se pasa a los kernels, a las copias y al handle cuBLAS. Esto permite mantener el orden copia, cálculo, normalización y siguiente operación sin sincronizar al CPU después de cada instrucción». 

**Importante:** un stream no significa “un hilo GPU” ni elimina el paralelismo de los hilos dentro de cada kernel. Tampoco significa que diferentes capas se ejecuten simultáneamente. Aquí se prioriza un orden explícito, con un solo stream y sin solapamiento programado.

### 6.5 Espera del host: al final de cada lote

**Abre:** [Network.drain_events](../src/asl_cuda/model.py#L191) — `src/asl_cuda/model.py:191`, líneas 191–197 y vuelve a [Network.train_step](../src/asl_cuda/model.py#L333) — `src/asl_cuda/model.py:333`, líneas 333–346.

```python
    def drain_events(self):
        self.stream.synchronize()
        result = {}
        for phase, start, end in self.events:
            result[phase] = result.get(phase, 0.) + cuda.event_elapsed_time(start, end) / 1000.
        self.events.clear()
        return result
```

**Di exactamente:**

> «Los lanzamientos pueden regresar al host antes de completar el cálculo. Antes de leer el error y los tiempos, espero el stream. Por eso nuestro `train_step` termina de forma sincronizada: cuando devuelve el control, ese lote ya finalizó en GPU. También evita reutilizar el staging pinned del siguiente lote mientras el anterior aún lo necesita». 

Esta espera **sí limita el solapamiento** y añade coste real. No se debe decir que se procesan simultáneamente varios lotes mediante múltiples streams.

### 6.6 Otras esperas que puedes encontrar

| Lugar | Motivo |
|---|---|
|[Network.restore](../src/asl_cuda/model.py#L168) — `src/asl_cuda/model.py:168`, líneas 168–181 |Completar restauración antes de usar el estado inicial. |
|[Network.export](../src/asl_cuda/model.py#L351) — `src/asl_cuda/model.py:351`, líneas 351–377 |Completar D2H antes de asignar pesos/BN en CPU o comprobar el contador. |
|[src/asl_cuda/model.py:318](../src/asl_cuda/model.py#L318) |Espera adicional si se activa `audit_hook` para inspección de gradientes; no se activa en el recorrido ordinario. |
|[scripts/benchmark_cuda.py:328](../scripts/benchmark_cuda.py#L328) |Cierre del trabajo GPU al finalizar la corrida. |

**Atomic no significa barrera.** Se explica en el apartado 8.

Apoyo conceptual: [barreras de bloque de Numba-CUDA](https://nvidia.github.io/numba-cuda/reference/kernel.html#synchronization-and-atomic-operations), [streams](https://nvidia.github.io/numba-cuda/user/memory.html#streams) y [cublasSetStream](https://docs.nvidia.com/cuda/archive/13.1.0/cublas/index.html#cublassetstream).

## 7. Softmax: un ejemplo pequeño y completo de cooperación

**Abre:** [softmax_loss](../src/asl_cuda/kernels.py#L230) — `src/asl_cuda/kernels.py:230`, líneas 230–269. Busca su lanzamiento en [src/asl_cuda/model.py:272](../src/asl_cuda/model.py#L272).

**Qué enseñar:** un bloque por imagen,32 hilos por bloque y 29 clases. `n=blockIdx.x` selecciona la muestra; `t=threadIdx.x` selecciona la clase cuando `t<29`.

**Di exactamente:**

> «Dedico un bloque de 32 hilos a cada imagen. Los primeros 29 representan las clases. Primero encuentro el máximo, lo resto para estabilizar las exponenciales y después reduzco la suma para normalizar. Los tres hilos sobrantes no representan clases, pero participan en las barreras aportando valores neutros». 

**Recorre:** máximo con `-inf`, barrera, exponenciales, suma compartida, probabilidades, `dlogits`, pérdida y predicción.

El gradiente divide entre `b`, el tamaño real del lote. El lote final de 20 no se normaliza como si tuviera 128. Restar el máximo es estabilidad numérica; no equivale a cambiar la tarea de clasificación.

**Si pregunta por qué no usar solo una imagen por hilo:**

> «Aquí hay una reducción entre las 29 clases de la misma imagen. Un bloque permite que esos hilos cooperen y compartan el denominador».

## 8. Condiciones de carrera, atomics y actualización de pesos

### 8.1 La política general: un escritor por salida

**Abre:** [col2im_gather](../src/asl_cuda/kernels.py#L337) — `src/asl_cuda/kernels.py:337`, líneas 337–349 y [pool_backward](../src/asl_cuda/kernels.py#L136) — `src/asl_cuda/kernels.py:136`, líneas 136–143.

> «En vez de que muchas contribuciones compitan escribiendo el mismo gradiente, cada hilo toma la responsabilidad de una salida y reúne sus contribuciones. Es un patrón gather. Evitamos atomics flotantes para estas acumulaciones». 

En pooling 2×2 stride 2, el backward usa `arg` para enviar el gradiente únicamente al elemento elegido. Los empates se resuelven según el primer máximo encontrado por el forward. El redondeo puede modificar rutas cerca de empates: por eso no se promete trayectoria CPU/GPU idéntica.

### 8.2 Sí hay atomics, pero sobre una bandera entera

**Abre:** [finite_check](../src/asl_cuda/kernels.py#L288) — `src/asl_cuda/kernels.py:288`, líneas 288–291 y [adam](../src/asl_cuda/kernels.py#L273) — `src/asl_cuda/kernels.py:273`, líneas 273–284.

> «Sí utilizamos `cuda.atomic.max`, pero sobre un indicador `int32` de error, no para acumular los gradientes de la red. Si varios hilos detectan un valor no finito, actualizan esa misma bandera de forma atómica. El host la lee después de esperar el stream». 

No afirmar “no hay atomics en todo el proyecto”. Tampoco que un atomic sincroniza todos los hilos ni que revierte una actualización: la bandera detecta el fallo y `train_step` lanza `FloatingPointError`; no constituye una transacción con rollback.

### 8.3 Backward completo antes de Adam

**Abre:** [Network.backward](../src/asl_cuda/model.py#L276) — `src/asl_cuda/model.py:276`, líneas 276–320, [Network.update](../src/asl_cuda/model.py#L322) — `src/asl_cuda/model.py:322`, líneas 322–331 y [adam](../src/asl_cuda/kernels.py#L273) — `src/asl_cuda/kernels.py:273`, líneas 273–284.

> «Primero calculo todos los gradientes con el estado de pesos del lote. Después actualizo los parámetros con Adam. Los buffers `g1` y `g2` se alternan para propagar gradientes sin reservar un tensor nuevo en cada capa». 

`current, other = other, current` intercambia referencias host a buffers GPU; no es una copia D2H. El orden del stream conserva las dependencias aunque los lanzamientos sean asíncronos.

Cada hilo Adam es dueño de un índice de `theta`, `grad`, `moment` y `velocity`. `alpha` incorpora corrección de sesgo con el número de paso. `advance_step` se lanza con 1 bloque/1 hilo para el contador GPU; por eso ese contador no necesita un atomic con múltiples escritores.

## 9. Dónde ocurre la multiplicación: cuBLAS e im2col

**Abre primero:** [Network.gemm](../src/asl_cuda/model.py#L199) — `src/asl_cuda/model.py:199`, líneas 199–220. Después [Blas.gemm](../src/asl_cuda/blas.py#L48) — `src/asl_cuda/blas.py:48`, líneas 48–53.

```python
    def gemm(self, a, b, out, M, N, K, transpose_a=False, transpose_b=False):
        """C_row=op(A)_row @ op(B)_row, implemented as C_col=B_col @ A_col."""
        alpha, beta = ct.c_float(1.), ct.c_float(0.)
        self._call("cublasSgemm_v2", self.handle, int(transpose_b), int(transpose_a),
                   N, M, K, ct.byref(alpha), self.ptr(b), K if transpose_b else N,
                   self.ptr(a), M if transpose_a else K, ct.byref(beta), self.ptr(out), N)
```

**Di exactamente:**

> «Los productos grandes de la versión medida se ejecutan con `cublasSgemm_v2`: multiplicación matricial en precisión simple en GPU. Python prepara los tamaños y pasa las direcciones de los buffers GPU mediante `ctypes`; no está calculando el producto con NumPy en CPU». 

**Qué señalar en `Blas.__init__`:** DLL del toolkit, firmas `ctypes`, `cublasSetStream_v2`, `CUBLAS_PEDANTIC_MATH` y `CUBLAS_ATOMICS_NOT_ALLOWED`. Este último es una política de cuBLAS, distinta de los atomics enteros propios de control de errores.

**Sobre row-major y column-major:**

> «Nuestros arreglos se interpretan en orden fila. El adaptador invierte el orden de operandos y dimensiones para usar la interfaz column-major de cuBLAS, de forma equivalente a transponer el producto». 

No es necesario afirmar que se crea una copia CPU transpuesta: el adaptador cambia la interpretación y los parámetros de la llamada.

**Para las convoluciones:** [im2col](../src/asl_cuda/kernels.py#L323) — `src/asl_cuda/kernels.py:323`, líneas 323–333, [col2im_gather](../src/asl_cuda/kernels.py#L337) — `src/asl_cuda/kernels.py:337`, líneas 337–349 y la rama `mode in (0,2)` de `Network.gemm`.

> «Transformo las ventanas 3×3 en una matriz con im2col, uso SGEMM y reorganizo los gradientes de entrada con gather cuando hace falta. Esto reutiliza un producto matricial optimizado a cambio de memoria temporal». 

CNN-B reserva un workspace de columnas de 576 MiB. Es un coste de memoria de esta ruta, no tamaño del dataset ni “memoria shared”. Los modos Dense 3/4/5 corresponden a forward, gradiente de entrada y gradiente de pesos.

### Qué no presentar como backend medido

[_make_gemm](../src/asl_cuda/kernels.py#L52) — `src/asl_cuda/kernels.py:52`, líneas 52–84 contiene el kernel tiled 16×16 con `sa/sb` shared y dos barreras por tile: una tras cargar y otra antes de reutilizar shared. Es útil si preguntan por multiplicación matricial manual, **pero es la alternativa explícita conservada, no el producto grande usado para los promedios finales**. Para demostrar la ruta medida vuelve al constructor con `backend="cublas"`.

La selección interna de kernels, grids y bloques de SGEMM pertenece a NVIDIA. Nuestro código **no fija** la geometría interna de cuBLAS. Referencia: [cuBLAS13.1](https://docs.nvidia.com/cuda/archive/13.1.0/cublas/index.html).

## 10. Qué cambió para FC-A y FC-B

**Abre:** [FCNetwork](../src/asl_cuda/fc.py#L76) — `src/asl_cuda/fc.py:76`, líneas 76–186 y luego [FCNetwork.reduce](../src/asl_cuda/fc.py#L178) — `src/asl_cuda/fc.py:178`, líneas 178–186.

**Di exactamente:**

> «FC significa totalmente conectada. No es una sola operación: son redes completas con forward, normalización, Dropout, backward y Adam. Heredan el motor CUDA de `Network`; la extensión adapta los buffers, la semilla y la reducción de BatchNormalization». 

| Diferencia que señalar | Dónde |
|---|---|
|FC-A128/64 yDropout 0.3; FC-B512/256/128 yDropout 0.4 |[fc_reference](../src/asl_cuda/fc.py#L41) — `src/asl_cuda/fc.py:41`, líneas 41–73 |
|`backend="cublas"`, stream propio y memoria ajustada |[FCNetwork.__init__](../src/asl_cuda/fc.py#L77) — `src/asl_cuda/fc.py:77`, líneas 77–157 |
|Captura `layer.momentum` |[src/asl_cuda/fc.py:110](../src/asl_cuda/fc.py#L110) |
|Sin workspace im2col: `self.columns=None` |[src/asl_cuda/fc.py:148](../src/asl_cuda/fc.py#L148) |
|Semilla por modelo para las máscaras |[FCNetwork.upload](../src/asl_cuda/fc.py#L160) — `src/asl_cuda/fc.py:160`, líneas 160–176 |
|Reducción BN que respeta momentum |[FCNetwork.reduce](../src/asl_cuda/fc.py#L178) — `src/asl_cuda/fc.py:178`, líneas 178–186 y [finish_fc](../src/asl_cuda/fc.py#L21) — `src/asl_cuda/fc.py:21`, líneas 21–38 |

FC-A yFC-B tienen BN sobre 12,288 características después de Flatten. Su momentum es 0.99; CNN-B usa 0.9. **No se copió el 0.9 de CNN-B a las FC.** Epsilon se conserva 0.001.

**Si pregunta por ausencia de `syncthreads` en `finish_fc`:**

> «En esa función cada hilo termina una característica distinta y usa sus propios acumuladores. No intercambia datos shared con otro hilo, por eso no necesita una barrera interna. La dependencia con el kernel que produce los parciales sigue garantizada por el mismo stream». 

Con `C=12288` y`L=128` o 20, la reducción parcial usa grid `(1,384)`, block `(32,8)`. La final usa 48 bloques de 256 hilos. La implementación FC hereda `train_step`, `forward`, `backward`, `update`, `restore` y `export`; encontrar esos nombres solo en `model.py` es correcto.

Buffers propios FC-A≈49.46 MiB yFC-B≈124.65 MiB. No son iguales a la VRAM observada del proceso/contexto ni acreditan saturación.

## 11. Cómo se midió el tiempo sin medir solo el envío de trabajo

**Abre eventos:** [Network.timed](../src/asl_cuda/model.py#L184) — `src/asl_cuda/model.py:184`, líneas 184–189 y [Network.drain_events](../src/asl_cuda/model.py#L191) — `src/asl_cuda/model.py:191`, líneas 191–197.

> «Registro eventos de inicio y fin en el mismo stream, espero a que terminen y obtengo el intervalo. La API devuelve milisegundos; divido entre 1000 para guardar segundos». 

**Abre tiempo principal:** [scripts/benchmark_cuda.py:233](../scripts/benchmark_cuda.py#L233) hasta la línea 282 del controlador CNN-B; para las FC, usa las celdas de entrenamiento del notebook indicadas más abajo.

**Señala:** `train_start = perf_counter_ns()`, restauración inicial incluida,15 épocas, `net.train_step`, `net.export`, `reference.evaluate` en cada época, exportación final Adam y `train_end`.

> «La cifra principal es tiempo pared del entrenamiento completo, no la suma de tiempos de kernels. Incluye preparación de lotes y máscaras, transferencias, esperas, forward, backward, Adam y validación CPU. El test final se mide aparte. Warmup, preparación inicial de imágenes, guardado y elaboración de gráficas no están dentro de ese fit». 

Las llamadas `train_step` y `export` esperan el stream antes de devolver el control. Por eso el final del intervalo no deja cálculo del último lote pendiente.

**Dos aclaraciones para evitar errores:**

- Los eventos agrupan operaciones y pueden incluir huecos mientras el host envía trabajo. No debes llamarlos «tiempo activo puro de los núcleos GPU».
- Validación y eventos son subintervalos del fit. No se vuelven a sumar al total de entrenamiento. La exportación final copia el modelo una vez adicional; ese coste se incluyó, no se restó.

Referencia de eventos/unidades: [CUDA Host API](https://nvidia.github.io/numba-cuda/reference/host.html#events).

## 12. Warmup, restauración y pruebas: qué mostrar si preguntan por validez

**Abre protocolo CNN-B:** [scripts/benchmark_cuda.py:175](../scripts/benchmark_cuda.py#L175) hasta la línea 209. **Pruebas nuevas FC:** [main](../scripts/validar_cuda_fc.py#L13) — `scripts/validar_cuda_fc.py:13`, líneas 13–79.

> «Antes de cronometrar ejercito lotes 128 y 20 y las rutas de validación/test CPU. Después restauro pesos, estadísticas BN, momentos/velocidades Adam y contador. Verifico la restauración; no inicio la medición con un modelo ya entrenado por el calentamiento». 

**Datos y estado:** muestra las comprobaciones SHA-256 y de tensores en el controlador antes de entrenar. En `cpu_reference`/`fc_reference` se restauran los archivos NPZ históricos por orden de variable, forma y dtype. No dependemos solo de volver a usar la misma semilla.

**Pruebas FC:** `GradientTape`, máscaras compartidas con el oráculo, comparación de probabilidades/gradientes, BN y primera actualización Adam; lotes 20 y 128 para ambas redes.

**Di exactamente sobre el límite de comparación:**

> «Reutilizamos la CPU histórica para no repetir horas de ejecución. Coinciden imágenes, particiones, pesos iniciales, arquitectura, número de pasos y precisión. El orden interno de lotes y las máscaras históricas no son idénticos, y el estado térmico no se alternó por pares. Por eso reportamos el speedup temporal y la calidad por separado». 

No etiquetar las pruebas locales como prueba de igualdad de toda la trayectoria de 15 épocas. Tampoco llamar determinismo general a cinco repeticiones GPU con el mismo estado y RNG.

## 13. Dónde enseñar las salidas reales dentro del notebook

Abre [entregables/notebooks/ASL_CUDA.ipynb](notebooks/ASL_CUDA.ipynb). Los módulos también aparecen como código visible en celdas `%%modulo` / `%%modulo_fc`. Las distintas sesiones están identificadas; los contadores de ejecución se reinician legítimamente.

**Para llegar rápido:** Ctrl+F con `%%modulo asl_cuda.model`, `%%modulo_fc asl_cuda.fc`, `train_start`, `EPOCH FC-A` o `EPOCH FC-B`. Los primeros encuentran código; los últimos, salidas guardadas.

| Celda visual (desde 1) | In [n] de esa sesión | Qué enseñar | Primera línea / búsqueda |
|---|---|---|---|
|12|8|Módulo runtime CNN-B|`%%modulo asl_cuda.runtime`|
|13|9|Kernels CNN-B|`%%modulo asl_cuda.kernels`|
|14|10|Adaptador cuBLAS CNN-B|`%%modulo asl_cuda.blas`|
|15|11|Clase Network CNN-B|`%%modulo asl_cuda.model`|
|18|13|Creación Network, backend medido|`conf, folder = locations(experiment)`|
|24|17|Preparación CPU|`state('preparacion_cpu', workers=6)`|
|29|20|Bucle real de entrenamiento|`state('entrenamiento_CNN-B', epoca=1, pasos=0)`|
|60|4|Kernels reutilizados FC|`%%modulo_fc asl_cuda.kernels`|
|61|5|cuBLAS reutilizado FC|`%%modulo_fc asl_cuda.blas`|
|62|6|Network reutilizada FC|`%%modulo_fc asl_cuda.model`|
|63|7|Extensión FCNetwork|`%%modulo_fc asl_cuda.fc`|
|67|9|Creación FCNetwork / warmup|`name='FC-A';repetition=1;run_id='RUN-20261118T134323Z-fc-a-01'`|
|68|10|Preparación CPU|`state('preparacion_cpu',workers=6)`|
|69|11|Bucle real de entrenamiento|`state('entrenamiento_'+name,epoca=1,pasos=0)`|
|79|4|Kernels reutilizados FC|`%%modulo_fc asl_cuda.kernels`|
|80|5|cuBLAS reutilizado FC|`%%modulo_fc asl_cuda.blas`|
|81|6|Network reutilizada FC|`%%modulo_fc asl_cuda.model`|
|82|7|Extensión FCNetwork|`%%modulo_fc asl_cuda.fc`|
|86|9|Creación FCNetwork / warmup|`name='FC-A';repetition=2;run_id='RUN-20261118T134514Z-fc-a-02'`|
|87|10|Preparación CPU|`state('preparacion_cpu',workers=6)`|
|88|11|Bucle real de entrenamiento|`state('entrenamiento_'+name,epoca=1,pasos=0)`|
|91|13|Creación FCNetwork / warmup|`name='FC-A';repetition=3;run_id='RUN-20261118T134514Z-fc-a-03'`|
|92|14|Preparación CPU|`state('preparacion_cpu',workers=6)`|
|93|15|Bucle real de entrenamiento|`state('entrenamiento_'+name,epoca=1,pasos=0)`|
|96|17|Creación FCNetwork / warmup|`name='FC-A';repetition=4;run_id='RUN-20261118T134514Z-fc-a-04'`|
|97|18|Preparación CPU|`state('preparacion_cpu',workers=6)`|
|98|19|Bucle real de entrenamiento|`state('entrenamiento_'+name,epoca=1,pasos=0)`|
|101|21|Creación FCNetwork / warmup|`name='FC-A';repetition=5;run_id='RUN-20261118T134514Z-fc-a-05'`|
|102|22|Preparación CPU|`state('preparacion_cpu',workers=6)`|
|103|23|Bucle real de entrenamiento|`state('entrenamiento_'+name,epoca=1,pasos=0)`|
|107|25|Creación FCNetwork / warmup|`name='FC-B';repetition=1;run_id='RUN-20261118T134514Z-fc-b-01'`|
|108|26|Preparación CPU|`state('preparacion_cpu',workers=6)`|
|109|27|Bucle real de entrenamiento|`state('entrenamiento_'+name,epoca=1,pasos=0)`|
|112|29|Creación FCNetwork / warmup|`name='FC-B';repetition=2;run_id='RUN-20261118T134514Z-fc-b-02'`|
|113|30|Preparación CPU|`state('preparacion_cpu',workers=6)`|
|114|31|Bucle real de entrenamiento|`state('entrenamiento_'+name,epoca=1,pasos=0)`|
|117|33|Creación FCNetwork / warmup|`name='FC-B';repetition=3;run_id='RUN-20261118T134514Z-fc-b-03'`|
|118|34|Preparación CPU|`state('preparacion_cpu',workers=6)`|
|119|35|Bucle real de entrenamiento|`state('entrenamiento_'+name,epoca=1,pasos=0)`|
|122|37|Creación FCNetwork / warmup|`name='FC-B';repetition=4;run_id='RUN-20261118T134514Z-fc-b-04'`|
|123|38|Preparación CPU|`state('preparacion_cpu',workers=6)`|
|124|39|Bucle real de entrenamiento|`state('entrenamiento_'+name,epoca=1,pasos=0)`|
|127|41|Creación FCNetwork / warmup|`name='FC-B';repetition=5;run_id='RUN-20261118T134514Z-fc-b-05'`|
|128|42|Preparación CPU|`state('preparacion_cpu',workers=6)`|
|129|43|Bucle real de entrenamiento|`state('entrenamiento_'+name,epoca=1,pasos=0)`|

La numeración anterior cuenta también las celdas Markdown y corresponde al hash del notebook registrado al crear esta guía. Si se añaden celdas después, las búsquedas por función/magic siguen siendo la referencia más estable. Las celdas de recuperación/presentación de resultados se identifican como lectura de archivos; no son un entrenamiento nuevo.

**Di exactamente al mostrar una salida:**

> «Estas salidas quedaron registradas durante las ejecuciones reales. Aquí podemos comprobar las 15 épocas, los 1440 pasos, las métricas y las capturas. Para revisar evidencia no necesitamos repetir el entrenamiento».

## 14. Resultados y evidencias: archivos exactos para responder con pruebas

| Si pregunta… | Abre |
|---|---|
|¿Cuánto aceleró cada red? |[comparacion.csv](../resultados/comparacion/ampliacion-fc/comparacion.csv): filas `entrenamiento_*`. |
|¿La exactitud se mantuvo? |[calidad.csv](../resultados/comparacion/ampliacion-fc/calidad.csv), y[calidad CNN-B](../resultados/paralelo/EXP-003/calidad_comparada.csv). |
|¿Hay varias repeticiones yDE? |[estadisticas.csv](../resultados/comparacion/ampliacion-fc/estadisticas.csv) y[observaciones.csv](../resultados/comparacion/ampliacion-fc/observaciones.csv). |
|¿Se verificaron checkpoints ypasos? |[validacion-final.json](../resultados/preparacion/ampliacion-fc/validacion-final.json). |
|¿Se comprobaron gradientes FC? |[validación del oráculo FC](../resultados/preparacion/ampliacion-fc/validacion-01/validacion.json). |
|¿La GPU trabajó de verdad? |[captura CNN-B](../resultados/paralelo/EXP-009/RUN-20261118T041129Z-cuda-notebook-01/captura-cuda.png), y su[JSON](../resultados/paralelo/EXP-009/RUN-20261118T041129Z-cuda-notebook-01/captura-cuda.json). |
|¿También trabajaron FC-A yFC-B? |[captura FC-A](../resultados/paralelo/EXP-011/RUN-20261118T134323Z-fc-a-01/captura-fc-a.png) y[captura FC-B](../resultados/paralelo/EXP-012/RUN-20261118T134514Z-fc-b-01/captura-fc-b.png). |
|¿Qué se copió por lote? |[transferencias FC-B](../resultados/paralelo/EXP-012/RUN-20261118T134514Z-fc-b-01/transferencias.json). |
|¿Qué geometrías yrecursos se registraron? |[lanzamientos FC-B](../resultados/paralelo/EXP-012/RUN-20261118T134514Z-fc-b-01/lanzamientos.json). |

**Resumen para decir:**

> «Con cinco corridas por modelo, CNN-B pasó de 805.65 a 418.61 segundos; FC-A, de 50.72 a 15.12; yFC-B, de 102.44 a 21.06. Los speedups son 1.92,3.35 y 4.86 veces. Incluyen transferencias y validación CPU. Esto no es el speedup de todo el notebook original». 

> «Las FC tuvieron menor exactitud que la referencia CPU: lo reportamos, no lo ocultamos. La mejora demostrada es temporal. Tampoco afirmamos saturación total deVRAM ni ocupación por SM: los monitores muestran actividad agregada y memoria usada». 

La capturaCNN-Bdel 91% pertenece a una corrida adicional n=1 del notebook; no reemplaza los cinco RUN del promedio. Registros compilados de registros/shared/local no equivalen a medir ocupaciónSM ni prueban que 256 hilos sea el tamaño óptimo global.

## 15. Preguntas técnicas probables y respuestas listas

| Pregunta | Respuesta que puedes decir | Dónde apuntar |
|---|---|---|
|¿Dónde está CUDA si todo esPython? |«En las funciones `@cuda.jit` ylos lanzamientos explícitos. Numba-CUDA compila el kernel; Pythonhost organiza el flujo». |[dropout](../src/asl_cuda/kernels.py#L113) — `src/asl_cuda/kernels.py:113`, líneas 113–115 |
|¿Por qué algunas funciones no llevan `@cuda.jit`? |«Son controladores del host. Dentro llaman kernels y cuBLAS; no toda líneaPython se ejecuta en GPU». |[Network.forward](../src/asl_cuda/model.py#L251) — `src/asl_cuda/model.py:251`, líneas 251–274 |
|¿`syncthreads` espera toda la GPU? |«No. Solo los hilos del mismo bloque». |[reduce_channels](../src/asl_cuda/kernels.py#L147) — `src/asl_cuda/kernels.py:147`, líneas 147–179 |
|¿El stream hace esperar al CPU? |«Encolar en el stream no implica esperar; `stream.synchronize` sí hace esa espera». |[Network.drain_events](../src/asl_cuda/model.py#L191) — `src/asl_cuda/model.py:191`, líneas 191–197 |
|¿Usaron varios streams? |«No. Uno por red yel mismo en cuBLAS, para conservar dependencias. Sin pipeline de lotes solapados». |[src/asl_cuda/blas.py:28](../src/asl_cuda/blas.py#L28) |
|¿Usaron shared? |«Sí, en reducciones BN, softmax ymétricas. ElGEMMtiled también tiene shared, pero esalternativo albackendcuBLASmedido». |[reduce_channels](../src/asl_cuda/kernels.py#L147) — `src/asl_cuda/kernels.py:147`, líneas 147–179 |
|¿Por qué 256 hilos? |«Es la configuración aplicada:8 warps yrecorrido con salto. No afirmamos haber demostrado el óptimo de ocupación». |[grid](../src/asl_cuda/model.py#L60) — `src/asl_cuda/model.py:60`, líneas 60–61 |
|¿Por qué 32 hilos ensoftmax sihay 29 clases? |«Un bloque cubre las clases de una imagen; los hilos sobrantes aportan neutros yparticipan enbarreras». |[softmax_loss](../src/asl_cuda/kernels.py#L230) — `src/asl_cuda/kernels.py:230`, líneas 230–269 |
|¿Todas las operaciones están paralelizadas? |«La matemática delentrenamiento deCNN-B, FC-AyFC-B está en GPU. Preparación, validación, test yotras ramas permanecen CPU». |[load_images](../scripts/benchmark_cuda.py#L102) — `scripts/benchmark_cuda.py:102`, líneas 102–128 |
|¿cuBLAS es hacer trampa o usar CPU? |«Es una biblioteca CUDA deNVIDIA que ejecutaSGEMMen GPU. Nuestroskernels manejan otras operaciones; separamos ambasresponsabilidades». |[Blas.gemm](../src/asl_cuda/blas.py#L48) — `src/asl_cuda/blas.py:48`, líneas 48–53 |
|¿Se perdió la sincronización con cuBLAS? |«No; su handle recibe exactamenteel mismo stream que los kernels». |[src/asl_cuda/blas.py:28](../src/asl_cuda/blas.py#L28) |
|¿Hay atomics? |«Sí, sobrela banderaentera deerror. No acumulamos gradientes conatomicsflotantes ennuestroskernels». |[finite_check](../src/asl_cuda/kernels.py#L288) — `src/asl_cuda/kernels.py:288`, líneas 288–291 |
|¿Cómo procesan el último lote? |«Su tamaño real es 20; la pérdida y las dimensiones se calculan con ese tamaño». |[Network.train_step](../src/asl_cuda/model.py#L333) — `src/asl_cuda/model.py:333`, líneas 333–346 |
|¿Por qué exportan cada época? |«Para validar en CPU manteniendo CUDA enentrenamiento; esecosteestá incluido». |[Network.export](../src/asl_cuda/model.py#L351) — `src/asl_cuda/model.py:351`, líneas 351–377 |
|¿Por qué las FC usanmenos GPU yaceleran más? |«Su referencia CPU y su carga son distintas; el speedup es un cociente de tiempos, no un porcentaje deocupación». |Desglose ycomparación del apartado 14. |
|¿Qué optimización futura harían? |«Evaluaría fusión de kernels, reducir esperas por lote ysolapar transferencias donde hayaindependencia. Son propuestas, no resultados medidos aquí». |Final de `train_step`, no prometer speedup futuro. |

## 16. Frases que conviene evitar y su versión correcta

| Evita afirmar | Di en su lugar |
|---|---|
|«Todo el Python corre en GPU». |«El host lanza kernels CUDA y SGEMM; preparación y validación siguen en CPU». |
|«`syncthreads` sincroniza todos los bloques». |«Sincroniza hilos del mismo bloque; entre etapas usamos kernels separados en el mismo stream». |
|«Como hay stream, cálculo ycopias se solapan». |«Hay un solo stream yuna espera por lote; no se implementó solapamiento». |
|«Los eventos miden tiempo puro denúcleos». |«Miden intervalos del stream; el principal es tiempo pared completo». |
|«El kernel tiled 16×16 produjo el speedup final». |«La serie final utiliza cuBLAS; el tiled es unbackend alternativo conservado». |
|«cuBLAS usa nuestros mismos 256 hilos». |«La biblioteca gestiona su propia jerarquía interna». |
|«No hay atomics». |«Hay atomics enteros para errores; sin atomics flotantes propios para acumular gradientes». |
|«Pinned significa memoria GPU». |«Es memoria host fijada; se copia explícitamente a buffers device». |
|«La GPU estuvo 91%ocupada por SM». |«nvidia-smi mostró 91%de actividad agregada en esa muestra». |
|«Se mantuvo exactamente la accuracy». |«Se mantuvo la carga; calidad y trayectorias se reportan por separado». |
|«Las presentaciones, guías y gráficas requieren entrenar otra vez». |«Son lecturas y documentación de resultados guardados». |

## 17. Cierre literal de tu demostración

> «La contribución no es únicamente activar una GPU. Separamos preparación CPU y entrenamiento GPU, definimos kernels y su jerarquía, controlamos memoria y dependencias, unificamos el stream con cuBLAS y medimos el coste completo, incluidas transferencias y validación. La espera por lote hace que los resultados y los tiempos se lean después de terminar el trabajo. Es una implementación verificable para tres redes, con los límites numéricos y de calidad declarados».

## Documentación técnica de apoyo

Las ubicaciones de código anteriores son la evidencia principal de **lo implementado**. Las referencias oficiales solo respaldan el significado de las API; no sustituyen las mediciones locales.

- NVIDIA. *Writing CUDA Kernels*, Numba-CUDA: [modelo de ejecución y lanzamientos](https://nvidia.github.io/numba-cuda/user/kernels.html).
- NVIDIA. *CUDA Kernel API*, Numba-CUDA: [sincronización y atomics](https://nvidia.github.io/numba-cuda/reference/kernel.html).
- NVIDIA. *Memory management*, Numba-CUDA: [pinned, shared y streams](https://nvidia.github.io/numba-cuda/user/memory.html).
- NVIDIA. *CUDA Host API*, Numba-CUDA: [dispositivo yeventos](https://nvidia.github.io/numba-cuda/reference/host.html).
- NVIDIA. *cuBLAS Library, CUDA Toolkit 13.1*: [stream,SGEMM ydisposición de matrices](https://docs.nvidia.com/cuda/archive/13.1.0/cublas/index.html).

Consulta de apoyo realizada durante esta inspección documental. No se ejecutó ningún kernel para crear la guía.
