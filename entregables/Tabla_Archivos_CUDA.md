# Archivos CUDA: función, uso y orden para exponer

Estos son los **archivos principales de la implementación CUDA final**. Algunos contienen cálculos GPU; otros los configuran o coordinan desde el CPU.

## Tabla de archivos

| Archivo | ¿Qué hace? | ¿Qué utiliza de CUDA? | ¿Cómo y por qué? |
|---|---|---|---|
| [runtime.py](../src/asl_cuda/runtime.py) | Prepara el acceso a CUDA y detecta la GPU. | `numba.cuda`, detección del dispositivo y consulta de memoria. | Configura las DLL del toolkit, comprueba disponibilidad y rechaza el simulador. Permite identificar el hardware real utilizado. |
| [model.py](../src/asl_cuda/model.py) | Es el **motor de entrenamiento**, inicialmente para CNN-B y reutilizado por las FC. | Memoria GPU, memoria host pinned, stream, eventos, transferencias y lanzamientos de kernels. | Ordena **H2D → forward → backward → Adam → control de errores**. Espera al finalizar cada lote y exporta los pesos para validar en CPU. |
| [kernels.py](../src/asl_cuda/kernels.py) | Contiene nuestras funciones matemáticas ejecutadas por hilos GPU. | `@cuda.jit`, índices de hilos/bloques, memoria shared, `syncthreads()` y atomics enteros. | Implementa Dropout, normalización, pooling, softmax, pérdida, Adam y otras operaciones. Reparte elementos y coordina las reducciones dentro de cada bloque. |
| [blas.py](../src/asl_cuda/blas.py) | Conecta Python con **cuBLAS**, la biblioteca matricial de NVIDIA. | `cublasSgemm_v2`, punteros a memoria GPU y `cublasSetStream_v2`. | Ejecuta las multiplicaciones grandes en FP32. Usa el **mismo stream** que nuestros kernels para conservar las dependencias. La geometría interna la decide NVIDIA. |
| [fc.py](../src/asl_cuda/fc.py) | Adapta el motor para entrenar **FC-A y FC-B completas**. | Hereda el entrenamiento CUDA; añade el kernel `finish_fc` y ajusta buffers, transferencias y reducciones. | Reutiliza `Network`, elimina el workspace de convoluciones y conserva el momentum BN de las FC: **0.99**, frente a 0.9 en CNN-B. |
| [benchmark_cuda.py](../scripts/benchmark_cuda.py) | Controla las mediciones CNN-B y prepara las imágenes en CPU. | Crea `Network(..., backend="cublas")` y solicita pasos de entrenamiento, eventos y exportaciones. | Realiza calentamiento, restaura el estado y registra tiempos, métricas y recursos. Permite medir varias corridas bajo un protocolo definido. |
| [preparar_experimento_fc_v3.py](../scripts/preparar_experimento_fc_v3.py) | Construye el notebook y la configuración de las pruebas FC pendientes. | Incorpora el código que crea `FCNetwork` y llama a `train_step`. | **Preparar este archivo no entrena.** Genera las celdas que posteriormente ejecutan las pruebas nuevas, reutilizando la primera FC-A ya guardada. |
| [ejecutar_fc_notebook_v3.py](../scripts/ejecutar_fc_notebook_v3.py) | Ejecuta las celdas del notebook FC y conserva sus salidas. | Uso indirecto: las celdas llaman al motor CUDA. | Coordina el kernel de Jupyter; la matemática GPU sucede en `fc.py`, `model.py`, `kernels.py` y cuBLAS. |
| [validar_cuda_fc.py](../scripts/validar_cuda_fc.py) | Comprueba la implementación FC frente a una referencia CPU. | Ejecuta operaciones de `FCNetwork` y recupera resultados GPU. | Compara probabilidades, gradientes, normalización y Adam para lotes de **20 y 128**. Comprueba también la restauración. |
| [ASL_CUDA.ipynb](notebooks/ASL_CUDA.ipynb) | Es la entrega visible del código y las ejecuciones. | Incluye módulos CUDA, lanzamientos y entrenamiento por fases. | Reúne **código, salidas reales, tiempos, gráficas y capturas**. Es el mejor lugar para demostrar la ejecución ya registrada. |

## Ubicar rápidamente la sincronización

| Necesidad | Dónde está | Qué significa |
|---|---|---|
| Coordinar hilos del mismo bloque | [kernels.py:147](../src/asl_cuda/kernels.py#L147), función `reduce_channels`; barreras en las líneas 169 y 175. | `cuda.syncthreads()` espera a los hilos del **mismo bloque** antes de continuar la reducción. |
| Mantener el orden entre copias, kernels y matrices | [model.py:82](../src/asl_cuda/model.py#L82) y [fc.py:83](../src/asl_cuda/fc.py#L83) crean el stream; [blas.py:28](../src/asl_cuda/blas.py#L28) asocia cuBLAS al mismo stream. | Las operaciones dependientes se encolan en orden. Esto no significa que el CPU espere después de cada lanzamiento. |
| Hacer esperar al CPU al terminar el lote | [model.py:191](../src/asl_cuda/model.py#L191), función `drain_events()`, llamada desde [train_step:333](../src/asl_cuda/model.py#L333). | `self.stream.synchronize()` espera el trabajo previo de ese stream antes de leer resultados y tiempos. |

**Frase para recordar:**

> La barrera coordina hilos de un bloque; el stream ordena operaciones GPU; `stream.synchronize()` hace esperar al CPU.

Se utiliza **un solo stream por red y una espera al final de cada lote**. No se implementó un pipeline de varios lotes solapados mediante múltiples streams.

## Orden recomendado para exponer el código

**Empieza por `model.py` → muestra un kernel en `kernels.py` → explica `blas.py` → termina con la adaptación de `fc.py`.**

| Paso | Qué abrir | Qué decir |
|---|---|---|
| 1. Recorrido del lote | [model.py:333](../src/asl_cuda/model.py#L333), `Network.train_step()` | «Aquí se organiza el entrenamiento de cada lote: copiar datos, calcular forward y backward, actualizar con Adam y esperar a que termine el stream». |
| 2. CUDA explícito | [kernels.py:113](../src/asl_cuda/kernels.py#L113), `dropout()`; muestra también el decorador de la línea anterior. | «Esta función se compila para GPU. Cada hilo procesa elementos del arreglo usando su índice global y un recorrido con salto». |
| 3. Sincronización | [kernels.py:147](../src/asl_cuda/kernels.py#L147), `reduce_channels()`, y [model.py:191](../src/asl_cuda/model.py#L191), `drain_events()` | «Dentro del bloque usamos barreras para compartir sumas parciales. Al terminar el lote, el CPU espera al stream para leer resultados completos». |
| 4. Multiplicaciones grandes | [blas.py:48](../src/asl_cuda/blas.py#L48), `Blas.gemm()`, y la asociación del stream en la línea 28. | «Los productos matriciales grandes se ejecutan en GPU mediante cuBLAS SGEMM. La biblioteca usa el mismo stream que nuestros kernels para respetar las dependencias». |
| 5. Redes adicionales | [fc.py:76](../src/asl_cuda/fc.py#L76), `FCNetwork`, y [fc.py:178](../src/asl_cuda/fc.py#L178), `reduce()`. | «FC-A y FC-B heredan el motor CUDA. Adaptamos la memoria, las semillas y la normalización, conservando el momentum original de 0.99». |

Para saltar a una ubicación en VS Code: **Ctrl+P → escribir `src/asl_cuda/model.py:333` → Enter**. No es necesario volver a ejecutar el entrenamiento para mostrar el código o sus salidas guardadas.

Para ampliar la explicación, consulta la [guía técnica completa](Guia_Codigo_CUDA.md).
