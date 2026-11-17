# Contraste de DIS-CUDA-001 con la clase Semana15

Estado: **contraste documental completado**, sin implementar ni ejecutar kernels.
Fecha:2026-11-17 UTC. Solicitud complementaria del usuario durante el paso4.
Relaciones: REQ-05/12, DEC-020, CAM-005, BIT-007.

## Fuente, conservación y precedencia

Original consultado:
`C:\Users\David\Documents\2026\S2\PARALELA\CUDA\PROYECTO-CUDA\semana_15_programaci_n_con_cuda.md`.
Se preservó una [copia exacta dentro del repositorio](../referencias/semana-15-programacion-cuda.md)
para que el contraste no dependa de una ruta externa.16005bytes,341líneas,
SHA-256`acce03142813c9aca0bf4329000639bb2bb986f92e2810bd1c291774e73fab26`.
Se conservó su texto y atribución, sin editar la presentación.

Precedencia: **PDF del proyecto → decisiones del proyecto sustentadas en fuentes
técnicas y evidencia propia → presentación como apoyo pedagógico**. La clase
no impone lenguaje C++,512threads, transferir todo al inicio ni usar streams
concurrentes. El PDF sigue exigiendo CUDA Python y separación CPU/entrenamientoGPU.

## Correspondencia con los temas vistos

| Sección de la presentación | Correspondencia concreta con el diseño | Valoración |
|---|---|---|
| 1. CUDA como coprocesador | CPU dirige preparación/control; GPU ejecuta matemática de entrenamiento CNN-B | Alineado con el modelo Host/Device |
| 2. Paralelismo masivo | Convoluciones, Dense, gradientes y operaciones por elemento sobre lotes | Alineado; no se generaliza un número de núcleos a todo hardware |
| 3. Host y kernels | Coordinador Python y contratos de kernels K01–K19; separación de código prevista | Mismo modelo, sintaxis Python por la consigna |
| 4. Grid/block/thread/warp | G1D, G16, G16S, R y RF; índice local/global, slabs y descomposición en warps | Alineado; se añade precisión SIMT y sincronización explícita |
| 5. Geometría según problema/recursos | Bloques16×16 y32×8 de256threads,32para softmax29,1para control; grids por formas reales | Criterios aplicados, sin asumir que256sea óptimo universal |
| 6. Barreras | Tiles y reducciones usan barreras uniformes; parcial global se consume en kernel posterior | Alineado, sin usar barrera de block como barrera de grid |
| 7. Divergencia | Colas neutras, guardas de borde, bucles de reducción comunes y flags limitados | Se minimiza donde conviene, sin eliminar comprobaciones necesarias |
| 8. Memorias | Activaciones/estados globales, tiles shared, escalares privados; recursos compilados se comprobarán | Alineado, distinguiendo registros de espacio local |
| 9. Coalescencia | NHWC, thread.x sobre columnas/canales contiguos; reducción32×8 revisada | Ajuste incorporado al diseño a partir del contraste |
| 10. Transferencias | Lotes contiguos, pesos/intermedios residentes, payload real; H2D/D2H cronometrados | Principios aplicados con compromisos documentados |
| 11. API | Correspondencias conceptuales CUDA C→CUDA Python y vida útil explícita | Alineado sin copiar sintaxis C++ al proyecto |

## Ajuste incorporado: reducción por canales con lectura contigua

La primera propuesta interna de R(L,C) asignaba un block1D a un canal y cada
lane a una fila NHWC distinta. Era correcta matemáticamente, pero las lecturas
de lanes consecutivas tenían strideC. El contraste hace conveniente mejorar ese
acceso **antes de implementar**:

- Block=(32,8,1),256threads; grid=(ceil(L/2048),ceil(C/32),1).
- Thread(tx,ty): canal c=32blockIdx.y+tx; posiciones
  l=2048blockIdx.x+ty+8j, j=0..255, con guardas de forma.
- Cada warp de32lanes lee32canales contiguos de una fila. Las ocho filas de
  threads acumulan partes distintas; reducción por ty mediante shared[8,32]
  y barreras uniformes. ty0 escribe el parcial si c<C.
- Mantiene exactamente una contribución por par(l,c), el mismo tamaño de
  partición2048, parciales≤256 por canal y el mismo presupuesto shared/global.
- La fusión final RF por canal conserva block256. Su acceso de parciales puede
  tener strideC, pero afecta a un volumen mucho menor. La convolución dX también
  mantiene gathers que no garantizan coalescencia total.
- Compromiso: más trabajo secuencial por thread y posible cambio de redondeo de
  sumas; debe pasar las tolerancias. **No se presenta como mejora medida**.

Se actualizaron catálogo, tabla de distribución y ejemplos B1/B4. Este refinamiento
pertenece a la revisión1 aún no implementada; no cambia ningún experimento previo.
No se alteran parámetros de red, datos, batch, payload o cantidad de épocas.

## Precisiones para aplicar con cautela las reglas de clase

Estas notas matizan simplificaciones pedagógicas; no convierten la presentación
en norma del proyecto ni modifican su copia original.

| Idea de la clase | Aplicación cautelosa |
|---|---|
| Warp descrito como SIMD y simultaneidad rígida | CUDA es SIMT; no se presupone avance lockstep para compartir datos. Se mantienen sincronizaciones explícitas. |
| 256/512threads maximizarían ocupación | Son puntos de partida; registros, shared y blocks residentes condicionan la ocupación. Mayor ocupación tampoco garantiza más velocidad. |
| Ternario equivaldría a ausencia de divergencia | El compilador decide predicación o saltos. No cambiar if→ternario como garantía de rendimiento. |
| Memoria local presentada junto a registros rápidos | El espacio local es privado, pero reside fuera del chip y puede cachearse; variable local del fuente no implica registro. |
| Una tarea por elemento exigiría igual número de threads | El recorrido por grid-stride permite varios elementos por thread. G1D cap4096blocks sigue cubriendo todos los elementos. |
| No habría sincronización entre blocks en un kernel | No se dispone de ella con una barrera normal de block. Existen mecanismos especializados, excluidos de este diseño; aquí se separan kernels. |
| Transferir todo al principio y usar asincronía | Son opciones sujetas a memoria/equivalencia; no requisitos. Esta propuesta usa lotes y un stream ordenado para auditar costes. |
| Salida printf mostrada ordenada | Es ilustrativa: el orden de ejecución/impresión entre threads no se usa como garantía de corrección. |

NVIDIA distingue [ocupación y parámetros de ejecución](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy),
[memoria local](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#local-memory)
y [predicación](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#branch-predication).
El [modelo oficial CUDA](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)
apoya la interpretación de jerarquía y SIMT. Se consultaron como fuentes técnicas
primarias complementarias al material del curso, el2026-11-17.

## Correspondencia de conceptos C++ a la API propuesta

Esta tabla es de diseño; las expresiones nombran APIs, no implementan kernels.

| Concepto ilustrado en clase | Equivalente previsto |
|---|---|
| __global__ | Kernel Python compilado mediante numba.cuda.jit |
| <<<grid,block>>> | Configuración explícita grid/block del dispatcher CUDA Python |
| threadIdx/blockIdx/blockDim/gridDim | cuda.threadIdx/cuda.blockIdx/cuda.blockDim/cuda.gridDim |
| cudaMalloc | cuda.device_array con dtype/shape explícitos |
| cudaMemcpy H2D/D2H | Copias explícitas hacia buffers device y copy_to_host, con stream y payload registrados |
| __shared__ | cuda.shared.array con forma y tipo estáticos |
| __syncthreads | cuda.syncthreads uniforme dentro de tiles/reducciones |
| cudaDeviceSynchronize | Sincronización del stream elegido para sus fases; contexto solo en fronteras que lo necesiten |
| cudaFree | Vida de buffers gestionada por Numba; liberar referencias/cerrar contexto al terminar tras sincronizar; no mezclar liberadores de distintos propietarios |

La [guía Numba-CUDA](https://nvidia.github.io/numba-cuda/user/kernels.html) y su
[API de memoria](https://nvidia.github.io/numba-cuda/user/memory.html) sustentan la
correspondencia. No se ejecuta nvcc sobre un .py ni se usa CUDA C por el hecho de
que los ejemplos de clase estén escritos en C++.

## Decisiones que se mantienen tras el contraste

- Bloques256para trabajo grande,32para las29clases de softmax,1para controles
  escalares. El warp parcial de29clases es una cola necesaria y no un fallo de
  cobertura; no se lanzan256threads para obtener más paralelismo ficticio ahí.
- Mantener activaciones, pesos y estados Adam en VRAM entre kernels/pasos.
  Solo los lotes train y máscaras entran, y el modelo sale por época para validar
  en CPU. No se copian activaciones intermedias hacia Host entre capas.
- Streaming de lotes limita la reserva fija de VRAM y explicita transferencias.
  Es posible estudiar residencia del train completo en otra variante, pero no
  se asume que quepa bajo toda carga del escritorio ni se oculta su coste inicial.
- Un stream sin solapamiento facilita auditar dependencias y tiempos. La
  asincronía concurrente se deja como optimización posterior con revisión/medición,
  sin presentarla como implementada ni necesaria para satisfacer la consigna.
- No reemplazar guardas por accesos fuera de rango para evitar divergencia.
  En tiles, lanes no válidas aportan0 y alcanzan todas las barreras.

Resultado: el diseño queda en sintonía con lo visto en clase, con un refinamiento
de coalescencia y límites explícitos. Esta conclusión es documental; rendimiento,
ocupación, compatibilidad y corrección de una implementación siguen pendientes.
