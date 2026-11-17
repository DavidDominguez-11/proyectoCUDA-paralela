# Estrategia de paralelización

Estado: **pasos 3 y 4 documentados; diseño DIS-CUDA-001 completo como propuesta;
implementación y validación pendientes**.
Relaciones: REQ-04/05/06; DEC-003/004/006/007/008/017/018/019.
Stack propuesto: NVIDIA numba-cuda con kernels Python explícitos, sin instalar.

## Diseño vigente

La propuesta completa está en [diseno-paralelo.md](diseno-paralelo.md), el
[catálogo de kernels](diseno-kernels-cuda.md) y el
[plan de validación/medición](diseno-validacion-cuda.md).
La [inspección](../../resultados/diseno/paso-4-inspeccion.md) registra hardware,
fuentes locales e identidades. CAM-005, BIT-007. Los tamaños son propuestos,
no configuraciones ejecutadas ni óptimos medidos.

La selección vigente y su evidencia se detallan en el
[paso 3](oportunidades-paralelizacion.md): prioridad a la matemática del
entrenamiento de CNN-B (71.94% de la referencia común CPU), con operaciones
análogas de otras redes como oportunidades secundarias. DEC-003/016, CAM-004.
Las secciones siguientes resumen el inventario y enlazan las fichas concretas.

## Separación de responsabilidades

CPU: archivos, lectura/decodificación, resize, normalización, etiquetas y
particiones; preparación con6workers acotados y orden fijo. Validación y evaluación
en CPU. No se añade limpieza ni augmentation al alcance CNN-B base.

GPU/CUDA: matemática del entrenamiento del modelo seleccionado. Preservar datos,
modelo, pesos iniciales, pérdida, optimizador y precisión respecto a la referencia.
Las copias de modelo para validación CPU y los estados finales están especificados.

Flujo que debe concretar el diseño: archivos → preparación CPU → tensores Host →
H2D → tensores Device → forward/backward/actualización → D2H → resultados y métricas.

## Candidatos y dependencias

| Operación | Paralelismo potencial | Dependencia / riesgo | Selección actual |
|---|---|---|---|
| Convoluciones | Muestras, salida espacial y filtros | Reducción por canales/kernel y dependencia entre capas | Seleccionada en CNN-B |
| Multiplicaciones densas | Muestras y unidades de salida | Reducción sobre entradas | Seleccionada en CNN-B |
| Activaciones | Elementos independientes | Ancho de banda y coste de lanzamiento | Candidata como parte de entrenamiento |
| Pooling | Ventanas/salidas | Bordes, índices para backward y empates | Candidata |
| Loss/softmax | Muestras/clases con reducciones | Estabilidad numérica y sincronización | Candidata |
| Gradientes y Adam | Parámetros/elementos | Estado del optimizador y orden de actualización | Candidata |
| BatchNormalization | Canales/elementos | Reducciones y estadísticas móviles | Candidata si el modelo la usa |
| Preparación de imágenes | Archivos independientes | I/O, workers, orden y etiquetas | CPU según PDF |
| PCA / Random Forest | Álgebra matricial / árboles | Menor prioridad temporal; RF ya solicita paralelismo CPU | Fuera del objetivo CUDA seleccionado |

Las épocas y las actualizaciones sucesivas dependen de los pesos anteriores.
No describirlas como tareas completamente independientes. Identificar qué
trabajo es paralelo dentro de cada paso y qué orden debe preservarse.

## Fichas del diseño propuesto

Las fichas K01–K19 cubren las nueve oportunidades seleccionadas; no existe aún
operación implementada. La tabla enlaza los contratos ya definidos.

| Campo | Valor |
|---|---|
| Operación / función CPU / contrato CUDA | Celdas33–34 y catálogo K01–K19; nombres futuros sin código |
| Entradas/salidas, dimensiones, dtype y layout | Diseño principal, secciones3/4; FP32 NHWC |
| Trabajo que resuelve un hilo | Catálogo, definición individual por familia |
| Dimensiones de bloque y número de hilos | GEMM16×16; reducción inicial32×8 y final/vector256; softmax32; control1 |
| Fórmula y dimensiones del grid | G1D/G16/G16S/R/RF del catálogo; slabs16384filas |
| Índices, límites y último bloque incompleto | Catálogo A–K, guards y b real20/128 |
| Reducciones, barreras, atómicos y prevención de carreras | Parciales en dos pases, stream único, sin atómicos flotantes |
| Memoria global/compartida/registros y bytes estimados | Diseño sección8 y catálogo; recursos compilados por comprobar |
| Vida de buffers, asignaciones y reutilización | Arenas persistentes; activaciones retenidas; gradientes ping-pong |
| H2D/D2H: qué, cuánto y con qué frecuencia | Diseño sección8; payload explícito por fase |
| Streams y solapamiento | Un stream explícito, sin solapamiento |
| API/librería y versión | numba-cuda/CUDA Python propuesto; lock compatible se verifica antes de implementar |
| Referencia de corrección y tolerancia | Plan de validación: siete puertas y umbrales previos |
| Riesgos de ocupación, ancho de banda o VRAM | Diseño secciones8/9: tope768MiB y condiciones de admisión |
| DEC / CAM / EXP / RUN | DEC-017/018/019; CAM-005; evidencia CPU EXP-002; ningún RUN GPU |

Si una librería decide la configuración de kernels, distinguir lo diseñado por
el equipo de lo gestionado por ella. Registrar la evidencia disponible y resolver
cómo se satisface REQ-05; no atribuir al equipo grids que no configura ni observa.

## Estado de cierre

Selección y propuesta de diseño están documentadas. Los criterios siguientes
incluyen validación posterior y no se consideran satisfechos por escribir el diseño:

- Selección justificada con análisis y mediciones CPU.
- Trabajo equivalente en CPU/GPU y preparación ubicada en CPU.
- Distribución hilos/bloques/grids explicada y vinculada a código.
- Presupuesto de memoria con entradas, pesos, activaciones, gradientes y optimizador.
- Medición de transferencias, coste de lanzamiento y total sincronizado.
- Verificación numérica, limitaciones y decisiones documentadas.
