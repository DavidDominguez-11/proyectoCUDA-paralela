# Estrategia de paralelización

Estado: **paso 3 completado; diseño del paso 4 pendiente, sin implementar**.
Relaciones: REQ-04, REQ-05, REQ-06; DEC-003/004/006/007/008.
No se ha elegido librería ni se ha implementado un kernel.

La selección vigente y su evidencia se detallan en el
[paso 3](oportunidades-paralelizacion.md): prioridad a la matemática del
entrenamiento de CNN-B (71.94% de la referencia común CPU), con operaciones
análogas de otras redes como oportunidades secundarias. DEC-003/016, CAM-004.
Las secciones siguientes conservan el inventario conceptual y la plantilla
vacía para una fase posterior; no constituyen un diseño aprobado.

## Separación de responsabilidades

CPU: archivos, lectura/decodificación, limpieza, resize, normalización, etiquetas,
particiones y aumento de imágenes si se incluye como preparación. Documentar
cómo se distribuye esta fase entre núcleos y si realmente mejora frente al bucle.

GPU/CUDA: matemática del entrenamiento del modelo seleccionado. Preservar datos,
modelo, pesos iniciales, pérdida, optimizador y precisión respecto a la referencia.
La evaluación/retorno de resultados debe tener ubicación y transferencias explícitas.

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

## Ficha obligatoria del diseño final

Completar una ficha por operación implementada; no inventar un tamaño de bloque
antes de elegir algoritmo, forma de tensores y hardware.

| Campo | Valor |
|---|---|
| Operación / función CPU / función CUDA | PENDIENTE |
| Entradas/salidas, dimensiones, dtype y layout | PENDIENTE |
| Trabajo que resuelve un hilo | PENDIENTE |
| Dimensiones de bloque y número de hilos | PENDIENTE |
| Fórmula y dimensiones del grid | PENDIENTE |
| Índices, límites y último bloque incompleto | PENDIENTE |
| Reducciones, barreras, atómicos y prevención de carreras | PENDIENTE |
| Memoria global/compartida/registros y bytes estimados | PENDIENTE |
| Vida de buffers, asignaciones y reutilización | PENDIENTE |
| H2D/D2H: qué, cuánto y con qué frecuencia | PENDIENTE |
| Streams y solapamiento, si se usan | PENDIENTE |
| API/librería y versión; qué abstrae el backend | PENDIENTE |
| Referencia de corrección y tolerancia | PENDIENTE |
| Riesgos de ocupación, ancho de banda o VRAM | PENDIENTE |
| DEC / CAM / EXP / RUN que sustentan el diseño | PENDIENTE |

Si una librería decide la configuración de kernels, distinguir lo diseñado por
el equipo de lo gestionado por ella. Registrar la evidencia disponible y resolver
cómo se satisface REQ-05; no atribuir al equipo grids que no configura ni observa.

## Criterios para cerrar la estrategia

- Selección justificada con análisis y mediciones CPU.
- Trabajo equivalente en CPU/GPU y preparación ubicada en CPU.
- Distribución hilos/bloques/grids explicada y vinculada a código.
- Presupuesto de memoria con entradas, pesos, activaciones, gradientes y optimizador.
- Medición de transferencias, coste de lanzamiento y total sincronizado.
- Verificación numérica, limitaciones y decisiones documentadas.
