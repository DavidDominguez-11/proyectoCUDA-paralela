# Paso 3. Identificación de oportunidades de paralelización

Estado: **completado mediante análisis estático y evidencia CPU existente**.
Fuente normativa: [PDF oficial](../Proyecto%203%20CUDA.pdf), página 2, apartado 3;
la separación preparación CPU/entrenamiento GPU se contrasta con la página 3.
Relaciones: REQ-04/12, PROC-11, DEC-003/016, CAM-004 y BIT-006.
Responsable: Codex, por solicitud del usuario; revisión humana pendiente.

Este documento identifica y justifica oportunidades. No define distribución entre
hilos, bloques o grids, tamaños, gestión de memoria, kernels, APIs ni algoritmo
paralelo. Esas decisiones corresponden al paso 4. No se modifica ni ejecuta el
programa, ni se presenta una aceleración GPU como resultado medido.

## Evidencias y método de análisis

1. Contrastar los apartados 3, 4 y 5 del PDF para delimitar esta fase.
2. Revisar las operaciones y repeticiones del [paso 1](analisis-original.md),
   verificables por celda en la [extracción estática](../../resultados/analisis-original/revision-estatica.md).
3. Priorizar con las cinco mediciones de [EXP-002](../../resultados/secuencial/EXP-002/resultados.md),
   sin ejecutar otro experimento ni modificar sus archivos sellados.
4. Examinar independencia de resultados, reducciones y dependencias de pesos;
   separar posibilidad matemática de conveniencia para esta solución.
5. Registrar selección, exclusiones y límites; reservar diseño e implementación.

La identidad del notebook original es SHA-256
`ce8824e860ae7e02684bf702fd6865b7d87cc9eb71917f56bfb0ee8d9ccb8e8a`;
la del PDF es
`2dd8efc1ffb36d1b0908fbb400808a713b99e1c1322f71843e3e22198f72eb6d`.
Ambos se conservan en el [inventario inicial](../../resultados/inventario-inicial.csv).

| Evidencia utilizada | Identidad SHA-256 / función |
|---|---|
| [resumen.csv](../../resultados/secuencial/EXP-002/resumen.csv) | `27873ed4ad1555b8fc0bd728c0ac7f46c0a60f112d817bbb9b0ed80f4f9dbc88`; medias por fase y tamaños muestrales |
| [resumen_referencia_comun.csv](../../resultados/secuencial/EXP-002/resumen_referencia_comun.csv) | `2847cfb45dc84c4ea46086070b72a7747f619eca51951f0cc54838e61b595610`; denominador común |
| [Verificación de equivalencia](../../resultados/secuencial/EXP-002/verificacion-equivalencia.json) | Condiciones compartidas y límites de repetibilidad; vinculada a los RUN del informe |
| [Manifiesto de imágenes](../../resultados/secuencial/EXP-002/manifest_fuente.csv) y [particiones](../../resultados/secuencial/EXP-002/manifest_splits.csv) | Identidades individuales y orden de los 17,400 archivos utilizados; no se vuelve a muestrear |

La inspección estática identifica el trabajo; los tiempos siguientes son mediciones
CPU previas; los porcentajes son cálculos derivados; el posible beneficio CUDA es
una hipótesis que deberá verificarse en fases posteriores.

## Prioridad sustentada en los tiempos CPU

Se usa la referencia común de **1119.928 s**, n=5, que suma 14 intervalos no
solapados (DEC-015). Es una suma de fases de pared, no una medición exclusiva de
operaciones matemáticas. Su coeficiente de variación es 0.243%. El total adaptativo
de 1858.617 s mezcla ramas distintas y no se utiliza como denominador de selección.

| Fase CPU | Media (s), n=5 | Porcentaje de referencia común | Implicación para la selección |
|---|---:|---:|---|
| Entrenamiento CNN-B | 805.650 | 71.94% | Prioridad principal: concentra la mayor parte del tiempo común. |
| Entrenamiento CNN-A | 124.839 | 11.15% | Oportunidad secundaria con las mismas familias convolucionales. |
| Entrenamiento FC-B | 102.441 | 9.15% | Oportunidad secundaria en operaciones densas. |
| Entrenamiento FC-A | 50.716 | 4.53% | Oportunidad secundaria en operaciones densas. |
| Preparación CPU | 17.325 | 1.55% | Mantener en CPU conforme al PDF; menor peso temporal observado. |
| PCA | 3.963 | 0.35% | Posible paralelismo matemático, baja prioridad en este alcance. |

Fórmula: porcentaje = 100 × media de la fase / 1119.9275623. Las filas son una
selección de fases, por lo que no suman 100%. Los cuatro entrenamientos neuronales
concentran juntos **96.76%**. Cada `fit` incluye validación anidada y administración
del framework: no puede atribuirse todo su tiempo a convoluciones. No existe un
perfil por operador que permita repartir esos segundos entre capas.

Los datos de cada red son imágenes RGB 64×64, `float32`, con 12,180 ejemplos train,
2,610 val y 2,610 test. EXP-002 fija batch 128 y 15 épocas: 96 pasos por época,
1440 actualizaciones por red; el último lote de entrenamiento tiene 20 imágenes.
El notebook original permite EarlyStopping; estos tiempos corresponden al protocolo
controlado del paso 2, no a una ejecución intacta de todas sus celdas.

## Operaciones seleccionadas y justificación

Se selecciona como objetivo principal **la matemática del entrenamiento de CNN-B**
(operación 13 del paso 1, celdas 33–34). Sus cinco convoluciones, capas densas,
normalización y operaciones asociadas constituyen trabajo repetido sobre muchos
valores. La selección es una propuesta para CUDA, no una decisión de backend.

| No. | Operación y origen | Datos que procesa | Trabajo que admite simultaneidad | Beneficio esperado y dependencia que debe respetarse | Selección |
|---|---|---|---|---|---|
| OP-P01 | Convoluciones forward; `build_cnn_b`, celda 33 | Lote, filtros y mapas de activación; cinco capas con 32/32/64/64/128 filtros | Salidas de distintas imágenes, posiciones espaciales y filtros dentro de una misma capa, con pesos fijos | Gran cantidad de productos y sumas repetidos; cada salida reduce aportes del kernel y canales. La capa siguiente necesita resultados de la anterior. | Principal |
| OP-P02 | Productos de capas Dense; celda 33 | Activaciones, matrices de pesos y sesgos; Dense 256 y salida de 29 clases | Resultados para distintas muestras y unidades de salida | Trabajo aritmético regular y repetido; cada salida requiere una suma sobre sus entradas. Aplica también a CNN-A y FC-A/B. | Principal |
| OP-P03 | Retropropagación de convoluciones y Dense; `fit`, celda 34 | Activaciones conservadas, errores, pesos y gradientes | Contribuciones a gradientes de distintos elementos, filtros y parámetros | Repite productos/sumas a gran escala en cada paso. Contribuciones a un mismo gradiente se agregan; el backward depende del forward y del gradiente posterior. | Principal |
| OP-P04 | ReLU y derivada; celda 33 y entrenamiento | Tensores de activaciones y gradientes | Aplicaciones elemento a elemento una vez disponibles sus entradas | Muchas evaluaciones simples independientes; evita que partes del mismo entrenamiento queden como trabajo escalar CPU. Beneficio aislado puede ser pequeño por movimiento de datos y lanzamiento. | Complementaria del objetivo principal |
| OP-P05 | MaxPooling y propagación del gradiente; celda 33 | Ventanas de mapas de activación e información de máximos; tres capas | Cálculo de distintas salidas de pooling | Muchas comparaciones repetidas. Cada máximo depende de su ventana; backward necesita respetar la selección del máximo y posibles aportes coincidentes. | Complementaria |
| OP-P06 | BatchNormalization forward/backward; celda 33 | Activaciones, parámetros entrenables y estadísticas; cuatro capas | Estadísticas de canales/unidades diferentes y transformaciones de elementos | Reducciones repetidas y transformación de muchos valores. Media/varianza requieren agregación; conservar actualización de estadísticas móviles y diferencia entre entrenamiento e inferencia. | Complementaria necesaria para CNN-B |
| OP-P07 | Softmax y pérdida sparse categorical crossentropy; celdas 33–34 | Logits de 29 clases, etiquetas y probabilidades de cada muestra | Trabajo entre muestras y términos de cada reducción | Cálculos repetidos en todos los pasos. Cada normalización y pérdida requiere agregaciones; preservar estabilidad numérica y reducción de pérdida. Las 29 clases por sí solas no aseguran gran beneficio. | Complementaria |
| OP-P08 | Actualización Adam; compilación y `fit`, celdas 33–34 | Pesos, gradientes y estados de primer/segundo momento | Componentes de parámetros una vez obtenidos los gradientes correspondientes | Repite operaciones sobre todos los parámetros en cada actualización. El estado temporal y el contador de Adam vinculan los pasos sucesivos. | Complementaria necesaria para entrenamiento completo |
| OP-P09 | Dropout en entrenamiento; celda 33 | Activaciones y máscaras aleatorias; cuatro capas | Aplicación de máscaras a elementos distintos | Transformación masiva simple. La generación aleatoria y su reproducibilidad requieren control posterior; en inferencia no se aplican máscaras. | Complementaria necesaria para conservar el modelo |

OP-P03, OP-P08 y las derivadas no aparecen como funciones manuales en el notebook:
son trabajo implícito de `fit`/Adam requerido por el entrenamiento. No se afirma
haber observado sus kernels ni sus tiempos individuales. Flatten reorganiza la
interpretación de los datos y no se prioriza como cálculo costoso independiente.

La primera convolución de CNN-B ilustra la escala sin diseñar su ejecución:
con un lote completo produce 128×64×64×32 = **16,777,216 valores**; cada uno reúne
3×3×3 = 27 productos, incluidos términos de padding según la definición matemática.
Son cantidades derivadas de la arquitectura, no instrucciones GPU ni operaciones
efectivamente medidas. No describen el último lote, las otras capas ni backward.

## Otras oportunidades y operaciones conservadas en CPU

| Operación original | ¿Existe simultaneidad potencial? | Decisión y fundamento |
|---|---|---|
| Entrenamiento CNN-A y FC-A/B, operaciones 12/15 | Sí; mismas familias de productos, activaciones y gradientes | Secundarias: su selección futura ampliaría el alcance. No se propone entrenar todos los modelos al mismo tiempo ni comparar una arquitectura con otra como speedup. |
| Validación, evaluación y predicción neuronal, operaciones 14/15/24 | Sí; forward con pesos fijos sobre muestras distintas | Oportunidad asociada al modelo seleccionado, sin backward/Adam ni Dropout activo. Su ubicación y medición quedan para diseño; no son el cuello principal observado. |
| CNN aumentada, operación 19 | Sí en su entrenamiento neuronal | Condicional: solo extender la matemática de la arquitectura que se fije después. No agrupar CNN-A+aug (n=1) y CNN-B+aug (n=4); son cargas diferentes. |
| Lectura, RGB, resize y normalización, operaciones 8/9/23 | Sí entre imágenes y, matemáticamente, entre muchos valores | Se conservan como preparación CPU por el PDF. Independencia no implica seleccionarlas para CUDA. |
| Rotación, zoom, traslación y contraste, operación 19 | Sí entre imágenes/píxeles con dependencias de interpolación | Preparación CPU según la separación exigida. El original las incluye dentro del modelo; cualquier cambio futuro de ubicación necesita trazabilidad y equivalencia, aún sin diseñarlo. |
| PCA, operación 16 | Sí en álgebra matricial y reducciones | Fuera del objetivo CUDA seleccionado; 0.35% de referencia común y rama diferente del entrenamiento CNN. No se descarta por imposibilidad matemática. |
| Random Forest, operación 17 | Sí entre árboles y algunas evaluaciones | Fuera del objetivo seleccionado; ya solicita paralelismo CPU con n_jobs=-1. Cambiar biblioteca/algoritmo exigiría verificar equivalencia y justificar un alcance distinto. |
| Argmax, accuracy y matriz de confusión, operaciones 14/24/26 | Sí entre muestras, con agregación de conteos | Sin aceleración CUDA independiente propuesta: pocos resultados frente al volumen interno de las redes. Conservar conteos y etiquetas; beneficio propio sin medir. |
| Archivos, muestreo, particiones, etiquetas y persistencia, operaciones 1–7/10/21/22 | Algunas tareas son independientes | Permanecen en CPU; trabajo de control, almacenamiento o preparación, sin justificación para priorizar CUDA en este proyecto. |
| Selección de modelo, tablas, gráficas y reflexión, operaciones 18/20/25/26 | Algunas presentaciones podrían generarse de forma independiente | CPU; dependen de métricas terminadas y no concentran la carga matemática seleccionada. |

## Dependencias que limitan la simultaneidad

- Dentro de una capa hay muchas salidas calculables con las mismas entradas;
  entre capas deben estar disponibles las activaciones correspondientes.
- Dentro de un lote hay contribuciones independientes, pero gradientes de pesos,
  pérdidas y estadísticas de BatchNormalization requieren agregación. Las muestras
  no son completamente independientes durante esa normalización de entrenamiento.
- Forward precede a pérdida y backward; Adam utiliza los gradientes para actualizar
  el estado. El siguiente paso usa los pesos actualizados. Épocas y actualizaciones
  sucesivas no se seleccionan como tareas independientes.
- Evaluar con pesos fijos permite paralelismo entre muestras. La selección de la
  CNN y su reentrenamiento aumentado dependen de resultados previos; no debe
  anticiparse una rama sin alterar el comportamiento original.

## Beneficio esperado, límites y cierre del paso

La oportunidad principal combina gran volumen aritmético, repetición a lo largo
de 1440 pasos y predominio temporal de CNN-B. Se espera que el procesamiento
masivo reduzca el tiempo de sus operaciones matemáticas; no se garantiza una
magnitud ni que toda operación pequeña mejore de manera aislada. La referencia
CPU ya usa librerías con varios hilos y no representa un único núcleo.

El beneficio real dependerá del dispositivo CUDA, compatibilidad del entorno,
volumen de datos, memoria disponible, reducciones y transferencias CPU↔GPU. El
TensorFlow utilizado para la referencia CPU no detectó GPU; eso no demuestra la
ausencia de una GPU física ni valida un entorno CUDA. Su verificación sigue abierta.
La repetibilidad numérica de CNN-B tampoco quedó demostrada en EXP-002, aun con
datos y estados iniciales coincidentes; no se atribuye una causa específica.

Seleccionar CNN-B no convierte automáticamente los 1119.928 s de referencia común
en su tiempo CPU: corresponden a alcances distintos. La fase `entrenamiento_CNN-B`
de 805.650 s incluye validación. Una comparación futura deberá mantener fases,
arquitectura, datos, pasos y condiciones equivalentes, y contabilizar transferencias.

REQ-04 queda cubierto por selección, justificación y análisis de dependencias.
REQ-05 permanece pendiente: no se ha diseñado el algoritmo paralelo. REQ-06/07/08
conservan sus pendientes de implementación, ejecución GPU y speedup. No se crea
EXP/RUN porque esta actividad es documental y utiliza evidencia ya existente.
