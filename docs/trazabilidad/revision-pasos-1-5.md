# Revisión de los pasos 1–5 antes del análisis final

Fecha: 2026-11-18 UTC (17 de noviembre en Guatemala).
Responsable: Codex por solicitud del usuario. CAM-008; BIT-010.
Fuente normativa: [Proyecto 3 CUDA.pdf](../Proyecto%203%20CUDA.pdf), leído
directamente en sus tres páginas. Esta revisión no constituye el paso 6 ni una
calificación del docente. Es una auditoría interna del trabajo existente.

## Dictamen

**Es razonable avanzar al paso 6 con los resultados existentes de CNN-B y las
reservas descritas aquí.** La cadena de datos, las cinco observaciones por
variante, el cálculo de tiempos y speedup y el uso real de CUDA tienen evidencia
consistente. No se encontró en los controles realizados una inconsistencia que
obligue a repetir entrenamientos o descartar esas mediciones.

Esto no certifica cumplimiento absoluto de toda la entrega: la condición literal
de saturación GPU no está demostrada, las condiciones CPU/GPU no fueron idénticas
en todos sus aspectos, la igualdad numérica global estricta no fue aprobada y
faltan el paso 6, la revisión del informe final y las contribuciones del equipo.
El alcance válido es CNN-B base con el protocolo documentado de 15 épocas.

## Procedimiento y evidencia de esta revisión

1. Leer directamente el PDF y contrastarlo con la matriz, análisis original,
   oportunidades, diseño, decisiones, cambios, bitácora y resultados CPU/CUDA.
2. Inspeccionar los controladores CPU/CUDA, restauración tras warmup, bucles de
   entrenamiento, exportación, validación, kernels y adaptador cuBLAS. Esta
   inspección es estática; no se importó ni ejecutó el modelo.
3. Verificar los manifiestos cerrados de los diez RUN completos y las 184
   entradas del sello global EXP-003 contra los archivos actuales.
4. Volver a leer las 17,400 imágenes únicamente para SHA-256. Contrastar rutas,
   contenidos, cantidades y hashes de tensores registrados; no volver a preparar
   tensores ni entrenar. Revisar estados iniciales y hashes de modelos finales.
5. Recalcular media y desviación de entrenamiento desde los diez tiempos
   individuales originales. Contrastar tabla derivada y cocientes de medias.
6. Revisar los resultados archivados de pruebas numéricas y de integración,
   incluidos los fallos, y las dos capturas seleccionadas en su imagen real.
7. Contrastar las 15 líneas de épocas del notebook con stdout original de la
   última corrida; verificar ausencia de errores de celda e identidad del archivo.

Resultado automatizado: **80 controles aprobados, cero controles rechazados** en
[verificacion.json](../../resultados/auditoria/pasos-1-5-revision-02/verificacion.json).
Estos controles son de integridad/coherencia; no son 80 pruebas nuevas del
algoritmo CUDA ni certificación independiente de ausencia de errores.

Comando desde la raíz, usando el entorno existente:

```powershell
.\PCUDA\Scripts\python.exe -X utf8 scripts/auditar_pasos_1_5.py --destino resultados/auditoria/pasos-1-5-revision-02
```

El [script](../../scripts/auditar_pasos_1_5.py) rechaza un destino existente.
Para repetir solo esta auditoría se debe elegir otra carpeta. No importa
TensorFlow, Numba ni CUDA; no lanza benchmark, entrenamiento ni descargas.

Se conserva un [primer intento de auditoría](../../resultados/auditoria/pasos-1-5/verificacion.json)
que terminó con `KeyError('ruta_relativa')`: el verificador suponía el nombre de
columna CPU para el manifiesto GPU, cuyo encabezado es `ruta`. Se corrigió el
lector para admitir ambos esquemas y se usó otro destino. Fue un error del nuevo
verificador; no una falla experimental ni un cambio de resultados. La primera
lectura del PDF también requirió repetir la extracción con `python -X utf8` por
un carácter que la consola cp1252 no representa; el PDF no se modificó.

## Contraste con la guía

| Paso / requisito del PDF | Comprobación | Evaluación y límites |
|---|---|---|
| 1, pp. 1–2: partes, operaciones y repeticiones | [Análisis original](analisis-original.md), dos tablas y referencias de celdas; original con SHA conservado | Cubierto mediante inspección estática; las salidas históricas se distinguen de ejecuciones nuevas. |
| 2, p. 2: varias ejecuciones CPU, promedio y factores de rendimiento | [EXP-002](../../resultados/secuencial/EXP-002/resultados.md), cinco procesos completos, logs, tablas, gráficas y dispersión | Cubierto bajo protocolo controlado: 15 épocas fijas sin EarlyStopping; no es ejecución literal de cada celda original. Total adaptativo y suma común correctamente separados. |
| 3, p. 2: seleccionar y justificar oportunidades | [Paso 3](oportunidades-paralelizacion.md), prioridad CNN-B por 71.94% del tiempo de las 14 fases comunes | Cubierto. El PDF permite seleccionar operaciones; no exige trasladar todos los modelos. El porcentaje no corresponde al pipeline adaptativo completo. |
| 4, p. 2: distribuir entre threads, blocks y grids | [Diseño](diseno-paralelo.md), [catálogo](diseno-kernels-cuda.md), revisiones DEC-021/022/023 y lanzamientos por RUN | Cubierto para kernels propios. El diseño inicial se conserva como antecedente y se distingue de lo implementado. Los grids internos cuBLAS son del backend NVIDIA. |
| 5, p. 2: implementar CUDA Python y ejecutarlo varias veces | Numba-CUDA, SGEMM cuBLAS, cinco RUN nuevos con salida 0, 15 épocas y 1440 pasos cada uno | Cubierto localmente. TensorFlow se usa en CPU como referencia, validación y persistencia; no se presenta como ejecución GPU. |
| 5, p. 2: mismos datos y condiciones, procesamiento y transferencias | Manifiestos y X/y coincidentes; mismo modelo, pesos, precisión, batch, épocas y validaciones; pared sincronizada con H2D/D2H | Carga equivalente verificada; condiciones no idénticas en sentido estricto. CPU histórica, RNG/shuffle distintos y preparación 1→6 workers deben acompañar la comparación. |
| Entregable código, p. 3: original y optimizado, separación de fases | [Notebook con salidas](../../notebooks/Lab3_ASL_CUDA_con_salidas.ipynb), fuente congelada y módulos documentados | Disponible. CPU procesa imágenes/archivos; GPU ejecuta matemática de entrenamiento; validación/test permanecen CPU. |
| Evidencia, p. 3: CPU por núcleos y saturación GPU | Logs, CSV por segundo, dos capturas legibles con UTC/RUN y GPU reconocida | Uso y carga demostrados; **saturación literal de memoria/núcleos no acreditada**. No equivale a cumplimiento íntegro de esa formulación. |
| Informe y análisis, pp. 2–3 | PDF de paso 5 y fuentes CSV/Markdown existentes | Material disponible; paso 6, conclusiones y entrega final todavía pendientes. |
| Equipo de tres, p. 1 | Responsables de actividades y repositorio | No se certifican tres contribuciones verificables; pendiente del equipo. |

## Validez de datos y calidad

- Se conservan **17,400 archivos: 600 por cada una de 29 clases**, con particiones
  train 12,180, validación 2,610 y test 2,610. Las rutas y hashes son únicos en la
  selección; no hay archivos exactamente idénticos cruzando estas particiones.
  Los 17,400 hashes recalculados coinciden con el manifiesto.
- Los diez registros de tensores CPU/GPU coinciden en SHA-256 de imágenes y
  etiquetas por partición. Preparar con seis workers no cambió esos datos.
  En esta revisión se verificaron esos registros sellados; no se reconstruyeron
  los tensores completos en memoria.
- Los estados iniciales CPU conservan su identidad y el estado importado en
  GPU coincide con el preservado. Los cinco archivos finales `.keras` conservan
  sus hashes. La auditoría HDF5 existente registra contador Adam 1440 y contenido
  numérico canónico idéntico entre las cinco corridas CUDA; esta revisión verifica
  su identidad mediante esos hashes, sin cargar modelos para inferir otra vez.
- El resultado GPU de test **98.6206889%** es coherente con **2574 aciertos de
  2610 imágenes**. Es un resultado del conjunto de prueba interno conservado,
  no del conjunto oficial de 28 fotos, fotos del equipo ni vídeo en vivo.
- Cinco repeticiones con las mismas claves aleatorias miden repetibilidad de
  tiempos; no son cinco semillas independientes para estimar calidad promedio
  sobre entrenamientos diversos. La identidad de resultados GPU es coherente
  con ese protocolo. La CPU histórica varía entre 92.874% y 98.889%: no se puede
  afirmar que CUDA mejore accuracy ni que las trayectorias sean iguales.
- No hay evidencia de separación por persona o sesión. La ausencia de duplicados
  exactos no elimina semejanza visual entre fotos ni certifica generalización a
  personas nuevas. Es un límite de calidad del dataset/partición, sin invalidar
  por sí mismo la comparación de tiempos sobre los mismos archivos.
- El original selecciona modelos con métricas de test en otras ramas; por ello
  su mejor resultado global no es una estimación libre de sesgo de selección.
  EXP-003 fija CNN-B por prioridad computacional antes de medir, sin seleccionar
  una nueva arquitectura o semilla por sus resultados de test. No usar el máximo
  histórico de las otras ramas para atribuir ventajas de calidad al hardware.

## Corrección de la implementación: evidencia y límites

La inspección es consistente con Conv/ReLU, BatchNorm, pooling, Dropout, Dense,
softmax/entropía cruzada, retropropagación y Adam de CNN-B. Se conserva el último
lote de 20 imágenes, FP32 y la actualización de parámetros después de calcular
los gradientes. El código sincroniza el stream antes de cerrar intervalos de
entrenamiento y exportar estado para evaluación CPU.

Hay **51 pruebas de operadores y 78 comparaciones de adjuntos locales aprobadas**
archivadas, más una integración de diez pasos sintéticos con restauración,
contador y guardado/recarga. Las comparaciones locales usan activaciones y
gradientes de entrada idénticos para aislar cada operación. La integración final
corresponde a las fuentes finales; las pruebas locales anteriores precedieron
la corrección de introspección de recursos compilados documentada en el entorno.

También existen diagnósticos estrictos de red completa **fallidos**, que se
mantienen como tales. Se documentaron diferencias de selección de máximos cerca
de empates FP32. Esto es compatible con sensibilidad de pooling a redondeo y
las pruebas locales aportan soporte adicional; no demuestra que toda discrepancia
global posible tenga exclusivamente esa causa ni sustituye una prueba formal.
La conclusión admisible es implementación respaldada por las verificaciones
realizadas, no corrección matemática absoluta ni equivalencia bit a bit CPU/GPU.

## Tiempos y comparación admisible

| Magnitud | CPU | Variante CUDA | Interpretación |
|---|---:|---:|---|
| Corridas completas consideradas | 5 | 5 | Procesos independientes, misma carga CNN-B fija. |
| Entrenamiento total, media ± DE muestral | 805.650 ± 2.723 s | 418.608 ± 1.916 s | Incluye validación y, en variante CUDA, staging/máscaras y H2D/D2H. |
| Speedup de entrenamiento | — | 1.924594× | Cociente de medias; reducción de tiempo aproximada 48.04%. |
| Preparación de imágenes | 17.325 s | 8.034 s | Ambas CPU; 1 frente a 6 workers. No atribuir a CUDA. |
| Evaluación final | 2.322 s | 2.492 s | Ambas CPU; no hay aceleración observada de esta fase. |
| Suma de tres fases comparables | 825.297 s | 429.134 s | Speedup combinado 1.923167×; incorpora también cambio de workers CPU. |

Los promedios y DE de entrenamiento se recalcularon desde los CSV de cada RUN;
coinciden con los publicados. Los cuatro cocientes de fases se verificaron.
El CV GPU 0.458% y el semiancho ICt95% nominal 0.568% satisfacen el criterio
preestablecido de 5%; no corresponde ampliar a 7/9 solo para mejorar cifras.
Cinco muestras con baja dispersión respaldan la estabilidad local del tiempo,
sin eliminar sesgos sistemáticos ni certificar su distribución estadística.

Precauciones obligatorias para el paso 6:

1. La referencia CPU usa librerías con paralelismo interno (6 hilos intra-op,
   BLAS 6); «secuencial» describe la variante original CPU, **no un único núcleo**.
   La aceleración compara implementaciones y backends completos; no aísla un
   efecto causal puro del dispositivo ni justifica calcular eficiencia por
   división entre cantidad de núcleos CUDA.
2. CPU y GPU se midieron en horarios distintos. Comparten máquina y plan de
   energía registrados; no hay alternancia experimental ni igualdad térmica
   acreditada. RNG y shuffle conservan distribución/trabajo, pero no máscaras
   históricas idénticas. Presentar el speedup como **observado frente a referencia
   histórica compatible**, sin afirmar condiciones absolutamente idénticas.
3. La mejora de preparación corresponde al cambio CPU 1→6 workers. La suma de
   tres fases describe la variante completa optimizada, no el beneficio exclusivo
   de kernels CUDA. No mezclarla con los 1119.928 s de 14 fases ni los 1858.617 s
   del pipeline CPU adaptativo completo.
4. La pared de entrenamiento es la métrica principal. Los eventos incluyen
   intervalos del stream y posibles huecos de envío Host; 340.815 s no es tiempo
   activo puro de kernels. Validación, copias y administración están anidados:
   no se suman nuevamente al entrenamiento ni se atribuyen todos los segundos
   de esa pared al cómputo GPU.
5. Warmup y restauración quedan fuera en ambas variantes; importación, EDA y
   guardado final no se incluyen. Es rendimiento del alcance declarado, no
   latencia total desde abrir Python ni lectura de disco frío.

## Capturas y cumplimiento pendiente

Las dos imágenes seleccionadas se inspeccionaron nuevamente y son legibles.
La de CPU registra preparación con 225.2% en el árbol de procesos y actividad
en varios procesadores lógicos. La de GPU registra entrenamiento, actividad
agregada 88% y VRAM 1213/4096 MiB. Los CSV resumen muestran medias de actividad
de entrenamiento de aproximadamente 86–88% y máximos de 91–94% por RUN.

**Esto acredita uso real y alta actividad agregada, no saturación total.**
1213/4096 MiB equivale aproximadamente a 29.6% de capacidad ocupada. La actividad
agregada no mide ocupación por SM y `gpu_mem_util_pct` no representa porcentaje
de capacidad VRAM ocupada. La captura CPU es de preparación en EXP-003, no una
captura del entrenamiento CPU histórico EXP-002; este último tiene logs y CSV
por procesador conservados, sin convertirlos retroactivamente en una captura.

La formulación de saturación de memoria/núcleos de la página 3 queda parcialmente
acreditada. Debe exponerse así en el informe final; no aumentar artificialmente
la memoria ocupada ni presentar porcentajes como si fueran otra métrica. Esta
reserva no invalida los tiempos medidos. Una interpretación docente que exija
ocupación por SM o saturación literal requeriría evidencia adicional específica;
no puede certificarse con los archivos actuales.

## Correcciones documentales de esta revisión y siguientes pasos

- Se aclara en la matriz que REQ-07 acredita carga equivalente con diferencias
  de condiciones y que REQ-09 no tiene demostrada la saturación literal.
- Se actualiza DAT-05: la importación del estado inicial CNN-B sí se verificó
  en EXP-003; el texto previo describía solamente el estado al terminar el paso 2.
- Los informes, CSV, RUN y notebook sellados se conservan intactos. Esta revisión
  complementa su interpretación y no cambia resultados ni algoritmos.
- Se puede redactar el paso 6 con las reservas anteriores y preparar después el
  informe PDF final. La revisión de contribuciones del equipo y de entrega sigue
  pendiente. No hay necesidad demostrada de repetir las campañas CPU/GPU para
  sostener el resultado de rendimiento local y alcance aquí delimitados.

La reproducción limpia en otra máquina no ha sido probada. Existen versiones,
comandos, fuentes, manifiestos, modelos y copia local de los datos seleccionados;
eso aporta trazabilidad local, sin certificar disponibilidad futura de paquetes
nightly ni conservación externa de `data/`.
