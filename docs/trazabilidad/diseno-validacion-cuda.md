# DIS-CUDA-001. Equivalencia, validación y medición

Parte del [diseño del paso 4](diseno-paralelo.md), revisión1. **Plan completo,
sin pruebas GPU ni experimentos nuevos ejecutados**. Amplía los contratos de
[reproducibilidad](reproducibilidad.md); no sustituye la evidencia CPU EXP-002.

## 1. Experimento futuro y cambios que deben declararse

Crear una identidad EXP nueva, todavía **no asignada ni congelada**: no reservar
un número como si existiera un experimento ejecutado. Pregunta prevista: tiempo
CPU frente a CUDA Python del entrenamiento CNN-B base con carga, orden y máscaras
equivalentes. Congelar código real, lock, hardware efectivo, todos los parámetros
de DIS-CUDA-001 y tolerancias **antes** de observar tiempos comparativos.

La versión CPU pareada tendrá la misma arquitectura, tensores y estado inicial,
con adaptador de entrenamiento explícito. Los cambios respecto a EXP-002 son:

| Aspecto | EXP-002 histórico | Comparación futura diseñada |
|---|---|---|
| Alcance | Varias redes, PCA/RF y ramas adaptativas | Una CNN-B base, sin augmentation |
| Preparación | Bucle de un worker | Seis workers, mismo resultado exacto; igual en ambos dispositivos |
| Shuffle | Gestionado por framework | Permutaciones CPU explícitas y compartidas |
| Dropout | RNG del framework | Máscaras CPU explícitas, idénticas para CPU/GPU |
| Validación | Incluida en fit | CPU al terminar cada época, misma cadencia/cantidad; copia del modelo incluida |
| Actualización | Adam de Keras | Misma fórmula, escalares por paso compartidos, actualización explícita auditada |
| Inicialización | Estados preservados | Se importa el mismo checkpoint CNN-B inicial de EXP-002 |

Se mantienen 15 épocas fijas, batch128, último lote20, Adam y todas las capas.
El adaptador CPU no elimina Dropout/BN ni altera el modelo para facilitar CUDA.
La nueva secuencia aleatoria mantiene la distribución prescrita pero **no reproduce
la trayectoria histórica**. Los 805.650 s históricos justifican prioridad; no son
el numerador directo de la comparación nueva. Tampoco se comparan 1119.928 s de
14 fases contra el tiempo de una sola red.

CPU de referencia: seis hilos intra-op, un inter-op, BLAS6, sin restricción de
afinidad adicional, mismo plan de energía y política de caché. GPU de referencia:
el coordinador CPU conserva esos límites para preparación/validación. Fijar
visibilidad TensorFlow a CPU **mediante su API antes de inicializar dispositivos**;
no establecer CUDA_VISIBLE_DEVICES=-1 en el proceso GPU porque ocultaría también
el dispositivo a Numba. Elegir UUID de la GTX1650 para CUDA y registrar su índice
efectivo. En el proceso exclusivamente CPU sí se puede ocultar GPU globalmente.
Verificar dispositivos y logs, no inferir la ubicación solo por el nombre PCUDA.

## 2. Aleatoriedad compartida y restauración

Semilla raíz43 para CNN-B. Parejas de repetición r=0..4; las dos variantes de
cada pareja utilizan idénticos datos, pesos iniciales, orden y máscaras.
Se utiliza NumPy del entorno fijado, generador **PCG64** inicializado mediante
SeedSequence con una tupla de enteros no negativos. La [documentación de NumPy](https://numpy.org/doc/stable/reference/random/bit_generators/pcg64.html)
define el generador; [SeedSequence](https://numpy.org/doc/stable/reference/random/parallel.html)
permite separar secuencias. Este contrato concreto es una decisión del proyecto:

- Permutación de cada época e: SeedSequence([43,r,e,0]); PCG64; permutar los
  índices0..12179. Guardar las15 permutaciones int64 y sus hashes antes de medir.
  No variar orden según duración, disponibilidad de worker, pérdida o accuracy.
- Máscara para capa d=1..4, época e y paso j: SeedSequence([43,r,e,j,d,1]);
  PCG64; generar E uniformes con Generator.random(E,dtype=float32) en orden C,
  mask=uint8(u≥float32(p)). Se conserva el generador/distribución exactos del lock.
- E=b×32×32×32, b×16×16×64, b×8×8×128 y b×256 respectivamente.
  Usar b real, incluso20. No generar una máscara b128 y seleccionar otro prefijo
  de manera implícita. Los factores FP32 1/(1-p) se comparten entre variantes.
- Se generan cuatro máscaras por paso en CPU y se conservan hasta backward.
  En GPU se transfieren y aplican; en CPU el adaptador las aplica directamente.
  No se ejecuta además el Dropout aleatorio estándar. Validación no consume RNG.
- Hash de contenido de cada máscara y de cada orden se verifica en las pruebas
  de equivalencia fuera del cronometraje. Se guardan claves, hashes y versiones;
  no se acumulan todas las máscaras en RAM ni se simula su generación como coste0.
- En timing se regeneran conforme al mismo contrato. Generación, asignación
  temporal Host y staging se incluyen en ambas variantes; hashes/serialización
  se hacen solo en la auditoría separada y con el mismo calendario CPU/GPU.

Después de warmup restaurar pesos, gamma/beta, BN móvil, m/v, contador0, métricas,
flags y claves del generador. Comprobar huella canónica antes/después de restaurar;
la huella se define por (capa ordinal,rol,shape,dtype,bytes C little-endian), no por
bytes de un ZIP con metadatos. No basta recrear el objeto modelo con la misma seed.

## 3. Puertas de aceptación antes de medir

| Puerta | Evidencia requerida | Acción si falla |
|---|---|---|
| V00 integridad | PDF/notebook originales, manifiestos, tensores, splits, labels y estado inicial con SHA-256 correcto | Detener antes de usar datos |
| V01 entorno | Versiones resueltas sin modificar dependencias de PCUDA; dispositivo real UUID/capacidad y compilación sm_75; CUDA simulator desactivado | Registrar incompatibilidad; no sustituir CPU por GPU silenciosamente |
| V02 recursos | Límites de block/grid/shared válidos, VRAM libre≥1536 MiB tras contexto, RAM Host libre≥4096 MiB, arenas device≤768 MiB y reserva exitosa | Conservar error; no cambiar batch/dtype |
| V03 preparación | Igualdad byte a byte de salida CPU1worker/6workers y tensor congelado, incluyendo etiquetas/orden | Investigar transformación o escritura; no medir comparación |
| V04 operadores | Referencia FP64 analítica de casos pequeños y backend CPU FP32; todos los casos de tabla posterior pasan | Corregir y repetir solo el conjunto afectado y sus dependencias |
| V05 paso completo | Forward, pérdida, todos los gradientes, Adam y BN correctos con mismo estado/lote/máscaras, b128 y b20 | No entrenar una serie comparativa |
| V06 varios pasos | Diez pasos consecutivos con un lote20 incluido, mismos estados/claves, sin acumulaciones involuntarias; contador correcto | Diagnosticar deriva; no ampliar tolerancia retrospectivamente |
| V07 época y calidad | Una pareja de entrenamiento completo15épocas auditada por repetición prevista; finitud, contadores y criterio de calidad | Conservar diagnóstico; resultados no se presentan como soluciones equivalentes |

V07 puede aprovechar la ejecución completa destinada a auditoría y no obliga a
duplicar cada benchmark cuando la misma observación contiene toda su evidencia.
Las pruebas costosas de hashes intermedios/profiling no se mezclan con el timing
principal; sus RUN se identifican como auditoría. El diseño no registra puertas
como aprobadas: aún no hay implementación.

### Tolerancias predefinidas

Para un elemento finito: `abs(gpu-ref) ≤ atol + rtol×abs(ref)`. Además publicar
máximo absoluto, RMSE y error relativo de norma L2, con denominador
max(norm(ref),1e-12). Todo tensor debe cumplir la cota elemento a elemento y la
cota L2; no se eligen solo promedios favorables. No aplicar una cota relativa
ingenua a gradientes casi0 sin término absoluto.

| Objeto comparado con referencia CPU FP32 | atol | rtol | Error relativo L2 máximo |
|---|---:|---:|---:|
| Forward lineal, activaciones, logits y probabilidades | 1e-5 | 1e-4 | 1e-4 |
| BN: media, varianza y salida | 5e-5 | 5e-4 | 5e-4 |
| Gradientes de una operación o un paso | 1e-4 | 1e-3 | 1e-3 |
| Adam con **gradiente idéntico** de entrada: theta/m/v | 1e-7 | 1e-5 | 1e-5 |
| Pérdida de un paso y estado después de diez pasos | 5e-4 | 2e-3 | 2e-3 |

Para índices, permutaciones, masks, labels, argmax sin casi empates, número de
pasos y contador Adam se exige igualdad exacta. Empates exactos de pooling/argmax
deben seguir la convención fijada. Si logits de clases distintas difieren menos
de la cota numérica, registrar margen y discrepancia de clase sin ocultarla;
las pruebas sintéticas de desempate exacto sí deben coincidir.

Gradient checking: casos pequeños FP64 deterministas, diferencias centrales
h=1e-5 y error relativo ≤2e-3 en elementos seleccionados; evitar quiebres de ReLU,
empates de máximos y máscaras variables. Comparar también gradientes analíticos
completos. Las diferencias finitas son prueba adicional y no sustituyen comparar
los tensores reales con la referencia CPU.

Calidad después de15épocas, mismo checkpoint/orden/máscaras por pareja: diferencia
absoluta de val_accuracy y test_accuracy≤0.02 y diferencia de pérdida final
≤0.05+0.10×abs(pérdida_CPU). Evaluar también conteo de predicciones discrepantes,
matriz de confusión y curvas completas. Son umbrales internos previos, no exigidos
por el PDF ni una prueba matemática de igualdad. Si fallan, no afirmar calidad
equivalente aunque coincidan los tiempos. La referencia histórica ya mostró
variabilidad de aprendizaje; no se presupone coincidencia bit a bit.

Ningún umbral se afloja después de ver resultados sin DEC/CAM y un experimento
nuevo. No declarar un fallo de tolerancia como simple ruido sin aislar su causa.

### Casos obligatorios por familia

| Familia | Casos que deben quedar cubiertos |
|---|---|
| Índices/tiles | Dimensiones1,15,16,17,29; canales3/32/64/128; b1/20/128; slabs con cola; valores de borde y entradas inválidas |
| Conv fwd/dX/dW/db | Filtro identidad/ceros/impulso; padding en cuatro bordes; Ci≠Co; misma imagen/etiqueta en distinto lote; dX gather vs referencia; particiones de reducción2047/2048/2049 |
| Dense | I/O no múltiplos de16, salida29, gradientes de bias, vistas Flatten en orden C y transpuestas virtuales |
| ReLU | Negativos, positivos, cero, derivada0 en0; no sobrescribir activaciones BN aún necesarias |
| Pooling | Ventanas negativas, máximos en cada una de cuatro posiciones, empates, dX cero en no máximos y suma de gradientes correcta |
| BN | L1, lote20 y128, canal constante, valores de media grande/varianza pequeña, gamma0, estado móvil actualizado una vez; validación sin mutación |
| Softmax/loss | Logits ±100 y constantes, clase0/28, empate exacto, shift de logits invariante, suma probabilidades≈1, gradiente de cada muestra suma≈0; etiqueta inválida aborta |
| Dropout | Máscaras0/1, factores por capa, misma máscara forward/backward, último lote, identidad en validación, regeneración/hash exactos |
| Adam | g0, positivos/negativos, m/v iniciales no cero, pasos1/2/1440, epsilon exacto, contador una vez por lote, BN móvil fuera del optimizador |
| Integración | Todo gradiente pertenece al mismo estado de pesos; no usar pesos actualizados para dX; arenas reutilizadas sin contaminación; flags sticky y fallos conservados |

La futura implementación usará herramientas de detección de acceso/carreras
CUDA disponibles en el entorno en ejecuciones de depuración. La disponibilidad
del detector y sus logs se registran; una revisión estática no certifica ausencia
de carreras. No medir rendimiento con instrumentación de depuración activa.

## 4. Temporización y comparación

### Fronteras y alcance

| Medida | Inicio → fin | Incluye |
|---|---|---|
| preparacion_cpu | Apertura del primer archivo → X/y completos | Lectura,6workers,RGB,resize,normalización,orden y cierres; sin hashes previos |
| inicializacion_frio | Imports/contexto → modelo/kernels listos | Compilación, asignaciones y warmup; se informa separado |
| entrenamiento_total | Estado restaurado, antes de H2D inicial → tras última validación y D2H final Adam | Staging CPU, máscaras, escalares Adam, H2D, kernels, flags, sincronización, D2H15modelos, carga espejo y validaciónCPU |
| validacion_cpu | Modelo espejo listo → métricas de2610val | ForwardCPU, BN inferencia, Dropout identidad; subintervalo de entrenamiento_total |
| evaluacion_final_cpu | Modelo final listo → métricas de2610test | batch32, sin cambio de pesos; fuera del entrenamiento_total |
| pipeline_total | Preparación medida → fin evaluación final | Intervalo de pared continuo, administración/monitorización incluidas; inicialización/warmup realizados antes |

Para permitir pipeline continuo sin insertar compilación en su interior,
inicializar/calentar con tensores de prueba fuera de ese intervalo y restaurar;
los tensores reales preparados medidos se usan después. Si la prueba previa de
datos los leyó antes, aplicar la misma política de caché en CPU y GPU y declarar
que no se mide disco frío. No sustraer el coste de monitorización.

La CPU pareada tiene las mismas fronteras de trabajo y restaura los mismos estados;
sus buffers ya están en Host y H2D/D2H se registran **no_aplica**, no0 medido. CPU
también carga el espejo de validación y conserva estados finales. Guardado de
checkpoints, gráficas y hashes se realiza tras finalizar intervalos principales.

### Relojes y descomposición

- Tiempo principal: perf_counter_ns de Host, sincronizar stream antes y después
  de cada tramo GPU que se presenta como tiempo completo. El retorno del lanzamiento
  no significa que el kernel haya terminado.
- Un stream explícito contiene H2D, forward, pérdida, backward, Adam, métricas,
  flags y D2H en ese orden; no usar el default stream de otra librería.
- Eventos CUDA alrededor de cada grupo de copias y grupo de cómputo; registrar
  el evento final, esperar su finalización y convertir ms a s. La
  [API Host oficial](https://nvidia.github.io/numba-cuda/reference/host.html)
  documenta eventos, sincronización y tiempos. No leer eventos incompletos.
- En el perfil principal agrupar forward/loss, backward y Adam/métricas por paso.
  Sus eventos miden **intervalos del stream**, incluidos posibles huecos de envío
  de trabajo desde Python; no llamarlos suma exclusiva de tiempo activo de kernels.
  Obtener tiempo activo por kernel solo en RUN de profiling separado.
- Cómputo, H2D y D2H no se solapan en este diseño. Aun así la suma de eventos
  no es igual al total Host: existen preparación de máscaras, lanzamiento,
  sincronización y validaciónCPU. Publicar ambos y el residuo de administración
  sin atribuirlo automáticamente a una operación ni forzar igualdad artificial.
- D2H del flag ocurre cada paso e incluye sincronización; no esconderlo dentro
  de tiempo de cómputo. Transferencias de modelo/estados son fases propias.
- No sumar validacion_cpu al entrenamiento_total por segunda vez. El tiempo
  exclusivo de entrenamiento sin validación es derivado y se etiqueta como tal;
  no es la métrica principal. Copias de pesos no se descuentan del beneficio GPU.

### Repeticiones propuestas

Cinco parejas CPU/GPU con procesos nuevos por RUN, r0..4, una carga fija. Orden
predefinido: CPU/GPU, GPU/CPU, CPU/GPU, GPU/CPU, CPU/GPU. No ejecutar pares
simultáneamente ni otras cargas experimentales. Conservar seeds, energía,
temperaturas y procesos competidores; no descartar una corrida solo por lenta.

Warmup fuera de tiempos: una pasada completa forward/backward/Adam por cada
b128/20 y todos los kernels/colas/slabs usados, más validación y evaluación CPU
con sus formas. Restaurar todo tras warmup. Registrar tiempos fríos y verificaciones
sin mezclarlos con las cinco mediciones estables.

Regla previa: calcular IC t95% de las medias del entrenamiento_total por dispositivo
al terminar5parejas; si algún semiancho relativo supera5%, ampliar ambos a7;
repetir criterio y, si corresponde, a9. Orden adicional r5GPU/CPU, r6CPU/GPU,
r7GPU/CPU,r8CPU/GPU. Al llegar9 informar dispersión aunque no se logre el5%.
No cambiar arquitectura según accuracy ni cerrar antes por un speedup favorable.
Un fallo detiene la serie para diagnóstico; se conserva el RUN y la exclusión
justificada, nunca se sobreescribe. No hay corridas de este plan todavía.

### Métricas y conclusiones permitidas

Publicar por fase y dispositivo todos los tiempos crudos, n, media, mediana,
DE muestral, rango, CV e IC nominal con sus supuestos. Speedup de entrenamiento
=media(CPU_entrenamiento_total)/media(GPU_entrenamiento_total). Pipeline se calcula
con sus propios totales. Un valor menor que1 es un resultado admisible.

No dividir tiempo de fitCPU completo entre un evento de convoluciónGPU; no calcular
speedup CPU para transferencias sin fase equivalente. Los microbenchmarks de
operadores, si se realizan, usan la misma forma/precisión y se publican aparte.
La preparación y evaluación CPU de ambas variantes también se muestran, aunque
no cambien de dispositivo. Reportar throughput con numerador exacto:182700
ejemplos train en15épocas, no sumar val/test al mismo numerador sin declararlo.

## 5. Errores, recursos y paquete de evidencia previsto

Cada RUN futuro debe registrar código/lock/config/manifiestos con SHA-256,
comando exacto, UTC, intérprete, OS, CPU, UUID GPU, driver/toolkit efectivo,
límites device y nombres especializados de kernels. Registrar forma, block,
grid físico/lógico, slab, shared compilado, registros/local por kernel y bytes
transferidos. Un listado propuesto no se presenta como configuración observada.

Monitor: psutil para árbol de procesos CPU/RAM por núcleo cada1s; nvidia-smi
cada1s para utilización GPU/memoria, VRAM, temperatura, potencia/relojes si están
disponibles. Un campo no soportado en WDDM queda no_disponible; no se inventa0.
Capturas durante preparaciónCPU y entrenamientoGPU deben contener timestamp
correlacionable con logs/fases. La utilización agregada no prueba ocupación por SM
ni saturación total: se limita la conclusión a lo que el monitor mide.

Ante error conservar stdout/stderr completos, paso/capa/slab en curso, tipo de
excepción CUDA, flag sticky, última medición completa, estados parciales y causa
conocida o hipótesis. No continuar con NaN, reiniciar driver, desactivar TDR,
cambiar dtype ni mover silenciosamente una capa a CPU. Las correcciones exigen
CAM y nueva identidad de ejecución; si cambia el trabajo, nuevo EXP pareado.

| Artefacto futuro | Contenido mínimo |
|---|---|
| Configuración congelada y contexto | DIS-CUDA-001, versiones resueltas, código exacto, seeds, workers, hardware, política de caché |
| Manifiestos de carga | Datos/splits/labels/estado inicial,15permutaciones por r, claves y hashes de máscaras de auditoría |
| Resultados de corrección | Casos/umbrales/errores de cada puerta, tensores diagnósticos pequeños y decisiones de exclusión |
| CSV de tiempos | Repetición,dispositivo,fase,época/paso,forma,ns/ms,anidamiento,bytes y estado |
| Registro de lanzamientos | K01..K19, especialización,slab,block,grid,shared y recursos compilados |
| Estado final | Pesos/BN/Adam/contador1440, métricas train/val/test y SHA-256; binarios grandes fuera de Git |
| Evidencia hardware | Logs CPU/GPU y capturas reales durante fases, con instante y RUN |
| Informe | Tablas/gráficas fuente, speedup válido, límites, cambios y PDF final cuando corresponda |

Estos nombres describen entregables previstos. No se crean CSV vacíos ni RUN
ficticios para dar por cumplida la ejecución. La verificación actual del proyecto
solo valida consistencia documental e integridad de originales.
