# Reproducibilidad y protocolo experimental

Estado: protocolo aplicado a cinco corridas CPU EXP-002 y cinco CUDA EXP-003.
Fuente: REQ-03, REQ-07, REQ-08 y REQ-09 del [PDF](../Proyecto%203%20CUDA.pdf).
El entorno GPU local pasó admisión y ejecución;no se certifica reconstrucción
limpia en otra máquina. Las decisiones DEC-011/012/013 concretan
la carga y el procedimiento inicial. DEC-014/015 registran la variación del
aprendizaje y la estratificación necesaria al cambiar la arquitectura seleccionada.
La referencia de 14 fases comunes está medida; el total adaptativo no es una
carga fija. [Comandos de reproducción y límites](reproducir-exp-002.md).

El [diseño del paso4](diseno-paralelo.md) y su
[plan de validación/medición](diseno-validacion-cuda.md) especifican la comparación
propuesta inicial CNN-B con orden/máscaras compartidos. DEC-021/022/023 la revisaron
por instrucción posterior:se reutilizó CPU histórica compatible,sin otra campaña,
y se midió CUDA en EXP-003. [Protocolo aplicado](implementacion-cuda.md) y
[reproducción CUDA](reproducir-cuda.md). EXP-002 permanece intacto.

## Entorno y código

Por cada RUN, conservar:

- Commit completo, estado de Git, cambios locales y archivos fuente exactos usados.
  Preferir un árbol limpio al medir. Si está sucio, el commit solo no reproduce
  la corrida: adjuntar parche y archivos nuevos o un archivo del código completo.
- Python ejecutable/versión, sistema operativo, arquitectura, zona horaria,
  comando exacto, directorio de trabajo, inicio/fin UTC y código de salida.
- Lista exacta de paquetes y archivo de dependencias fijadas por plataforma.
  Conservar también procedencia y SHA-256 de wheels/lock cuando sea necesario.
  `requirements.txt` actual sin versiones no basta; un listado instalado tampoco
  garantiza por sí solo que la instalación pueda reconstruirse.
- CPU (modelo, núcleos físicos/lógicos, RAM), GPU (modelo, VRAM), controlador,
  versiones CUDA/cuDNN efectivas, framework y dispositivos detectados.
- Parámetros de hilos CPU, afinidad si se usa, workers, límites de memoria,
  modo de energía, procesos competidores, temperatura y throttling si se observan.
- Variables relevantes permitidas: CUDA_VISIBLE_DEVICES, OMP_NUM_THREADS,
  MKL_NUM_THREADS, TF_NUM_INTRAOP_THREADS, TF_NUM_INTEROP_THREADS,
  TF_DETERMINISTIC_OPS y CUBLAS_WORKSPACE_CONFIG, cuando apliquen.

La herramienta [capturar_contexto.py](../../scripts/capturar_contexto.py) prepara
un contexto estático, listado de paquetes y copia de fuentes. Se ejecuta con el
Python del entorno que se desea registrar y **no** identifica ni prueba la GPU.
Sus campos pendientes deben completarse con evidencia real de las ejecuciones.

Ejemplo desde la raíz, una vez elegido el intérprete:

```powershell
python scripts/capturar_contexto.py --destino resultados/secuencial/RUN-AAAAMMDDTHHMMSSZ-cpu-01/contexto
```

El destino debe ser nuevo. Las capturas del entorno se realizan antes de medir;
no se incluye su coste en tiempos CPU/GPU. No capturar mientras se modifica código.

## Congelación de un experimento

Copiar [la configuración de referencia](../../configuraciones/experimento.plantilla.json)
a `configuraciones/EXP-001.json`, completar todos los `null` y registrar DEC.
Es una plantilla documental, no un archivo que el notebook actual consuma.
Antes de medir, la implementación deberá consumir o verificar esos valores.

| Debe coincidir CPU/GPU | Registro requerido |
|---|---|
| Entradas y particiones | SHA-256 de manifiesto, tensores y label_mapping |
| Modelo y pesos iniciales | Arquitectura, checkpoint inicial y estado del optimizador |
| Trabajo matemático | Épocas/pasos fijos, batch, último batch, validación y aumentos |
| Precisión y optimización | dtype, mixed precision, TF32, XLA y determinismo |
| Aleatoriedad | Semillas Python/NumPy/framework, shuffle, augmentation y workers |
| Alcance de medición | Mismos límites de fases e inclusiones/exclusiones |
| Condiciones externas | Máquina, energía, carga, versiones y política de caché |

Para medir velocidad, usar una carga fija. Desactivar EarlyStopping en ese
experimento o establecer otro protocolo que iguale explícitamente el trabajo;
no comparar entrenamientos de distinta longitud. Los experimentos de calidad
con parada temprana son separados de la comparación de carga fija.

Declarar si la referencia CPU permite varios hilos en librerías matemáticas.
La palabra «secuencial» del enunciado no demuestra ejecución en un solo núcleo.
Si se incluye una referencia de un hilo, identificarla por separado. No reducir
artificialmente CPU sin explicarlo ni mezclar referencias al resumir.

## Procedimiento de repeticiones

1. Definir y registrar cantidad de warmups, repeticiones medidas, orden CPU/GPU,
   semillas por repetición y tolerancias **antes de ver resultados**.
2. Ejecutar CPU y GPU con la misma configuración común. Registrar todo ajuste
   específico de dispositivo. Separar compilación/inicialización del régimen
   estable y conservar también su coste para una lectura de ejecución en frío.
3. Reiniciar pesos y optimizador por repetición. El warmup puede modificarlos:
   restaurar el estado inicial después del calentamiento. Registrar política de
   procesos nuevos, caché de disco y caché de compilación.
4. Sincronizar la GPU antes/después de intervalos medidos. Un lanzamiento asíncrono
   no mide la finalización del cálculo. Elegir y documentar la API según el stack.
5. Registrar los valores individuales sin redondear prematuramente. Capturar fallos,
   salida completa y código de retorno. Justificar cualquier exclusión.
6. Reportar número de observaciones, media, mediana, desviación estándar muestral
   y mínimo/máximo. Para n<2, la desviación muestral es no aplicable, nunca cero
   inventado. Si se usan intervalos de confianza, describir su método.

No hay número de repeticiones fijado en el PDF; decidirlo en DEC-005. El protocolo
debe equilibrar coste y variabilidad observada, documentando cualquier ajuste.

## Fases y límites temporales

| Fase CSV | Qué mide | Qué debe quedar explícito |
|---|---|---|
| adquisicion | Descarga/extracción | Reportar aparte; no mezclar caché contra descarga |
| preparacion_cpu | Lectura, decodificación, resize, normalización, etiquetas | Mismos archivos, workers y caché |
| inicializacion | Creación de modelo, contexto, compilación y asignaciones | Coste en frío frente a warmup excluido |
| h2d | Transferencias Host→Device | Tensores/pesos incluidos y frecuencia |
| entrenamiento | Forward, loss, backward, actualización | Pasos, épocas y si incluye validación |
| validacion | Evaluación de validación | Separada si no está incluida en entrenamiento |
| d2h | Transferencias Device→Host | Salidas/métricas/pesos realmente retornados |
| evaluacion_final | Inferencia y métricas finales | Misma evaluación y precisión |
| pipeline_total | Tiempo de pared del tramo completo definido | Límites idénticos CPU/GPU; transferencia incluida |

No confundir la suma de eventos de dispositivos con tiempo de pared. Si hay
solapamiento de transferencias y cómputo, sumar ambos sobrecuenta: publicar las
mediciones parciales y el total sincronizado, describiendo el solapamiento.
Si el framework oculta transferencias, obtener evidencia mediante profiler o
instrumentación adecuada; no registrar `0` ni atribuir todo el tiempo de fit al kernel.
Para CPU, h2d/d2h son `no_aplica`; para GPU no medido es `pendiente`.

Registrar también FPS/imágenes por segundo si resulta útil, con numerador y
alcance exactos. Esta métrica complementa los tiempos solicitados.

## Comparación y corrección

Speedup por fase comparable = media(T_CPU) / media(T_GPU).
Calcular también el speedup del pipeline completo, incluyendo transferencias.
No promediar cocientes individuales y llamarlos cociente de medias.
No calcular speedup aislado CPU/GPU para h2d/d2h, pues no hay fase CPU equivalente.
Un resultado menor que 1 es válido y requiere explicación.

Comparar shapes, etiquetas, orden de clases, valores finitos, pérdida, métricas
y salidas usando tolerancias definidas antes del experimento. El entrenamiento
puede tener diferencias numéricas acumuladas; documentar tolerancia y método,
sin exigir identidad de pesos bit a bit cuando el backend no la garantiza.
Registrar por separado reproducibilidad de datos, de configuración, numérica y
estadística. Conservar lo observado si alguna no puede garantizarse.

## Recuperación desde cero

En un entorno limpio: recuperar commit/fuentes → instalar entorno fijado →
recuperar DAT y verificar hashes → reconstruir/verificar particiones → restaurar
pesos iniciales → ejecutar comando registrado → validar corrección → medir →
regenerar resúmenes/gráficas con su script y configuración.

La reproducción solo se declara verificada cuando se realiza esta secuencia y
se adjunta un RUN de reproducción, sus diferencias y el criterio de aceptación.


## Ampliación posterior vigente: FC-A y FC-B

El alcance inicial CNN-B se amplió con dos entrenamientos completos adicionales en CUDA. Se conservaron las mediciones y evidencias anteriores. [Diseño,procedimiento,decisiones,resultados,notebooks e informe actualizados](ampliacion-fc.md). Los conteos/hashes de la entrega anterior descritos arriba son antecedentes; la entrega actual CUDA tiene92celdas de código y30PNG, y el DOCX16tablas/13imágenes con el diseño editado por el usuario.
