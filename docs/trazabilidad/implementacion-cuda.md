# Paso 5. Implementación CUDA Python

Estado: **completado para CNN-B:cinco corridas GPU verificadas, informe y notebook ejecutado disponibles**.
Fuente absoluta: Proyecto 3 CUDA.pdf, páginas 2–3. REQ-06/07/09/12;
CAM-006; BIT-008; DEC-021. El notebook original se conserva intacto.

La [revisión posterior de pasos 1–5](revision-pasos-1-5.md) verifica integridad y
cálculos sin entrenar otra vez. Precisa que la comparación usa carga equivalente
con condiciones históricas distintas y que la saturación literal de la GPU no
está demostrada por las capturas disponibles.

## Revisión del diseño antes de implementar

Por indicación posterior del usuario, no se repite la campaña CPU completa.
Se reutilizará la fase CNN-B de las cinco corridas EXP-002 siempre que coincidan
datos, arquitectura, batch128,15 épocas, último lote20 y frontera temporal con
validación incluida. Las nuevas mediciones GPU tendrán otro EXP, sin alterar
configuración ni evidencias CPU. La comparación con una referencia histórica
no alternada debe explicitar ese límite de condiciones térmicas/temporales.

Se retira la obligación de nuevas máscaras exactamente iguales a CPU histórica.
Para timing basta mismo Dropout/distribución y volumen; para pruebas unitarias
se usan máscaras fijas pequeñas. No se elimina Dropout ni se fuerza determinismo
numérico completo como requisito para reutilizar la carga CPU. Los cambios de
diseño y entorno se registran antes de medir; no se comparan cargas distintas.

Se utilizará PCUDA y el dataset cacheado, sin reinstalar sus paquetes existentes.
La inspección encuentra toolkit CUDA13.1 local y GPU sm_75. El perfil12.x propuesto
se adapta al toolkit existente para evitar descargas innecesarias. Solo faltan
las dependencias CUDA Python/Numba; su resolución preliminar admite NumPy2.5.5.
Antes de entrenar se requiere prueba real de compilación/ejecución y corrección.

Los resultados de compatibilidad,ensayos,corrección y medición se distinguen
en las evidencias de cierre siguientes;los errores permanecen conservados.

## Resultado de cierre

Por solicitud posterior se añadió [notebook con salidas reales incrustadas](../../notebooks/Lab3_ASL_CUDA_con_salidas.ipynb),
con las15épocas de la última corrida al principio. La [validación de esa vista](../../resultados/notebook-salidas/validacion.json)
confirma44celdas sin errores,15líneas de época,tablas/gráficas y ninguna corrida
adicional. La fuente congelada y la serie sellada se conservan intactas;CAM-007.

- EXP-003:cinco RUN terminados con código0,23:16:28–23:54:32UTC del2026-11-17.
- CPU histórica:805.649726±2.722684s; CUDA:418.607683±1.916308s,media±DE,n=5.
- Aceleración entrenamiento completo1.924594×,sin excluir H2D/D2H,CPUstaging o validación.
- GPU CV0.4578%,ICt95%[416.228272,420.987094]s;semiancho relativo0.5684%,sin ampliar serie.
- Las cinco corridas obtienen accuracy test0.986206889;estados finales HDF5
  canónicos idénticos en este entorno,contadorAdam1440. El contenedor.keras puede
  diferir por metadatos;los hashes de archivo y de estado se distinguen.
- [Informe completo](../../resultados/paralelo/EXP-003/resultados.md),
  [PDF](../../resultados/paralelo/EXP-003/informe-paso-5.pdf),
  [notebook validado](../../resultados/paralelo/EXP-003/notebook-validado.ipynb).

El primer notebook ejecutó las mediciones y sus37celdas de código sin errores de
celda,pero IPython intentó escribir historial SQLite fuera del workspace y la
shell reportó avisos/error nativo. Se conserva completo. Se ajustó únicamente
el ejecutor de presentación a historial en memoria/IPYTHONDIR local y se guardó
otra vista,sin entrenar de nuevo. La [salida validada](../../resultados/paralelo/EXP-003/notebook-validado.salida.json)
registra código nativo0 y SHA idéntico de serie antes/después. El aviso de kernel
TCP local de ipykernel7.4 queda en stderr;no es un fallo de entrenamiento.

## Implementación y responsabilidades

| Archivo | Función |
|---|---|
| [Notebook copia](../../notebooks/Lab3_ASL_CUDA.ipynb) | Conserva código CPU con guardas y flujo CUDA reproducible; no repite CPU al abrirlo |
| [runtime.py](../../src/asl_cuda/runtime.py) | Descubrir toolkit/DLL, rechazar simulator y comprobar dispositivo |
| [kernels.py](../../src/asl_cuda/kernels.py) | Kernels Python propios:convolución de referencia,im2col/gather,pooling,BN,Dropout,softmax,Adam,control |
| [blas.py](../../src/asl_cuda/blas.py) | SGEMM FP32 cuBLAS sobre los mismos buffers/stream; adaptación row-major/column-major explícita |
| [model.py](../../src/asl_cuda/model.py) | Arquitectura original,checkpoint,memoria,forward/backward,restauración,transferencias y espejo CPU |
| [benchmark_cuda.py](../../scripts/benchmark_cuda.py) | Congelar EXP,CPUpreparación,procesos nuevos,medición,monitorización y sellado de RUN |
| [resumir_cuda.py](../../scripts/resumir_cuda.py) | Auditar datos,comparar fases equivalentes,generar CSV/gráficas/PDF desde observaciones reales |
| [ejecutar_notebook_cuda.py](../../scripts/ejecutar_notebook_cuda.py) | Ejecutar copia con kernelspec local PCUDA y conservar notebook con salidas |
| [capturar_monitor_cuda.py](../../scripts/capturar_monitor_cuda.py) | Monitor de muestras reales psutil/nvidia-smi y capturas de su ventana |

El original conserva SHA-256ce8824e860ae7e02684bf702fd6865b7d87cc9eb71917f56bfb0ee8d9ccb8e8a.
La copia tiene83celdas, incluye las68originales con código protegido por
EJECUTAR_ORIGINAL=False, retirando salidas para no confundirlas con datos nuevos.
Los kernels fuente se muestran también como apéndice legible del notebook;
la única implementación ejecutable está en los módulos importados.

Preparación CPU:6workers/12tareas pendientes, mismos RGB/resizeBICUBIC/float32/255,
coordinador escribe en orden congelado. Los hashes de X/y deben coincidir con
EXP-002 antes de entrenar. No se lee JPEG ni se redimensiona en GPU. im2col es
indexación interna de la convolución sobre tensores preparados, no preparación
de archivos. CPU valida2610imágenes por época y evalúa test en batch32.

## Diseño realizado y cambios registrados

DIS-CUDA-001 revisión1 queda como antecedente; DEC-021/022/023 describen revisión
de implementación:sin campaña CPU nueva,cuBLAS para productos grandes,workspace
reutilizable576MiB,tope1408MiB y admisión numérica por operadores/adjuntos iguales.
El ensayo tiled≈0.60s/lote y SGEMM≈0.26s/lote fue preliminar,n=1,solo selección
de backend. Los resultados publicados se basarán en cinco o más RUN completos.

Se conserva explícita la propiedad de jerarquías:im2col,col2im,activaciones,pooling,
Dropout,Adam y finitud:block256,gridmin(4096,ceil(E/256)),recorrido por stride;
BN/bias:block32×8,grid(ceil(L/2048),ceil(C/32));fusión:block256,gridC;
softmax:block32,gridb. Los blocks/cuadrículas SGEMM los configura NVIDIA cuBLAS;
no se afirman configuraciones internas no observadas. El modo pedantic mantiene
FP32; sin TF32,mixed precision,atómicos flotantes o entrenamiento CPU oculto.

Las transferencias tienen eventos y payload:imágenes,labels,máscaras y estados
iniciales H2D;control por paso,modelo/BN por época y Adam final D2H. Una copia
final extra del modelo por exportación conservadora se mide y declara, sin
descontarla. Los eventos representan intervalos del stream; el tiempo principal
es pared sincronizada e incluye staging,máscaras,control y validaciónCPU.

## Entorno y reproducción del trabajo

Se usa el [entorno registrado](../../resultados/preparacion/cuda/entorno.md).
Los paquetes existentes se conservaron. Dataset y checkpoint inicial se reutilizan
de caché/data, sin descargar otra vez. CUDA13.1 ya estaba instalado.

Comandos ejecutados desde la raíz para congelar y ejecutar:

```powershell
.\PCUDA\Scripts\python.exe -X utf8 scripts/preparar_notebook_cuda.py
.\PCUDA\Scripts\python.exe -X utf8 scripts/benchmark_cuda.py prepare --experimento EXP-003
.\PCUDA\Scripts\python.exe -u -X utf8 scripts/ejecutar_notebook_cuda.py --salida resultados/paralelo/EXP-003/notebook-ejecutado.ipynb
```

Durante admisión se actualizó la copia generada antes de congelar mediante
preparar_notebook_cuda.py --actualizar; este modo rechaza modificarla si ya
existe EXP-003. La configuración y fuentes utilizadas por los workers no se
editan durante las mediciones. Cada RUN guarda comando exacto,contexto/fuentes,
paquetes,datos,tensores,hardware,épocas,eventos,recursos y checkpoint con hash.

Pruebas ejecutadas (directorios nuevos para conservar fallos y revisiones):

```powershell
.\PCUDA\Scripts\python.exe -u -X utf8 scripts/probar_operadores_cuda.py --destino resultados/preparacion/cuda/operadores-002 --backend cublas
.\PCUDA\Scripts\python.exe -u -X utf8 scripts/probar_adjuntos_cuda.py --destino resultados/preparacion/cuda/adjuntos-001
.\PCUDA\Scripts\python.exe -u -X utf8 scripts/probar_integracion_cuda.py --destino resultados/preparacion/cuda/integracion-002
```

No volver a ejecutar esos destinos:son evidencias existentes inmutables. Para
repetir una prueba usar otro destino. Abrir el notebook no ejecuta automáticamente
la CPU original; ASL_CUDA_RUN=1 habilita GPU y reutiliza la serie si está completa.
Una serie incompleta no se sobreescribe. La referencia histórica se preserva.

## Límites de comparación

- CNN-B CPU histórica tiene cinco tiempos de fit comparables, no un total de
  tres horas para esta red. No se comparará GPU contra todo el programa adaptativo.
- Mismos datos,arquitectura,pasos,precisión y distribución Dropout; no se promete
  misma secuencia RNG interna ni identidad de trayectoria/accuracy CPU-GPU.
- CPU histórica y GPU no se alternaron temporalmente. Se verificó el mismo plan
  ASUS Recommended; monitorización registra carga/temperatura. Se conserva esa
  limitación de estado térmico/instante en lugar de repetir procesos innecesarios.
- Preparación1worker frente a6workers se compara como mejora CPU, no como CUDA.
  Validación CPU es parte de entrenamiento y no se suma dos veces. Suma de fases
  comparables no equivale al intervalo continuo de todo el notebook.
- Una curva/calidad buena no sustituye comprobar gradientes y pasos; una captura
  agregada GPU tampoco demuestra ocupación de cada SM ni saturación total de VRAM.


## Ampliación posterior vigente: FC-A y FC-B

El alcance inicial CNN-B se amplió con dos entrenamientos completos adicionales en CUDA. Se conservaron las mediciones y evidencias anteriores. [Diseño,procedimiento,decisiones,resultados,notebooks e informe actualizados](ampliacion-fc.md). Los conteos/hashes de la entrega anterior descritos arriba son antecedentes; la entrega actual CUDA tiene92celdas de código y30PNG, y el DOCX16tablas/13imágenes con el diseño editado por el usuario.
