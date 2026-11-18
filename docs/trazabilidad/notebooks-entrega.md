# Notebooks separados con ejecución directa y salidas nativas

Actividad completada:DEC-026;CAM-011;BIT-013. Responsable:Codex por solicitud del
usuario. El usuario pidió la opción más correcta para revisión detallada del
docente:se decidió ejecutar una corrida adicional CPU y unaCUDA directamente
en celdas,por separado,y conservar sus salidas nativas. No se repite la campaña
de cinco observaciones por variante ni se cambian los promedios del informe.

## Preparación y separación

- CPU:programa secuencial completo medido,con cinco entrenamientos de15épocas,
  PCA,dos bosques,selección,diagnóstico e inferencia en fotos propias.
- CUDA:preparaciónCPU,matemática del entrenamientoCNN-B enGPU,validación/testCPU.
  Incluye todos los módulosPythonCUDA como celdas ejecutables `%%modulo`,sin
  copiar ni ejecutar el pipelineCPUcompleto dentro de este notebook.
- Las operaciones proceden del AST de los controladores congelados. Se mantienen
  los cálculos,datos,semillas,pasos y threadpool6 entre celdas. Las funciones de
  datos/arquitectura originales se muestran y los kernels se ejecutan directamente.
- Datos,tablas,historiales completos,métricas,gráficas,logs y recursos quedan
  incrustados. Los cinco registros históricos adicionales están identificados
  en un apartado separado de la corrida nueva.
- Un notebook ejecutado puede abrirse y mostrar todas sus salidas sin volver a
  entrenar. Ejecutarlo otra vez requiere otro EXP/RUN;los destinos cerrados se
  rechazan para no sobrescribir evidencia.

## Intentos y configuraciones

1. EXP-004CPU:falló en la celda de funciones originales por faltar SEED/IMG_SIZE
   como constantes globales de argumentos predeterminados. No inició entrenamiento.
   Se conservaron notebook fallido,traceback,logs y monitor. EXP-005CUDA quedó
   preparado,pero no se ejecutó tras ese falloCPU.
2. Se añadieron las constantes antes de definir funciones y se prepararon nuevas
   identidades:EXP-006CPU yEXP-007CUDA. No se sobrescribieron EXP-004/005.
   EXP-006se detuvo antes del entrenamiento por faltar datetime/timezone en los
   imports globales de los helpers;EXP-007no llegó a ejecutarse. Se preserva el
   intento y se incorporaron todos los imports originales de módulo automáticamente.
   La ejecución vigente usaEXP-008CPU yEXP-009CUDA,con nuevas carpetas de registro.
3. Se observó notebooks/Lab3_ASL_CUDA.ipynb con SHA2740b45016095833a556c2cc93ac380382d5768e3aeebe583a34b6af57ce663b,
   diferente del histórico53050f...;esa copia de presentación no se modifica en
   esta actividad. Los nuevos experimentos dependen de sus plantillas propias y
   fuentes matemáticas congeladas,en vez de la identidad volátil de una vista
   histórica con salidas. El original de laboratorio y los módulos permanecen.

## Alcance de los tiempos nuevos

El tiempo por fase de entrenamiento/evaluación conserva su frontera numérica
original;las gráficas/resúmenes se producen entre fases. **El total continuo del
notebook adicional también incluye esa presentación entre celdas**,por lo que
no debe sustituir el total de los controladores históricos que excluían figuras.
La descripción genérica «gráficas fuera de intervalos medidos» de la configuración
se interpreta solo para fit/evaluate individuales,no para pipeline_total.
Esta precisión se documenta sin reescribir configuraciones ya congeladas.

La corrida adicional no es una nueva estimación n=5ni se añade a la media del
informe. Su propósito es evidenciar ejecución real de código por celda,con todos
sus resultados visibles. Las limitaciones históricas de comparabilidad,corrección
global y ocupaciónSM/saturaciónVRAM siguen vigentes.

## Comandos de la revisión vigente

Desde la raíz,con PCUDA existente:

```powershell
.\PCUDA\Scripts\python.exe -X utf8 scripts/preparar_notebooks_entrega.py --revision notebooks-entrega-v03 --cpu-exp EXP-008 --cuda-exp EXP-009
.\PCUDA\Scripts\python.exe -u -X utf8 scripts/ejecutar_notebooks_entrega.py --revision notebooks-entrega-v03
```

El controlador se lanzó oculto con Start-Process,redirección stdout/stderr y PID
persistido para sobrevivir a una desconexión del agente. Ejecuta primeroCPU y
despuésCUDA. Guarda el notebook tras cada celda y al fallar;no ejecuta un script
de entrenamiento opaco en un subproceso:las operaciones están en sus celdas.

La monitorización por segundo y una captura genuina de su ventana documentan
los recursos. La ventana permanece oculta hasta una fase de entrenamiento apta
y se muestra brevemente para la captura;no se captura el escritorio ajeno.

## Cierre real de las ejecuciones

| Variante | Experimento / RUN | Inicio–fin del notebook UTC | Código / épocas nuevas / PNG |
|---|---|---|---|
| CPU | EXP-008 / RUN-20261118T041129Z-cpu-notebook-01 | 04:11:34–04:45:55,18nov2026 | 0 /75 /10 |
| CUDA | EXP-009 / RUN-20261118T041129Z-cuda-notebook-01 | 04:45:57–04:53:50,18nov2026 | 0 /15 /4 |

Son corridas sucesivas,no simultáneas. El identificadorRUN se creó al preparar
las configuraciones;el inicio efectivo es el de invocacion/notebook-ejecucion.json.
CPU37celdas de código yCUDA31,todas con contador nativo consecutivo y sin errores.
Se verificó que fuentes de celdas corresponden a las plantillas congeladas.
CPUguardó cinco modelos con1440actualizacionesAdam cada uno;CUDAguardóCNN-B con1440.
Los hashes deX/y coinciden con la referencia. Se verificaron capturas embebidas
sin alterar y los diez archivos stdout/stderr históricos completos por notebook.

La muestra CPU sostenida registra507.3%del proceso yGPU0%;laGPU91%de actividad
y1213/4096MiB. Hay dos capturasCPU(entrada y carga sostenida) y unaCUDA. Son
capturas nuevas de una ventana real,correlacionadas con recursos.csv;no saturaciónSM.
Las14imágenes de los dos notebooks se inspeccionaron en contactos visuales.

Como observación adicional,n=1:fitCNN-Bnuevo818.1088875sCPU y418.9302988sCUDA.
TestCPU0.982375503;testCUDA0.986206889. No se incorporan a medias/DE/speedup del
informe;la nueva ejecución presenta resultados reales propios,con su alcance.

Se preservaron los notebooks nativos antes de añadir solo dos celdasMarkdown
de alcance/diagnósticos nativos completos del kernel. No se altera ningún código,
contador ni salida ejecutada. Finalizar y verificar desde raíz:

```powershell
.\PCUDA\Scripts\python.exe -X utf8 scripts/finalizar_notebooks_entrega.py --revision notebooks-entrega-v03
.\PCUDA\Scripts\python.exe -X utf8 scripts/verificar_notebooks_entrega.py --revision notebooks-entrega-v03
```

- [CPU entregado](../../entregables/notebooks/ASL_Secuencial.ipynb).
- [CUDA entregado](../../entregables/notebooks/ASL_CUDA.ipynb).
- [Validación final](../../resultados/preparacion/notebooks-entrega-v03/validacion-final.json).
- [Cierre documental con identidad anterior/final](../../resultados/preparacion/notebooks-entrega-v03/cierre-documental.json).
- [Original CPU nativo](../../resultados/preparacion/notebooks-entrega-v03/notebooks-nativos/ASL_Secuencial.ipynb).
- [Original CUDA nativo](../../resultados/preparacion/notebooks-entrega-v03/notebooks-nativos/ASL_CUDA.ipynb).

Los184artefactos históricosEXP-003 y el DOCX final conservaron sus hashes. No se
reentrenaron las campañas originales ni se exportó un PDF nuevo.


## Ampliación posterior vigente: FC-A y FC-B

El alcance inicial CNN-B se amplió con dos entrenamientos completos adicionales en CUDA. Se conservaron las mediciones y evidencias anteriores. [Diseño,procedimiento,decisiones,resultados,notebooks e informe actualizados](ampliacion-fc.md). Los conteos/hashes de la entrega anterior descritos arriba son antecedentes; la entrega actual CUDA tiene92celdas de código y30PNG, y el DOCX16tablas/13imágenes con el diseño editado por el usuario.
