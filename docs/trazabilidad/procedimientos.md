# Procedimientos y registro de pasos

Cada aplicación de un procedimiento genera una entrada BIT y, cuando hay
ejecución experimental, una carpeta RUN usando [la plantilla](plantillas/ejecucion.md).
Registrar responsable, UTC, comando, entrada, salida, errores y decisión siguiente.

| ID | Procedimiento | Entradas | Pasos y salida verificable |
|---|---|---|---|
| PROC-01 | Revisar especificación | PDF y commit | Leer completo; verificar hash; actualizar REQ; registrar discrepancias sin sustituir PDF |
| PROC-02 | Inventariar y preservar original | Notebook, fotos y dependencias | Hashes; análisis estático; conservar archivo original y salidas históricas identificadas |
| PROC-03 | Preparar entorno | Stack seleccionado y hardware | Instalar versiones exactas; contexto; dispositivos efectivos; prueba mínima documentada; lock por plataforma |
| PROC-04 | Adquirir y preparar datos | DAT-01/02 | Registrar fuente/licencia; hashes; validar imágenes; congelar selección/splits/mapeo; preparar en CPU |
| PROC-05 | Diseñar comparación | Análisis, DAT, modelo | Decidir fases, carga fija, semillas, repeticiones, warmup, tolerancias y orden; cerrar EXP/configuración |
| PROC-06 | Medir referencia CPU | EXP completo y entorno validado | Forzar/confirmar CPU; restaurar estado por repetición; capturar fases, logs y recursos; guardar RUN/tiempos |
| PROC-07 | Diseñar e implementar CUDA | Cuellos medidos y REQ-04/05/06 | DEC con alternativas; memoria, hilos/bloques/grids; cambio de código vinculado; validar corrección |
| PROC-08 | Medir versión GPU | Mismo EXP y DAT | Confirmar CUDA real; warmup/restauración; sincronizar; medir H2D/cómputo/D2H/total; capturar recursos |
| PROC-09 | Analizar resultados | RUN cerrados y comparables | Validar exclusiones/equivalencia; estadísticas; speedup; gráficas reproducibles; causas y limitaciones |
| PROC-10 | Reproducir y entregar | Código, entorno, manifiestos y resultados | Reproducción desde cero; auditoría de enlaces/hashes; informe PDF y paquete final |
| PROC-11 | Identificar oportunidades CUDA, paso 3 | PDF, análisis original y mediciones CPU cerradas | Revisar independencia/reducciones/dependencias; priorizar fases; justificar selección y exclusiones; registrar DEC/CAM/BIT sin diseñar ni implementar |
| PROC-12 | Diseñar algoritmo CUDA, paso4, sin implementar | PDF, paso3, código original, CPU medida e inventario | Fijar alcance/semántica; formas, kernels y jerarquía; memoria/copias; aleatoriedad, pruebas y medición; registrar límites de entorno y DEC/CAM/BIT |

## Registro de comandos

Para cada comando conservar el texto exacto y el directorio de trabajo; anotar
intérprete, fecha, salida y código de salida. No basta una captura parcial del
terminal. Distinguir instalación/configuración de ejecución del algoritmo.
Los comandos pendientes se identifican como tales, no como comandos ya probados.

## Evidencia de recursos

- Capturar el CPU durante preparación con vista por procesador lógico y fecha
  correlacionable; guardar workers configurados y muestreo si se dispone de él.
- Capturar GPU durante entrenamiento con uso de cómputo y VRAM; guardar el nombre
  y versión del monitor, intervalo de muestreo y comando usado.
- Conservar capturas originales y logs de monitorización, vinculados a RUN y fase.
  Una captura en reposo o un listado de dispositivos no prueba uso durante cálculo.
- Registrar cuánto afecta la monitorización al experimento. Mantener la misma
  política entre repeticiones comparadas o separar corridas de evidencia de las
  de timing y declarar esa separación.
- No editar porcentajes, timestamps ni resultados en las capturas. Si se añade
  una anotación para el informe, conservar el original y explicar la derivación.

## Fallos y cambios de procedimiento

Guardar errores completos y causa conocida o hipótesis. Registrar corrección
en CAM/DEC y crear otro RUN. Las observaciones fallidas conservan su estado.
Si cambia el método de medición, no mezclar series incompatibles; volver a medir
CPU/GPU o presentar experimentos separados con su explicación.


## Ampliación posterior vigente: FC-A y FC-B

El alcance inicial CNN-B se amplió con dos entrenamientos completos adicionales en CUDA. Se conservaron las mediciones y evidencias anteriores. [Diseño,procedimiento,decisiones,resultados,notebooks e informe actualizados](ampliacion-fc.md). Los conteos/hashes de la entrega anterior descritos arriba son antecedentes; la entrega actual CUDA tiene92celdas de código y30PNG, y el DOCX16tablas/13imágenes con el diseño editado por el usuario.
