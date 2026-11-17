# Trazabilidad del Proyecto 3 CUDA

## Autoridad y alcance

La fuente de verdad es [Proyecto 3 CUDA.pdf](../Proyecto%203%20CUDA.pdf), de tres
páginas. Ante una diferencia, prevalece ese PDF sobre cualquier otro archivo.
Las convenciones de esta carpeta son acuerdos internos para satisfacerlo;
no deben presentarse como instrucciones adicionales del docente.

Estado: **preparación documental; sin nuevas mediciones CPU/GPU**.
Fecha de apertura: 2026-11-17, zona de trabajo America/Guatemala.
Las [instrucciones del repositorio](../../AGENTS.md) mantienen estas convenciones
para trabajos posteriores.

## Mapa de documentación

| Documento | Contenido |
|---|---|
| [Requisitos](requisitos.md) | Relación entre páginas del PDF, trabajo y pruebas de cumplimiento. |
| [Análisis original](analisis-original.md) | Flujo, funciones, operaciones repetitivas y antecedentes. |
| [Datos](datos.md) | Procedencia, versiones, manifiestos, particiones y transformaciones. |
| [Reproducibilidad](reproducibilidad.md) | Entorno, semillas, equivalencia, medición y recuperación. |
| [Procedimientos](procedimientos.md) | Pasos desde la preparación hasta la entrega. |
| [Decisiones](decisiones.md) | Alternativas, justificación, estado e impactos. |
| [Estrategia de paralelización](estrategia-paralelizacion.md) | Operaciones candidatas, memoria y diseño CUDA pendiente. |
| [Cambios](cambios.md) | Alcance y consecuencias de cada modificación. |
| [Bitácora](bitacora.md) | Actividades, responsables y evidencias por paso. |
| [Entrega](entrega.md) | Criterios para cerrar el proyecto y producir el informe PDF. |
| [Plantilla de ejecución](plantillas/ejecucion.md) | Registro detallado de una ejecución. |
| [Resultados](../../resultados/README.md) | Corridas CPU/GPU, tiempos, logs, imágenes y comparación. |

## Identificadores y relaciones

| Prefijo | Entidad | Ejemplo |
|---|---|---|
| REQ | Requisito extraído del PDF | REQ-05 |
| DAT | Datos o particiones versionadas | DAT-03 |
| PROC | Procedimiento | PROC-04 |
| DEC | Decisión | DEC-003 |
| CAM | Cambio | CAM-002 |
| BIT | Actividad | BIT-002 |
| EXP | Experimento con una configuración y una pregunta | EXP-001 |
| RUN | Una ejecución individual | RUN-20261117T180000Z-cpu-01 |

Una ejecución vincula EXP, requisitos, datos, procedimiento, decisiones, cambios,
commit y configuración. Cada resultado debe poder seguirse hasta los archivos
de entrada y su SHA-256. Cada afirmación del informe debe señalar los RUN y
artefactos que la sostienen. Un nombre de archivo sin versión no identifica datos.

## Reglas de registro

1. Registrar el paso y su finalidad antes o durante la actividad; adjuntar el
   resultado y los errores al terminar. Usar UTC ISO 8601 en artefactos y bitácoras.
2. Escribir responsables reales. `PENDIENTE` significa que no existe evidencia;
   no completar campos con supuestos, ceros ni resultados esperados.
3. Distinguir **observado en ejecución**, **salida histórica**, **inspección
   estática**, **cálculo derivado** e **hipótesis**.
4. Conservar las corridas fallidas, descartadas e interrumpidas y justificar su
   exclusión. No reemplazar una corrida para hacer que el resultado mejore.
5. Cada RUN vive en una carpeta nueva. Al cerrarla, no sobrescribir sus archivos;
   las correcciones se registran como un RUN nuevo que referencia al anterior.
6. Documentar cambios de datos, código, entorno y método. Si afectan la
   equivalencia, establecer otra configuración o experimento y repetir ambos lados.
7. No guardar contraseñas, tokens, archivos `.env` ni volcados de variables de entorno.
   Los artefactos deben contener solo información necesaria para reproducir.

## Situación inicial

Existe el laboratorio original con salidas históricas. No existen todavía una
versión CUDA propia, un entorno GPU validado, mediciones repetidas comparables,
capturas nuevas ni el informe final. Esta carpeta prepara su registro y no
certifica reproducibilidad hasta ejecutar y verificar los procedimientos.
