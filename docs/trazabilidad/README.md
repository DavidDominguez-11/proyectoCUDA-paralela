# Trazabilidad del Proyecto 3 CUDA

## Autoridad y alcance

La fuente de verdad es [Proyecto 3 CUDA.pdf](../Proyecto%203%20CUDA.pdf), de tres
páginas. Ante una diferencia, prevalece ese PDF sobre cualquier otro archivo.
Las convenciones de esta carpeta son acuerdos internos para satisfacerlo;
no deben presentarse como instrucciones adicionales del docente.

Estado: **pasos 1, 2 y 3 completados; cinco corridas CPU EXP-002; diseño e implementación CUDA pendientes**.
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
| [Reproducir EXP-002](reproducir-exp-002.md) | Comandos concretos, manifiestos congelados y recuperación sin descarga. |
| [Procedimientos](procedimientos.md) | Pasos desde la preparación hasta la entrega. |
| [Decisiones](decisiones.md) | Alternativas, justificación, estado e impactos. |
| [Oportunidades de paralelización](oportunidades-paralelizacion.md) | Paso 3: selección, justificación, dependencias y límites a partir del análisis y tiempos CPU. |
| [Estrategia de paralelización](estrategia-paralelizacion.md) | Selección del paso 3 y ficha pendiente del diseño CUDA del paso 4. |
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

## Situación actual

Existe el laboratorio original, su análisis y cinco ejecuciones CPU verificadas.
El [informe del paso 2](../../resultados/secuencial/EXP-002/resultados.md) contiene
tiempos, cuatro gráficas y análisis de variabilidad. La referencia común suma
14 fases equivalentes; el total adaptativo se presenta por separado porque una
corrida cambió la CNN seleccionada. La repetibilidad numérica del entrenamiento
no se logró en todas las redes y se documenta como límite observado.

El [paso 3](oportunidades-paralelizacion.md) selecciona como prioridad la matemática
del entrenamiento CNN-B, sustentada en su 71.94% de la referencia común CPU.
La preparación permanece en CPU. La selección no constituye diseño ni validación GPU.

No existe aún versión CUDA, validación GPU, speedup ni informe PDF final. La
reconstrucción completa del entorno en otra máquina no se ha certificado.
