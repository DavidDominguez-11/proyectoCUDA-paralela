# Matriz de requisitos y evidencias

Fuente normativa: [PDF oficial](../Proyecto%203%20CUDA.pdf).
Los identificadores son internos. El estado evalúa los artefactos existentes,
no la mera creación de una plantilla.

| ID | Página | Requisito | Documento / trabajo | Evidencia de cierre | Estado |
|---|---|---|---|---|---|
| REQ-01 | 1 | Tres integrantes y contribuciones en repositorio compartido | Bitácora y commits | Nombres, responsabilidades y contribuciones verificables de los tres | Pendiente completar equipo |
| REQ-02 | 1–2 | Descomponer solución original y registrar operaciones repetitivas, datos y repeticiones | [analisis-original.md](analisis-original.md) | Dos tablas solicitadas, referencias de celdas y [extracción estática del original](../../resultados/analisis-original/revision-estatica.md) | Completado: paso 1 por revisión estática; benchmark posterior pendiente |
| REQ-03 | 2 | Varias ejecuciones CPU equivalentes y tiempo promedio | reproducibilidad.md; resultados/secuencial | RUN, configuración, tiempos crudos y resumen | Pendiente ejecutar |
| REQ-04 | 2 | Seleccionar y justificar oportunidades de paralelización | estrategia-paralelizacion.md; decisiones.md | Selección sustentada en análisis y mediciones | Candidatos identificados |
| REQ-05 | 2 | Diseñar distribución entre hilos, bloques y grids | estrategia-paralelizacion.md | Diseño concreto ligado a operaciones y código | Pendiente diseñar |
| REQ-06 | 2–3 | Implementar con CUDA Python y separar preparación CPU del entrenamiento GPU | Código y cambios.md | Versiones original/optimizada documentadas y ejecución real | Pendiente implementar |
| REQ-07 | 2 | Misma carga y condiciones CPU/GPU; promedios con procesamiento y transferencias | reproducibilidad.md; configuraciones | Pares de RUN compatibles y H2D/cómputo/D2H registrados | Pendiente ejecutar |
| REQ-08 | 2–3 | Tablas, gráficas, speedup por fase y discusión | resultados/comparacion | Datos fuente, cálculo reproducible y figuras | Pendiente medir |
| REQ-09 | 3 | Logs y capturas de carga por núcleos CPU y memoria/núcleos GPU | RUN/logs y RUN/capturas | Evidencia durante cada fase y vinculada a sus intervalos | Pendiente capturar |
| REQ-10 | 3 | Explicar reparto CPU/GPU, cambio de tiempo, límites y funciones/librerías GPU | Entrega e informe | Respuestas técnicas apoyadas por resultados propios | Pendiente resultados |
| REQ-11 | 3 | Entregar código documentado e informe PDF | entrega.md | Archivos finales, referencias y comprobación del paquete | Pendiente entregar |
| REQ-12 | 1 | Evidenciar datos, procedimientos y decisiones de análisis/diseño | Toda la trazabilidad | Enlaces completos entre DAT, PROC, DEC, CAM, EXP y RUN | Estructura preparada |

## Rúbrica

| Criterio | Puntos | Requisitos relacionados |
|---|---:|---|
| Refactorización de código | 40 | REQ-02, REQ-04, REQ-05, REQ-06 |
| Evidencia de ejecución | 30 | REQ-03, REQ-07, REQ-09 |
| Reporte de análisis | 30 | REQ-08, REQ-10, REQ-11 |

El PDF no fija cantidad de repeticiones, tamaño de bloque, modelo concreto ni
umbral mínimo de aceleración. Esos parámetros deben decidirse, justificarse y
quedar congelados antes de las mediciones. La mención de CUDA Python y del diseño
de hilos/bloques/grids debe resolverse explícitamente; disponer de TensorFlow no
demuestra por sí solo que ambos puntos estén cubiertos.
