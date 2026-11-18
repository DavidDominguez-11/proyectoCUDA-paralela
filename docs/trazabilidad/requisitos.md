# Matriz de requisitos y evidencias

Fuente normativa: [PDF oficial](../Proyecto%203%20CUDA.pdf).
Los identificadores son internos. El estado evalúa los artefactos existentes,
no la mera creación de una plantilla.

| ID | Página | Requisito | Documento / trabajo | Evidencia de cierre | Estado |
|---|---|---|---|---|---|
| REQ-01 | 1 | Tres integrantes y contribuciones en repositorio compartido | Bitácora y commits | Nombres, responsabilidades y contribuciones verificables de los tres | Pendiente completar equipo |
| REQ-02 | 1–2 | Descomponer solución original y registrar operaciones repetitivas, datos y repeticiones | [analisis-original.md](analisis-original.md) | Dos tablas solicitadas, referencias de celdas y [extracción estática del original](../../resultados/analisis-original/revision-estatica.md) | Completado: paso 1 por revisión estática |
| REQ-03 | 2 | Varias ejecuciones CPU equivalentes y tiempo promedio | [Informe CPU](../../resultados/secuencial/EXP-002/resultados.md) | Cinco RUN, tiempos crudos, tablas, cuatro gráficas y referencia común con carga verificada | Completado con alcance explícito DEC-015; total adaptativo separado |
| REQ-04 | 2 | Seleccionar y justificar oportunidades de paralelización | [Paso 3](oportunidades-paralelizacion.md); DEC-003/016 | Tabla de operaciones seleccionadas, simultaneidad, dependencias, exclusiones y prioridad sustentada en EXP-002 | Completado: selección justificada; sin ejecución CUDA |
| REQ-05 | 2 | Diseñar distribución entre hilos, bloques y grids | [Diseño inicial](diseno-paralelo.md); [kernels](diseno-kernels-cuda.md); [implementación revisada](implementacion-cuda.md); [contraste clase](contraste-semana-15.md) | Contratos,jerarquía propia explícita y recursos compilados por RUN;cuBLAS distinguido como backend | Completado para CNN-B con revisiones DEC-021/022/023 vinculadas a código real |
| REQ-06 | 2–3 | Implementar con CUDA Python y separar preparación CPU del entrenamiento GPU | [Paso5](implementacion-cuda.md); [notebook](../../notebooks/Lab3_ASL_CUDA.ipynb) | Kernels Python/cuBLAS,original preservado,cinco RUN reales y controles de pasos/estado | Completado para alcance CNN-B |
| REQ-07 | 2 | Misma carga y condiciones CPU/GPU; promedios con procesamiento y transferencias | [Informe CUDA](../../resultados/paralelo/EXP-003/resultados.md); EXP-003 | Cinco CPU históricas y cinco GPU;datos/estados/pasos verificados,H2D/D2H medidos;DEC-021/023 | Carga equivalente verificada; condiciones no idénticas (referencia histórica, RNG/shuffle y workers). Comparación válida dentro de los límites de la [revisión](revision-pasos-1-5.md); no certificar igualdad absoluta de condiciones |
| REQ-08 | 2–3 | Tablas, gráficas, speedup por fase y discusión | [Resultados](../../resultados/paralelo/EXP-003/resultados.md) | CSV de fuentes,estadísticas,cuatro gráficas,speedup por alcance | Disponible para CNN-B;revisión global paso6 pendiente |
| REQ-09 | 3 | Logs y capturas de carga por núcleos CPU y saturación de memoria/núcleos GPU | [Evidencias](../../resultados/evidencias/EXP-003/README.md) | CPU12lógicos y GPU/memoria/temperatura por segundo,capturas UTC/RUN correlacionadas | Parcial: uso/carga reales demostrados; saturación literal no acreditada, ocupación por SM no medida y VRAM observada 1213/4096 MiB. Véase [revisión](revision-pasos-1-5.md) |
| REQ-10 | 3 | Explicar reparto CPU/GPU, cambio de tiempo, límites y funciones/librerías GPU | [Implementación](implementacion-cuda.md);informe | Discusión con tiempos reales,cuBLAS/Numba,CPUpreparación/validación,límites de comparación | Disponible para alcance CNN-B |
| REQ-11 | 3 | Entregar código documentado e informe PDF | [Notebook](../../notebooks/Lab3_ASL_CUDA.ipynb); [PDF](../../resultados/paralelo/EXP-003/informe-paso-5.pdf) | Código fuente/copia ejecutada e informe de8páginas con tablas/gráficas/capturas | Archivos disponibles;paquete/revisión final de equipo pendientes |
| REQ-12 | 1 | Evidenciar datos, procedimientos y decisiones de análisis/diseño | Toda la trazabilidad | Enlaces completos entre DAT, PROC, DEC, CAM, EXP y RUN | Aplicado a pasos1–5;fallos,ensayos,revisiones y mediciones conservados |

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
