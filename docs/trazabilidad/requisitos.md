# Matriz de requisitos y evidencias

Fuente normativa: [PDF oficial](../Proyecto%203%20CUDA.pdf).
Los identificadores son internos. El estado evalúa los artefactos existentes,
no la mera creación de una plantilla.

| ID | Página | Requisito | Documento / trabajo | Evidencia de cierre | Estado |
|---|---|---|---|---|---|
| REQ-01 | 1 | Tres integrantes y contribuciones en repositorio compartido | Bitácora y commits | Usuario confirmó David Domínguez23712,Jonathan Diaz23873,Luis Padilla23663;portada DOCX | Equipo identificado;contribuciones individuales/commits pendientes de acreditar |
| REQ-02 | 1–2 | Descomponer solución original y registrar operaciones repetitivas, datos y repeticiones | [analisis-original.md](analisis-original.md) | Dos tablas solicitadas, referencias de celdas y [extracción estática del original](../../resultados/analisis-original/revision-estatica.md) | Completado: paso 1 por revisión estática |
| REQ-03 | 2 | Varias ejecuciones CPU equivalentes y tiempo promedio | [Informe CPU](../../resultados/secuencial/EXP-002/resultados.md) | Cinco RUN, tiempos crudos, tablas, cuatro gráficas y referencia común con carga verificada | Completado con alcance explícito DEC-015; total adaptativo separado |
| REQ-04 | 2 | Seleccionar y justificar oportunidades de paralelización | [Paso 3](oportunidades-paralelizacion.md); DEC-003/016 | Tabla de operaciones seleccionadas, simultaneidad, dependencias, exclusiones y prioridad sustentada en EXP-002 | Completado: selección justificada; sin ejecución CUDA |
| REQ-05 | 2 | Diseñar distribución entre hilos, bloques y grids | [Diseño inicial](diseno-paralelo.md); [kernels](diseno-kernels-cuda.md); [implementación revisada](implementacion-cuda.md); [ampliación FC](ampliacion-fc.md) | Contratos,jerarquía propia explícita y recursos compilados por RUN;cuBLAS distinguido como backend; perfil FC con BN momentum0.99 y arena ajustada | Completado para CNN-B,FC-A yFC-B; DEC-021/022/023/027 |
| REQ-06 | 2–3 | Implementar con CUDA Python y separar preparación CPU del entrenamiento GPU | [Paso5](implementacion-cuda.md);[CPU final](../../entregables/notebooks/ASL_Secuencial.ipynb);[CUDA final](../../entregables/notebooks/ASL_CUDA.ipynb) | Kernels Python/cuBLAS;salidas nativas CNN-B y150épocas FC;CPUanterior intacta;[validación](../../resultados/preparacion/ampliacion-fc/validacion-final.json) | Completado para CUDA CNN-B/FC-A/FC-B;37celdas CPU y92CUDA,0errores |
| REQ-07 | 2 | Misma carga y condiciones CPU/GPU; promedios con procesamiento y transferencias | [Informe CUDA](../../resultados/paralelo/EXP-003/resultados.md); EXP-003 | Cinco CPU históricas y cinco GPU;datos/estados/pasos verificados,H2D/D2H medidos;DEC-021/023 | Carga equivalente verificada; condiciones no idénticas (referencia histórica, RNG/shuffle y workers). Comparación válida dentro de los límites de la [revisión](revision-pasos-1-5.md); no certificar igualdad absoluta de condiciones |
| REQ-08 | 2–3 | Tablas, gráficas, speedup por fase y discusión | [Paso 6 inicial](../../resultados/comparacion/paso-6-revision-02/analisis-y-conclusiones.md);[ampliación](../../resultados/comparacion/ampliacion-fc/resultados.md) | Cinco observaciones por modelo/dispositivo;tablas,DE/IC,CSV,gráficas,speedup de entrenamiento/test ycalidad | Completado para tres redes e informe DOCX;PDF por usuario |
| REQ-09 | 3 | Logs y capturas de carga por núcleos CPU y saturación de memoria/núcleos GPU | [Evidencias](../../resultados/evidencias/EXP-003/README.md) | CPU12lógicos y GPU/memoria/temperatura por segundo,capturas UTC/RUN correlacionadas | Parcial: uso/carga reales demostrados; saturación literal no acreditada, ocupación por SM no medida y VRAM observada 1213/4096 MiB. Véase [revisión](revision-pasos-1-5.md) |
| REQ-10 | 3 | Explicar reparto CPU/GPU, cambio de tiempo, límites y funciones/librerías GPU | [Implementación](implementacion-cuda.md);[ampliación FC](ampliacion-fc.md);[informe final](../../entregables/Informe_Proyecto_3_CUDA.docx) | Discusión por factor,medición/derivación/hipótesis,cuBLAS/Numba,CPUpreparación/validación,calidad ylímites | Integrado al informe DOCX para CNN-B/FC-A/FC-B |
| REQ-11 | 3 | Entregar código documentado e informe PDF | [CPU](../../entregables/notebooks/ASL_Secuencial.ipynb);[CUDA](../../entregables/notebooks/ASL_CUDA.ipynb);[informe DOCX](../../entregables/Informe_Proyecto_3_CUDA.docx);[revisión](informe-final.md) | DOCX24páginas,seis pasos,16tablas,13imágenes,logs/capturas reales yAPA7;Word/estructura ypreservación del estilo verificados | DOCX ampliado listo;PDF por usuario pendiente;PDF previo corresponde solo al paso5 |
| REQ-12 | 1 | Evidenciar datos, procedimientos y decisiones de análisis/diseño | Toda la trazabilidad | Enlaces completos entre DAT, PROC, DEC, CAM, EXP y RUN;fuentes/derivados del paso6 con SHA-256 | Aplicado a pasos1–6;fallos,ensayos,revisiones,mediciones y análisis conservados |

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


## Actualización de evidencia: ampliación FC

REQ-04/05/06/07/08/10/12 se amplían con FC-A/FC-B: código CUDA adicional, perfil/hilos/bloques/grids documentados, cinco mediciones por red, comparación CPU histórica, calidad y límites explícitos. REQ-09 añade dos capturas vivasGPU, sin certificar saturación literal. REQ-11 mantiene dos notebooks separados y DOCX actualizado;PDF por usuario pendiente. No cambia el estado de contribuciones humanas. [Evidencia yvalidación](ampliacion-fc.md).
