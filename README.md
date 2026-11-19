# proyectoCUDA-paralela
Proyecto 3: Mejorando algoritmos con CUDA

La especificación normativa es [Proyecto 3 CUDA.pdf](docs/Proyecto%203%20CUDA.pdf).
La versión Markdown de la consigna es material explicativo secundario.

- [Trazabilidad y estado del proyecto](docs/trazabilidad/README.md).
- [Requisitos y evidencias](docs/trazabilidad/requisitos.md).
- [Informe final editable DOCX](entregables/Informe_Proyecto_3_CUDA.docx).
- [Notebook secuencial: ejecución completa y salidas](entregables/notebooks/ASL_Secuencial.ipynb).
- [Notebook CUDA: ejecución completa y salidas](entregables/notebooks/ASL_CUDA.ipynb).
- [Verificación de pasos 1–5 y reservas antes del paso 6](docs/trazabilidad/revision-pasos-1-5.md).
- [Análisis del algoritmo original](docs/trazabilidad/analisis-original.md).
- [Reproducibilidad y protocolo experimental](docs/trazabilidad/reproducibilidad.md).
- [Resultados y artefactos](resultados/README.md).
- [Paso 2: mediciones CPU, tablas y gráficas](resultados/secuencial/EXP-002/resultados.md).
- [Paso 3: identificación de oportunidades de paralelización](docs/trazabilidad/oportunidades-paralelizacion.md).
- [Paso 4: diseño completo del algoritmo CUDA](docs/trazabilidad/diseno-paralelo.md).
- [Paso 5: notebook CUDA Python con salidas reales](notebooks/Lab3_ASL_CUDA_con_salidas.ipynb).
- [Paso 6: análisis y conclusiones, tablas y gráficas](resultados/comparacion/paso-6-revision-02/analisis-y-conclusiones.md).
- [Notebook fuente congelado de las mediciones](notebooks/Lab3_ASL_CUDA.ipynb).
- [Resultados CPU/CUDA, CSV y gráficas](resultados/paralelo/EXP-003/resultados.md).
- [Informe PDF del paso 5](resultados/paralelo/EXP-003/informe-paso-5.pdf).

El laboratorio original está en `LAB3-DATASCIENCE/` y permanece intacto. Se
completaron cinco corridas CPU en PCUDA; se separan las fases comunes y las
ramas adaptativas. DIS-CUDA-001 especifica el diseño CNN-B, sus kernels,
memoria, equivalencia y medición. Se implementó y ejecutó CNN-B con CUDA Python
y cuBLAS: cinco corridas, media418.608s frente a805.650s CPU, aceleración1.925×,
incluyendo transferencias y validaciónCPU; accuracy test98.62%.
Los cambios del diseño se documentan en DEC-021/022/023. No se repitió la campaña
CPU; el original permanece intacto. La ampliación posterior incorpora también FC-A y FC-B; no se aceleró todo el notebook.

El paso 6 analiza el speedup y sus posibles causas con las cinco observaciones
por variante y los límites de la revisión previa. Se conserva en Markdown con
CSV y tres gráficas reproducibles. El informe final DOCX está disponible con
24páginas,16tablas y13imágenes, conservando el estilo editado por el usuario; la exportación PDF queda a su cargo.


[Ampliación final FC-A/FC-B: implementación, resultados y evidencias](docs/trazabilidad/ampliacion-fc.md). FC-A15.123s vs50.716sCPU(3.354×), FC-B21.062s vs102.441sCPU(4.864×), cinco observaciones por modelo. Notebook CUDA actualizado con salidas nativas de las tres redes; CPU/CNN-B previas no se volvieron a ejecutar.


[Presentación PowerPoint del proyecto](entregables/Presentacion_Proyecto_3_CUDA.pptx) · [Guion para exponer](entregables/presentacion/Guion_de_exposicion.md) · [Vista general](entregables/presentacion/Vista_general.png).


[Presentación corta:15diapositivas](entregables/Presentacion_Proyecto_3_CUDA_Corta.pptx) · [Guion corto](entregables/presentacion-corta/Guion_corto.md). La versión de31se conserva sin cambios.


[Guía de exposición del código CUDA](entregables/Guia_Codigo_CUDA.md):qué enseñar,qué decir yarchivo/función/línea exactos.
