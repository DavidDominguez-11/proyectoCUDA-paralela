# proyectoCUDA-paralela
Proyecto 3: Mejorando algoritmos con CUDA

La especificación normativa es [Proyecto 3 CUDA.pdf](docs/Proyecto%203%20CUDA.pdf).
La versión Markdown de la consigna es material explicativo secundario.

- [Trazabilidad y estado del proyecto](docs/trazabilidad/README.md).
- [Requisitos y evidencias](docs/trazabilidad/requisitos.md).
- [Verificación de pasos 1–5 y reservas antes del paso 6](docs/trazabilidad/revision-pasos-1-5.md).
- [Análisis del algoritmo original](docs/trazabilidad/analisis-original.md).
- [Reproducibilidad y protocolo experimental](docs/trazabilidad/reproducibilidad.md).
- [Resultados y artefactos](resultados/README.md).
- [Paso 2: mediciones CPU, tablas y gráficas](resultados/secuencial/EXP-002/resultados.md).
- [Paso 3: identificación de oportunidades de paralelización](docs/trazabilidad/oportunidades-paralelizacion.md).
- [Paso 4: diseño completo del algoritmo CUDA](docs/trazabilidad/diseno-paralelo.md).
- [Paso 5: notebook CUDA Python con salidas reales](notebooks/Lab3_ASL_CUDA_con_salidas.ipynb).
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
CPU; el original permanece intacto. El alcance medido es CNN-B, no todo el notebook.
