# Trazabilidad del Proyecto 3 CUDA

## Autoridad y alcance

La fuente de verdad es [Proyecto 3 CUDA.pdf](../Proyecto%203%20CUDA.pdf), de tres
páginas. Ante una diferencia, prevalece ese PDF sobre cualquier otro archivo.
Las convenciones de esta carpeta son acuerdos internos para satisfacerlo;
no deben presentarse como instrucciones adicionales del docente.

Estado: **pasos 1–5 documentados; cinco corridas CPU EXP-002 y cinco CUDA EXP-003;
CNN-B implementada/medida, informe PDF disponible; entrega global pendiente de revisión**.
Fecha de apertura: 2026-11-17, zona de trabajo America/Guatemala.
Las [instrucciones del repositorio](../../AGENTS.md) mantienen estas convenciones
para trabajos posteriores.

## Mapa de documentación

| Documento | Contenido |
|---|---|
| [Requisitos](requisitos.md) | Relación entre páginas del PDF, trabajo y pruebas de cumplimiento. |
| [Revisión antes del paso 6](revision-pasos-1-5.md) | Auditoría contra el PDF: 80 controles, validez de datos/tiempos y reservas de comparación, corrección y saturación GPU. |
| [Análisis original](analisis-original.md) | Flujo, funciones, operaciones repetitivas y antecedentes. |
| [Datos](datos.md) | Procedencia, versiones, manifiestos, particiones y transformaciones. |
| [Reproducibilidad](reproducibilidad.md) | Entorno, semillas, equivalencia, medición y recuperación. |
| [Reproducir EXP-002](reproducir-exp-002.md) | Comandos concretos, manifiestos congelados y recuperación sin descarga. |
| [Procedimientos](procedimientos.md) | Pasos desde la preparación hasta la entrega. |
| [Decisiones](decisiones.md) | Alternativas, justificación, estado e impactos. |
| [Oportunidades de paralelización](oportunidades-paralelizacion.md) | Paso 3: selección, justificación, dependencias y límites a partir del análisis y tiempos CPU. |
| [Diseño del algoritmo paralelo](diseno-paralelo.md) | Paso4: arquitectura, contratos, reparto, flujo y memoria. |
| [Catálogo de kernels](diseno-kernels-cuda.md) | Threads/blocks/grids, índices, reducciones y backward por operación. |
| [Validación y medición CUDA](diseno-validacion-cuda.md) | Aleatoriedad, pruebas, tolerancias, CPU pareada y cronometraje previstos. |
| [Contraste con Semana15](contraste-semana-15.md) | Correspondencia con clase, precisiones y refinamiento de coalescencia; fuente conservada. |
| [Inspección del diseño](../../resultados/diseno/paso-4-inspeccion.md) | Inventario GPU, fuentes locales, hashes y presupuestos derivados; sin ejecutar kernels. |
| [Implementación CUDA](implementacion-cuda.md) | Paso5: código, revisiones del diseño, ejecución real, tiempos y límites. |
| [Notebook con salidas reales](../../notebooks/Lab3_ASL_CUDA_con_salidas.ipynb) | Última corrida:15épocas,tablas,métricas,tiempos y gráficas incrustados,sin entrenar otra vez. |
| [Reproducir CUDA](reproducir-cuda.md) | Consultar sin repetir corridas o preparar otra identidad con datos/estado preservados. |
| [Estrategia de paralelización](estrategia-paralelizacion.md) | Índice de selección y fichas del diseño vigente. |
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

El [paso4](diseno-paralelo.md) desarrolla esa selección para CNN-B base, con
kernels Python explícitos propuestos mediante NVIDIA numba-cuda. Se consultó
inventario GTX1650 de4GiB y se diseñó el presupuesto; no se ejecutaron kernels.
DEC-021 retiró la campaña CPU pareada obligatoria por instrucción posterior del
usuario. Se reutilizan fases compatibles históricas; máscaras/orden internos
no son idénticos, aunque sí arquitectura,datos,pasos,precisión y distribución.

El [paso5](implementacion-cuda.md) ejecutó cinco corridas reales CNN-B con
Numba-CUDA/cuBLAS:418.608±1.916s frente a805.650±2.723s CPU, aceleración1.925×
(media±DE). Incluye H2D/D2H,CPUstaging y validaciónCPU; test98.62%. Hay
[tablas/gráficas](../../resultados/paralelo/EXP-003/resultados.md),
[capturas](../../resultados/evidencias/EXP-003/README.md) e
[informe PDF](../../resultados/paralelo/EXP-003/informe-paso-5.pdf).
La reconstrucción limpia en otra máquina, contribuciones del equipo y revisión
de entrega global no se han certificado. No se promete identidad numérica CPU/GPU.
