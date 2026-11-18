# Registro de cambios

## CAM-001 — Preparación de trazabilidad y reproducibilidad

- Fecha: 2026-11-17.
- Origen: solicitud del usuario; DEC-001/002; REQ-12.
- Responsable de preparación: asistente Codex. Revisión y responsables humanos: pendientes.
- Antes: notebook del laboratorio, README mínimos y consigna; sin estructura
  común para decisiones, datos, experimentos, resultados y evidencias.
- Después: documentación Markdown, matriz REQ, análisis original, protocolos,
  plantillas, esquema de configuración, inventario SHA-256 y captura de contexto.
- Continuidad: AGENTS.md indica cómo mantener los registros en trabajos futuros;
  .gitignore permite versionar logs dentro de resultados/.
- Código del algoritmo/datos/modelos: no modificado. No se han ejecutado entrenamientos.
- Impacto en rendimiento/calidad del modelo: no medido; no aplica a este cambio documental.
- Verificación: enlaces locales, JSON/CSV, identidad del original y prueba de
  captura de contexto en directorio temporal; detalle en BIT-003.
- Commit de incorporación: pendiente de realizar por el equipo.

## CAM-002 — Completar paso 1: análisis del original

- Fecha: 2026-11-17. Responsable: asistente Codex; revisión humana pendiente.
- Origen: solicitud de ejecutar exclusivamente el paso 1; REQ-02/12; DEC-010.
- Antes: inventario inicial de operaciones con cuatro columnas, sin el formato
  exacto solicitado ni conteo detallado del resto de bucles y rutas de inferencia.
- Después: explicación de cinco partes principales; tabla numerada de 26
  operaciones y tabla de 25 operaciones repetitivas con datos y frecuencias;
  fórmulas, alcance de recarga y épocas históricas identificados.
- Evidencia añadida: extracción Markdown de las 33 celdas de código y salidas
  seleccionadas; script reproducible `scripts/inspeccionar_original.py` que lee
  JSON y rechaza sobrescribir su archivo de destino.
- Documentos relacionados: análisis, requisitos, decisiones, bitácora, índice
  documental y catálogo de resultados. Original, datos y modelo sin cambios.
- Verificación: extracción estática, comprobación del inventario SHA-256,
  revisión de cálculos derivados y comprobaciones documentales; BIT-004.
- Impacto: completa el análisis del paso 1; no mide tiempos ni resuelve la
  selección de operaciones CUDA del paso 3. No crea corridas experimentales.
- Commit de incorporación: pendiente de realizar por el equipo.

## CAM-003 — Instrumentación y medición CPU del paso 2

- Fecha: 2026-11-17. Responsable: Codex; solicitud del usuario para ejecutar CPU.
- Estado: completado con cinco corridas EXP-002; EXP-001 conserva el intento
  interrumpido por el monitor inicial. Resultados en resultados/secuencial/EXP-002.
- REQ-03/07/12; PROC-03/04/05/06; DEC-011/012/013; BIT-005.
- Se añade benchmark_cpu.py: extrae por AST las funciones originales, congela
  selección/particiones, instrumenta fases, lanza procesos nuevos y monitoriza
  CPU/RAM. El notebook permanece intacto.
- Cambios de control experimental: seeds explícitas,15 épocas fijas sin parada
  temprana y número de hilos documentado; se preservan arquitecturas y operaciones.
- Se añade resumir_cpu.py para reconstruir estadísticas, tablas y figuras desde
  tiempos crudos, con intervalos t95%, variabilidad y revisión de equivalencia.
- No hay reinstalación ni descarga. Datos grandes/checkpoints locales en data/,
  ignorados por Git; manifiestos, hashes, configuraciones y evidencia en resultados/.
- Criterio de cierre: corridas completas, hashes/tensores/estados iniciales
  equivalentes, informe y gráficas revisados. Los errores se conservan.
- Resultados observados: referencia común 1119.928 ± 2.723 s (DE muestral,
  n=5); total adaptativo 1858.617 ± 314.366 s. Una selección cambió a CNN-A+aug;
  se estratifican ramas y se documenta el alcance común mediante DEC-014/015.
- El controlador original congelado no se editó después de preparar EXP-002.
  Un supervisor cerró la serie tras cinco salidas correctas; su estado anterior
  se conserva. La extensión06 solo preparó contexto y no produjo una medición.
- Se añadieron utilidades de cierre, sellado y preservación, y una guía/script
  para preparar otra identidad con los mismos manifiestos sin volver a muestrear.
- Auditoría: checkpoints iniciales/finales y manifiestos íntegros; 25 contadores
  Adam de 1440; tensores/paquetes/fotos/estados iniciales coincidentes. La tolerancia
  de accuracy 1e-6 no se alcanzó en CNN-B/variante aumentada; limitación explícita.
- Copia local determinista de 17,400 imágenes en data/EXP-002/fuente-seleccionada.zip,
  fuera de Git, con identidad en preservacion-datos.json/.md. No hubo descarga.
- Cuatro gráficas renderizadas e inspeccionadas. Verificación documental y límites
  en resultados/preparacion/verificacion-preparacion.md y BIT-005.
- No es optimización CUDA ni medición de speedup; no se generó el informe PDF final.


## CAM-004 — Identificación documental de oportunidades CUDA, paso 3

- Fecha UTC: 2026-11-17. Responsable: Codex; revisión humana pendiente.
- REQ-04/12; PROC-11; DEC-003/016; BIT-006.
- Identidad previa: notebook SHA-256
  `ce8824e860ae7e02684bf702fd6865b7d87cc9eb71917f56bfb0ee8d9ccb8e8a`,
  intacto; estrategia previa SHA-256
  `c68471b1ba561f40b26bacd399c385836236b1258e4d0652447eaccc48e9c10f`.
- Antes: candidatos genéricos sin selección sustentada tras la medición CPU.
  Ahora: [paso 3](oportunidades-paralelizacion.md) con nueve oportunidades,
  independencia, dependencias, prioridad CNN-B y exclusiones justificadas.
- Archivos: nuevo documento; índices raíz/trazabilidad, requisitos, estrategia,
  decisiones, procedimientos, cambios y bitácora. La comprobación documental
  actualiza su reporte mutable en resultados/preparacion.
- Datos y procedimiento experimental: sin cambios; reutiliza resultados EXP-002
  con identidad, sin nuevas corridas ni modificaciones de evidencias selladas.
- Impacto: completa REQ-04; no diseña distribución CUDA ni modifica código.
  Los porcentajes por fase son cálculos derivados, no speedup ni perfil por kernel.
- Validación: verificador documental y diff registrados en BIT-006; no requiere
  entrenamiento. Commit de incorporación pendiente de realizar por el equipo.

## CAM-005 — Diseño completo DIS-CUDA-001, sin implementación

- Fecha:2026-11-17 UTC. Responsable:Codex; revisión humana pendiente.
- Solicitud:paso4 completo y cauteloso; REQ-05/12; PROC-12;
  DEC-017/018/019/020; BIT-007. Paso5 no iniciado.
- Identidad previa: commit limpio
  `2a14000a54fb38bc72dc9a45d5438ca726ae76bf`; estrategia previa SHA-256
  `8e1a250ade67268cce47898e7b8442d0d3c9b25d74c048242e360c358ae7ccc3`.
  El notebook mantiene su identidad inicial, no se modifica código.
- Antes: selección del paso3, sin reparto concreto. Ahora: diseño principal,
  catálogo K01–K19, validación/medición e inspección de evidencia. Se especifican
  arquitectura, índices, grids/blocks/threads, barreras/reducciones, gradientes,
  Adam/BN, máscaras, memoria, transferencias y condiciones de aceptación.
- Stack propuesto numba-cuda/CUDA Python; consulta de inventario GTX1650,4GiB,
  sm_75. No se instalan paquetes ni se prueba capacidad de ejecutar kernels.
- Diseño requiere entrenamiento CNN-B base y futura CPU pareada con orden y
  máscaras explícitos. Las modificaciones de código y creación de EXP quedan
  reservadas al paso5; no se editan configuraciones o evidencias EXP-002.
- Archivos: seis Markdown nuevos, incluida copia exacta de la clase y contraste;
  índices raíz/trazabilidad/resultados,
  requisitos, estrategia, reproducibilidad, decisiones, procedimientos, cambios
  y bitácora. Verificador actualiza su reporte mutable fuera de RUN sellados.
- Impacto observado: documental e inventario; no hay tiempos ni speedup nuevos.
  Presupuesto de679.698MiB es derivado, no reserva/consumo real observado.
- Validación: revisión cruzada de fórmulas y lectura del estado NPZ, más verificador
  documental y git diff --check registrados en BIT-007. GPU permanece sin validar.
- Commit de incorporación pendiente del equipo. No sustituye el paso3 ni borra
  sus decisiones históricas; desarrolla su selección para el alcance CNN-B.
- Petición adicional: contrastar Semana15 como fuente complementaria. Se preserva
  el archivo externo por bytes, se añade matriz de correspondencia y precisiones;
  se refina R(L,C) a32×8 para lectura contigua de canales, sin variar presupuesto.

## CAM-006 — Copia del notebook e implementación CUDA Python

- Estado:completado para CNN-B,2026-11-17UTC; responsable:Codex por solicitud del usuario.
- Identidad original preservada SHA-256
  ce8824e860ae7e02684bf702fd6865b7d87cc9eb71917f56bfb0ee8d9ccb8e8a.
- REQ-06/07/09/12; DEC-021; BIT-008. Solo se modifica copia y módulos nuevos;
  no se edita benchmark_cpu.py ni EXP-002 congelado.
- Revisión previa: se cancela obligación de otra campaña CPU; reutilizar fases
  compatibles de CNN-B, sin exigir igualdad de máscaras históricas Dropout.
- Entorno: PCUDA, añadir solo paquetes ausentes; toolkit13.1 local. Instalaciones,
  pruebas y resultados se registran al producirse, sin estados ficticios.
- Se crearon copia de83celdas,módulos src/asl_cuda,kernels Python y wrapper
  SGEMM;controlador/validadores/notebookejecutor/análisis/captura/reproducción.
  Original y benchmark_cpu.py mantienen identidades congeladas.
- Revisión DEC-022:SGEMM reduce ensayo preliminar0.60→0.26s/lote;workspace576MiB,
  tope1408MiB. Grids propios explícitos,grids internos cuBLAS gestionados por NVIDIA.
- Admisión:51pruebas de operadores,78adjuntos locales y10pasos sintéticos con
  guardado/recarga. Diagnósticos globales estrictos fallidos conservados;DEC-023
  distingue sensibilidad de rutas pooling cercanas a empates de error de operador.
- EXP-003:cinco corridas completas,15épocas/1440pasos cada una,sin nueva campaña
  CPU. Media418.607683±1.916308s frente a805.649726±2.722684s CPU;speedup1.924594×,
  H2D/D2H/CPUstaging/validación incluidos;test98.620689% en las cinco.
- Contenido canónico de cinco checkpoints idéntico,contadorAdam1440;hashes de
  archivo y de contenido se distinguen. Datos X/y coinciden byte a byte con CPU.
- Capturas reales inspeccionadas CPU/GPU,cuatro gráficas,CSV e informe PDF8páginas;
  pruebas,DPI e historialIPython conservados/corregidos sin sobrescribir RUN.
- Se guardó notebook de presentación validado,código nativo0,37celdas sin error;
  consulta la serie completa sin generar entrenamientos adicionales.
- [Resultados](../../resultados/paralelo/EXP-003/resultados.md),
  [implementación](implementacion-cuda.md) y guía de reproducción disponibles.
  Verificación documental y cierre en BIT-008. Commit/revisión humana pendientes.

## CAM-007 — Salidas reales de la última corrida dentro del notebook

- Solicitud adicional del usuario;actividad de presentación,sin otro EXP/RUN.
- Se entrega notebooks/Lab3_ASL_CUDA_con_salidas.ipynb,con código y salidas
  reales de RUN-20261117T234658Z-gpu-05:15épocas,métricas,tiempos,hardware y curvas.
- Se preserva notebook fuente congelado SHA-256
  53050fbcc27a8629850d36165c5c4387982714277f09a58e256e3537b051b86f.
- Ejecutor previo SHA-256eafddcb8c5d54ca068b6bcb3e1385a355c1baf4210b019aa346d8c224c8fd5a1;
  se añade argumento --entrada,sin cambiar el default ni lógica de entrenamiento.
  Su versión anterior está preservada en analisis_fuentes del EXP-003 sellado.
- Se añade notebook_con_salidas.py,plantilla y registro de ejecución. La vista
  fuerza GPUFalse y lee evidencias reales;los logs no se simulan como fit nuevo.
- Validación:nativo0,44celdas sin errores,15líneas EPOCH,5imágenes incrustadas.
  SHAserie idéntico antes/después y184archivos sellados intactos. BIT-009.
- Índices actualizados para abrir directamente el notebook con salidas.
  No se edita el original,los modelos,las mediciones ni el informe sellado.

## CAM-008 — Auditoría previa al paso 6

- Solicitud: verificar proceso, datos y resultados contra el PDF sin repetir
  experimentos. Responsable: Codex; BIT-010.
- Se añade revision-pasos-1-5.md, script auditar_pasos_1_5.py y resultados de
  auditoría en carpetas nuevas. Se ajustan índices, DAT-05 y estados REQ-07/09
  para distinguir carga equivalente de condiciones idénticas y uso de saturación.
- Identidades previas SHA-256 de documentos ajustados:

| Archivo | SHA-256 previo |
|---|---|
| requisitos.md | 11e766b79a372432781c00b435649dc75488cc695b608a12863daa6f692aed36 |
| datos.md | 2fe031574f6e64009236255118124b7364ae4cde37db5243b82788a4adfca2ce |
| README de trazabilidad | 9f3db3cc8f1ae52249c5d78f1f936ae431090da8d570ad72fc0e4ba5f0ae9f9c |
| README raíz | 50d0753313305e371af37d1ed45d86e53465cf72d7e1409070e2c1cb35e534d9 |
| implementacion-cuda.md | 268a9576ceca7d6e8aacd987e5e0b50da2c709b2beeb92fc1f54804dbcba15bf |

- Primer verificador: SHA58928b025cc1a31a9cd00991f1038826bd730062de7ff270e531589e1c807e56;
  registró un error de lectura de encabezado GPU. Se corrigió a esquema dual
  ruta/ruta_relativa y argumento --destino antes de un segundo intento nuevo.
  Script aprobado SHA5bdc5ea9f00379a1c65d83389c3578968c372b90536309367e0dfb5c8e53178a.
- Resultado:80 controles aprobados;10 manifiestos RUN,184 entradas del sello,
  datos crudos,estados/modelos,estadísticas y salidas de notebook coherentes.
- No se modifica código de entrenamiento,datos,notebooks,configuraciones ni
  evidencias selladas CPU/GPU. No hay nuevas métricas experimentales. Las
  reservas documentadas no se convierten en validaciones completas.

## CAM-009 — Paso 6: análisis y conclusiones sin informe final

- Solicitud explícita del usuario;DEC-024;PROC-09;BIT-011;REQ-08/10/12.
- Se añade scripts/analizar_paso_6.py y derivados en
  resultados/comparacion/paso-6-revision-02:Markdown,CSV,3PNG,3SVG,procedencia y
  manifiestos. No se modifica implementación ni mediciones ni se exporta PDF.
- Identidades previas SHA-256 de documentos ajustados:

| Archivo | SHA-256 previo |
|---|---|
| README raíz | 270533eebeafb7620721cf166a2cc06ddc0495c60d7c6af75820d1d8bd7a5add |
| README trazabilidad | 7428b570c45f01fb069c1095bdfbc1dbf08cc58c22fab82fd64c62d8424a867 |
| requisitos.md | 027923347b16247ae6cdfe746b075158842b3bdebe437417d7eb3f841cde6186 |
| README resultados | 024347b50e5a1badaefda152a054d6618322f56e92e5aaa81db376f74b62143e |
| README comparación | 662ead6b67b8643d0f10b0d4ebfb8145868f03197a74bb31235dafc35b4698d9 |

- Se conserva primera revisión en resultados/comparacion/paso-6. Revisión visual
  detectó nota inferior recortada en figura2;se retiró esa nota redundante y se
  ajustó bbox,padding y precisión del texto sobre residuo/eventos y condiciones.
  No cambian cifras. Fuente previa0b8c825b4e416a2eb942d72c64a728e19528e91f28ad46cac38936ddc1f92e51
  conservada en resultados/preparacion/paso-6-borrador/analizar_paso_6.py.
- Fuente finalafc0895763f9dbdb746ece077fe6cfd80843bd93bc365d8cdb589ddb2fb43ed2.
  Manifiesto final de13derivados06a459c79c1b23fcd8060272a05f7a7b419b24ed72e1ef33fbff8c6114114087;
  27entradas fuente verificadas. Segunda revisión es la vista vigente.
- Análisis conserva límites:CPUhistórica/RNG/workers;alcanceCNN-B,transferencias
  anidadas,fallosglobales y saturación no demostrada. Hipótesis causales se
  distinguen de observaciones y cálculos. REQ-07/09 mantienen sus reservas.
- Se actualizan índices y requisitos del paso6;informe/PDFfinal,equipo y entrega
  global siguen pendientes. No se alteran notebooks ni evidencias selladas.

## Plantilla de cambio

```markdown
## CAM-NNN — Título
- Fecha UTC, autor y revisor:
- Motivo / REQ / DEC / incidencia:
- Archivos y funciones afectadas:
- Comportamiento anterior → nuevo:
- Datos, entorno o protocolo afectados:
- Impacto esperado y riesgos:
- Validación realizada y evidencia:
- Impacto observado (RUN y métricas, o pendiente):
- Comparabilidad con resultados anteriores:
- Commit / PR / cambio al que sustituye:
```

Un cambio de optimización requiere demostrar corrección, además de rendimiento.
Un cambio de medición o datos puede invalidar comparaciones y exige explicarlo.
No reescribir retrospectivamente los registros de corridas anteriores.
