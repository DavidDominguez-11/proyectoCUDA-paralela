# Preparación y verificación del informe final DOCX

Responsable:Codex,por solicitud del usuario. DEC-025;CAM-010;BIT-012.
Fecha:2026-11-18UTC,17noviembre local. No se repitieron entrenamientos.

## Archivo vigente

[Informe editable](../../entregables/Informe_Proyecto_3_CUDA.docx):13páginas,
12tablas y7figuras. SHA-256:
c1fa886f3bf4b747c1dbb17937dc193eabf645a088e3cabd46c2edea802a22c7.

La portada usa los nombres/carnés confirmados por el usuario. No se inventan
contribuciones individuales ni se certifican commits de los tres integrantes.
La instrucción de esta entrega es **crear DOCX y no exportar PDF**;el usuario
hará la conversión. El requisito de PDF final queda pendiente de ese paso.

## Cobertura de la guía en el documento

| Contenido | Páginas Word | Evidencia o tratamiento |
|---|---|---|
| 1.Análisis secuencial | 2 | Partes,recorrido,tabla de12operaciones agrupadas y8repetitivas;distingue historial y protocolo fijo. |
| 2.Medición CPU | 3–4 | Dataset,hardware,procedimiento,cinco corridas,medias/DE,14fases comunes,gráfica y factores. |
| 3.Oportunidades | 5 | Ocho grupos seleccionados,beneficio esperado,exclusiones y dependencias. |
| 4.Diseño | 6 | Arquitectura,flujo CPU→H2D→GPU→D2H→CPU,threads/blocks/grids,memoria y decisiones realizadas. |
| 5.Implementación | 7–9 | Stack,condiciones,cinco GPU,restauración,timing,validación y evidencia. |
| Captura CPU | 8 | Original23:39:47UTC;12procesadores lógicos,CPU225.2%,GPU1%;preparaciónG04. |
| Captura GPU | 9 | Original23:37:06UTC;muestraGPU88%,1213/4096MiB,CPU98.9%;entrenamientoG03. |
| Logs y reconocimiento | 8–9 | Dispositivo efectivoCPU,GTX1650real,logsCPUliterales y cierre1440pasosGPU;cinco salidas0. |
| 6.Análisis | 10–13 | Tabla por fase,fórmulas,gráficas,desglose,calidad y conclusiones. |
| Pregunta CPU/GPU | 11 | Explicación basada en tareas y tiempos,evitando una afirmación universal sobreCPU. |
| Pregunta cambio por CUDA | 11 | Kernels/SGEMM,buffers y ahorro387.042s,48.041%;distingue preparaciónCPU. |
| Pregunta factores/límites | 12 | Copias,validación,dependencias,Host,memoria y límites de perfilado. |
| APA7 | 13 | Cuatro referencias primarias verificadas,citas en texto,sangría francesa y títulos en cursiva. |

## Reservas que permanecen

REQ-07 conserva diferencias de CPUhistórica,horario,RNG y preparación1→6workers.
REQ-09 acredita uso/carga real;la saturación literalGPU y ocupación porSM no se
demuestran. No se genera una falsa captura del fitCPUhistórico:esa fase aporta
logs/CSV,y la capturaCPUes preparación de la variante optimizada. Los fallos de
comparación global estricta siguen registrados como fallidos. El DOCX expresa
estas reservas;no declara cumplimiento íntegro de una condición no observada.

## Producción y revisión

Se utilizó Python3.14.3 del sistema con python-docx1.2.0 ya instalado. PCUDA no
se modificó. Word16.0 abrió en lectura para repaginar y exportar exclusivamente
XPS;PyMuPDF renderizó13PNG para inspección. El DOCX conservó su SHA-256 tras
abrir/cerrar Word. El comando `/preview` no está disponible;se usó esa revisión
local conforme al mecanismo alternativo de la habilidad preview.

Las primeras tres iteraciones se preservaron y solo cambiaron presentación:
espaciado tipográfico,unidades,sangría/cursivasAPA y agrupación de miles. Las
cifras,gráficas y capturas no se alteraron. La fuente de cada iteración anterior
está preservada en su carpeta;fuente final en scripts/crear_informe_docx.py.

Comandos finales desde raíz:

```powershell
python -X utf8 scripts/crear_informe_docx.py --destino entregables/informe-final-v02
python -X utf8 scripts/revisar_docx_word.py --entrada entregables/informe-final-v02/Informe_Proyecto_3_CUDA.docx --destino resultados/preparacion/informe-docx-revision-entrega
python -X utf8 scripts/verificar_informe_docx.py --carpeta entregables/informe-final-v02 --revision resultados/preparacion/informe-docx-revision-entrega/revision.json --salida resultados/preparacion/informe-docx-revision-entrega/validacion.json
Copy-Item -LiteralPath entregables/informe-final-v02/Informe_Proyecto_3_CUDA.docx -Destination entregables/Informe_Proyecto_3_CUDA.docx
.\PCUDA\Scripts\python.exe -X utf8 scripts/verificar_preparacion.py
git diff --check
```

Los generadores y validadores rechazan destinos existentes. Para reproducir se
deben usar nombres nuevos;no sobrescribir la entrega o sus evidencias.

- [Generación con intérprete/comando/UTC](../../entregables/informe-final-v02/generacion.json).
- [Revisión Word y páginas de cada tabla](../../resultados/preparacion/informe-docx-revision-entrega/revision.json).
- [Validación de estructura y33fuentes](../../resultados/preparacion/informe-docx-revision-entrega/validacion.json).
- [Fuentes de la composición](../../entregables/informe-final-v02/fuentes.csv).
- [Manifiesto de entrega y revisión](../../entregables/informe-final-v02/artefactos.csv),con rutas relativas a la raíz del repositorio.
- [Fuente informativa Markdown](../../entregables/informe-final-v02/informe-rendimiento.md).

## Fuentes públicas verificadas para las referencias

Consultadas17noviembre2026local. Se verificaron títulos/autor institucional y
definiciones técnicas;no se usaron para sustituir mediciones locales:

- [Akash:ASL Alphabet,Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).
  El buscador de la página identifica al creador como Akash;sin fecha estable
  verificada,se usa s.f.en la referencia.No se atribuye otro autor o año supuesto.
- [NVIDIA:Writing CUDA Kernels](https://nvidia.github.io/numba-cuda/user/kernels.html).
  Fuente primaria para jerarquía explícita y kernels PythonGPU.
- [NVIDIA:cuBLAS13.1](https://docs.nvidia.com/cuda/archive/13.1.0/cublas/index.html).
  Fuente primaria versionada para BLASGPU y adaptación de almacenamiento.
- [NVIDIA:nvidia-smi](https://docs.nvidia.com/deploy/nvidia-smi/index.html).
  Fuente primaria sobre monitoreo;actividad agregada no se renombra como ocupaciónSM.

El cuerpo del DOCX no contiene referencias a rutas Markdown internas. Esta
carpeta conserva esas relaciones de trazabilidad fuera del informe autónomo.


## Ampliación posterior vigente: FC-A y FC-B

El alcance inicial CNN-B se amplió con dos entrenamientos completos adicionales en CUDA. Se conservaron las mediciones y evidencias anteriores. [Diseño,procedimiento,decisiones,resultados,notebooks e informe actualizados](ampliacion-fc.md). Los conteos/hashes de la entrega anterior descritos arriba son antecedentes; la entrega actual CUDA tiene92celdas de código y30PNG, y el DOCX16tablas/13imágenes con el diseño editado por el usuario.
