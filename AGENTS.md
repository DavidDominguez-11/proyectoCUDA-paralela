# Instrucciones del proyecto

## Fuente de verdad

`docs/Proyecto 3 CUDA.pdf` es la autoridad absoluta. Consultar la matriz
`docs/trazabilidad/requisitos.md` y contrastar cualquier duda con el PDF.
La transcripción Markdown contiene ampliaciones explicativas no normativas.

## Trazabilidad de cada trabajo

- Antes de modificar el proyecto, revisar `docs/trazabilidad/README.md`, las
  decisiones abiertas y la bitácora. Conservar el laboratorio original como
  referencia; documentar cualquier modificación mediante CAM y su identidad previa.
- Registrar actividad, procedimiento, datos, decisiones y cambios en los Markdown
  correspondientes. Actualizar el estado de requisitos solo con evidencia.
- Documentación informativa en `.md`; resultados en `.md` con sus CSV, logs,
  configuraciones, capturas y gráficas fuente. El informe final también debe
  exportarse a PDF según la consigna.
- Distinguir inspección estática, antecedentes históricos, hipótesis, medición y
  cálculo derivado. No inventar resultados ni marcar pendientes como completados.
- Para ejecutar experimentos, seguir `docs/trazabilidad/reproducibilidad.md` y
  `procedimientos.md`: EXP/configuración congelada y RUN nuevo por ejecución.
  No sobrescribir evidencias experimentales. Conservar errores y exclusiones.
- Mantener identidad SHA-256 de datos, particiones, configuraciones y fuentes;
  comandos exactos, entorno, semillas y responsables. No mezclar cargas distintas
  en una comparación CPU/GPU.
- Un cambio documental no requiere entrenar. Validarlo con
  `python -X utf8 scripts/verificar_preparacion.py` y registrar sus límites.
- No guardar credenciales. Conservar datos/modelos grandes fuera de Git con
  manifiestos y ubicación durable; mantener evidencia liviana versionada.

Estas reglas son convenciones internas solicitadas para la trazabilidad. No
certifican que el algoritmo o hardware ya hayan sido ejecutados o validados.
