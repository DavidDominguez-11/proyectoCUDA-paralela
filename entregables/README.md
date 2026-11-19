# Informe final editable

Código por separado,con ejecuciones nuevas y todas las salidas nativas:
[notebookCPU](notebooks/ASL_Secuencial.ipynb) y
[notebookCUDA](notebooks/ASL_CUDA.ipynb). [Detalle y huellas](notebooks/README.md).

Entrega vigente: [Informe_Proyecto_3_CUDA.docx](Informe_Proyecto_3_CUDA.docx).

- 13 páginas verificadas en Microsoft Word16.0.
- Portada confirmada:David Domínguez23712,Jonathan Diaz23873,Luis Padilla23663.
- Seis apartados obligatorios en orden,12tablas y7figuras incrustadas.
- Capturas CPU/GPU originales,logs reales,monitorización y tres preguntas técnicas.
- Referencias APA7;el cuerpo no remite a archivos Markdown internos.
- Sin PDF nuevo:el usuario realizará su exportación.

La [fuente Markdown](informe-final-v02/informe-rendimiento.md),sus imágenes y
[fuentes con SHA-256](informe-final-v02/fuentes.csv) permiten revisar la procedencia.
El DOCX de esa carpeta y el enlace principal contienen exactamente los mismos bytes.
SHA-256:c1fa886f3bf4b747c1dbb17937dc193eabf645a088e3cabd46c2edea802a22c7.
[Manifiesto de entrega y revisión](informe-final-v02/artefactos.csv),con bytes y
SHA-256;rutas relativas a la raíz del repositorio.

La [validación](../resultados/preparacion/informe-docx-revision-entrega/validacion.json)
comprueba estructura,datos principales,33identidades fuente,imágenes embebidas y
ausencia de tablas cortadas. La vista XPS/PNG se utilizó solo para inspeccionar
maquetación,con Word en lectura;no se convirtió a PDF.

Las carpetas informe-rendimiento-v01,informe-rendimiento-v02 e informe-final
conservan iteraciones previas de tipografía/maquetación. La versión vigente es
informe-final-v02 y la copia principal enlazada arriba;no hubo nuevas mediciones.

El uso real del hardware queda documentado. Las capturas no acreditan saturación
literal de VRAM ni ocupación porSM,limitación que se explica expresamente. La
exportación PDF y contribuciones verificables del equipo siguen pendientes.


## Presentación visual

[PowerPoint editable](Presentacion_Proyecto_3_CUDA.pptx),31diapositivas;[guion](presentacion/Guion_de_exposicion.md) y[vista general](presentacion/Vista_general.png). Gráficas/tablas editables,capturas reales ynotas en todas las láminas.


## Guía para mostrar código

[Guia_Codigo_CUDA.md](Guia_Codigo_CUDA.md):ruta rápida,ubicaciones exactas yguion literal,con atención especial asincronización,streams,memoria,kernels,cuBLAS yFC.
