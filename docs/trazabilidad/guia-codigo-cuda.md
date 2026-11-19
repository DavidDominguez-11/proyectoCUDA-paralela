# Guía para demostrar el código CUDA

Entrega: [Guia_Codigo_CUDA.md](../../entregables/Guia_Codigo_CUDA.md). DEC-030, CAM-015, BIT-017.

Documento técnico para exposición: ruta rápida de seis paradas,17apartados,guion literal,archivo/función/líneas,búsquedas del notebook ypreguntas frecuentes. La sección6distingue barrera de bloque,orden de operaciones de un stream yespera del host. Se explica memoria pinned/device/shared,grid-stride,BN,softmax,backward,Adam,banderas atómicas,cuBLAS/im2col,FC-A/FC-B,warmup ymedición. Se identifica la ruta cuBLAS realmente medida frente albackend explícito alternativo.

La información se obtuvo leyendo las fuentes actuales,analizando AST yleyendo el JSON del notebook. Se consultaron páginas oficiales de NVIDIA Numba-CUDA para significado de API ycuBLAS13.1 para stream ySGEMM. No se ejecutaron kernels,modelos,benchmarks ni celdas. El documento separa comportamiento observado en código,mediciones históricas,derivaciones ypropuestas futuras;no convierte la inspección en certificación de nueva ejecución GPU.

Validación:104enlaces locales,82anclas de línea,45ubicaciones de celdas y5bloquesPython de sintaxis válida. Funciones/clases contrastadas con rangos AST;hashes de fuentes yentregables previos iguales. Se verificaron específicamente la activación `backend="cublas"` del controlador,la captura de momentum enFC,la creación delstream yasociación cuBLAS. Se corrigieron dos referencias puntuales antes del cierre:controlador178ymomentumFC110. El índice delnotebook cuenta celdas desde1,incluyendoMarkdown;contadores deejecución reiniciados por sesión se muestran aparte.

Artefactos: [fuentes ySHA-256](../../resultados/preparacion/guia-codigo-cuda-v01/fuentes.csv),[símbolos](../../resultados/preparacion/guia-codigo-cuda-v01/simbolos.json),[índice de notebook](../../resultados/preparacion/guia-codigo-cuda-v01/indice-notebook.json),[validación](../../resultados/preparacion/guia-codigo-cuda-v01/validacion.json),[identidades previas](../../resultados/preparacion/guia-codigo-cuda-v01/identidades-previas.json).

Comandos de documentación,desde la raíz:

```powershell
.\PCUDA\Scripts\python.exe -X utf8 scripts/crear_guia_codigo_cuda.py
.\PCUDA\Scripts\python.exe -X utf8 scripts/verificar_guia_codigo_cuda.py --pulir
.\PCUDA\Scripts\python.exe -X utf8 scripts/cerrar_guia_codigo_cuda.py
.\PCUDA\Scripts\python.exe -X utf8 scripts/verificar_preparacion.py
```

El creador rechaza destinos existentes;no repetirlo sobre la misma carpeta. El verificador hace análisis estático,comprueba hashes ypuede ajustar el formato del documento con --pulir;no importa la implementación CUDA. Los snippets solo añaden espaciado yel decorador visible para exponer;no se modifican fuentes. No se tocaron losPPTX,PDF,DOCX,notebooks niproyectos experimentales.
