"""Record measured FC results and update documentation; never train a model."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,csv
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'resultados/preparacion/ampliacion-fc'
def append(path,text):
    with (ROOT/path).open('a',encoding='utf8') as f:f.write('\n\n'+text.strip()+'\n')
def replace(path,a,b):
    p=ROOT/path;s=p.read_text(encoding='utf8');assert a in s,(path,a[:50]);p.write_text(s.replace(a,b),encoding='utf8')
def main():
    data=json.loads((ROOT/'resultados/comparacion/ampliacion-fc/resumen.json').read_text(encoding='utf8'))
    review=json.loads((BASE/'revision-word-04/revision.json').read_text(encoding='utf8'))
    valid=json.loads((BASE/'validacion-final.json').read_text(encoding='utf8'));assert valid['estado']=='aprobado'
    nb=json.loads((BASE/'integracion-notebook.json').read_text(encoding='utf8'))
    pages=review['paginas']
    text=f'''# Extensión a FC-A y FC-B: diseño, ejecución y cierre

Estado: implementada y medida. DEC-027, CAM-012, BIT-014. Autor de implementación/ejecución: Codex por solicitud expresa del usuario. No se atribuyen contribuciones individuales humanas sin evidencia. El PDF normativo sigue siendo la autoridad; esta extensión amplía la selección inicial CNN-B.

## Alcance autorizado y preservación

Solo ejecutar trabajo nuevo FC; reutilizar las cinco referencias CPU EXP-002 y CNN-B EXP-003. No se entrenó de nuevo ninguna CPU ni CNN-B. No se instalaron paquetes ni se descargó dataset. Los cinco RUN CPU utilizados son los01–05 de serie.json; el intento06 histórico no forma parte de esa referencia. EXP-008/009 continúan como observaciones adicionales n=1, sin mezclarse con los promedios.

El DOCX editado por el usuario se preservó antes de añadir contenido: SHA-256 ba0952a8acd1e1b1abaf38cde208f4faf064c3aa2b18a2db17b8b56e1e35b427. [Identidades y copias previas](../../resultados/preparacion/ampliacion-fc/identidades-previas.json). Se verificó la conservación de152elementos XML previos, estilos, tema y pie de página. Los únicos cambios de la entrega CUDA son celdas adicionales; sus53celdas anteriores se conservan idénticas. El notebook CPU permanece idéntico byte a byte.

## Decisiones de diseño y código

La extensión está en [fc.py](../../src/asl_cuda/fc.py). Reutiliza métodos Dense/BN/Dropout/softmax/backward/Adam, el adaptador cuBLAS y kernels ya medidos, sin editar los módulos CNN-B congelados. Nueva clase FCNetwork define buffers según perfil FC, semilla por arquitectura y reducción final BN por característica. No basta cambiar un nombre de modelo: se conserva momentum0.99 en FC frente a0.9 en CNN-B. Epsilon0.001, Adam0.001/beta0.9/0.999/epsilon1e-7 y FP32 se mantienen.

| Propiedad | FC-A | FC-B |
|---|---|---|
| Entrada |64×64×3, Flatten12,288 |Igual |
| Capas ocultas Dense/ReLU |128,64 |512,256,128 |
| Dropout tras cada oculta |0.3 |0.4 |
| Salida |Dense29/softmax |Igual |
| Parámetros entrenables |1,607,709 |6,484,509 |
| Buffers CUDA propios |51,864,680bytes,49.46MiB |130,704,488bytes,124.65MiB |
| Semilla |44 |45 |

Hay BN antes de las Dense, sobre12,288características, no sobre3canales RGB. Se usa media/varianza poblacional por lote. Workspace de convoluciones/im2col eliminado del perfil FC. Valores de buffers propios no incluyen el contexto ni allocations internas cuBLAS y no equivalen a saturación VRAM.

| Operación | Threads por block | Blocks/grid |
|---|---|---|
| Reducción BN parcial |(32,8,1) |(ceil(B/2048),ceil(C/32),1); entradaC12288 yB128/20:1×384 |
| Finalización BN |256 |ceil(C/256); entrada48bloques; un hilo por característica |
| ReLU/Dropout/BN/Adam/finite |256 |min(4096,ceil(E/256)), recorrido grid-stride |
| Softmax/pérdida |32 |B bloques;128/20 |
| Dense forward y gradientes |Interno cuBLAS |SGEMM FP32, PEDANTIC y sin atomics; no inventar jerarquía interna |

Dependencias entre capas y actualizaciones permanecen ordenadas en un único stream. Los gradientes se completan antes de actualizar pesos. CPU prepara lotes/máscaras y recibe estado para validar; toda la aritmética entrenable de estas redes se lanza a GPU. Preparación de archivos y validación/test se identifican comoCPU.

## Pruebas nuevas y experimento aplicado

[Validación FC](../../resultados/preparacion/ampliacion-fc/validacion-01/validacion.json):4casos, ambas redes con lotes20/128. Se compararon probabilidades, todos los gradientes, medias/varianzas móviles, primera actualización Adam y restauración contra un oráculo CPU con los mismos datos/máscaras. Error máximo absoluto probabilidades1.67e-6, Adam6.32e-6; tolerancias en el script, sin evaluar nuevamente las campañas anteriores. Esta prueba local no demuestra identidad de15épocas con RNG histórico distinto.

Datos:17,400imágenes del manifiesto histórico,12180train/2610val/2610test;64×64RGB,float32/255. Estados iniciales completos históricos FC-A/FC-B (incluye Adam) restaurados; hashes en configuración. Cada RUN compara SHA-256 de los seis tensores X/y contra CPU histórica. Se mide con15épocas,96lotes/época,1440pasos y último lote20. Preparación CPU6workers; warmup128+20trainGPU/128valCPU/32testCPU, excluido y restaurado. Validación CPU batch128 cada época, testCPU32 separado. Tiempo pared incluye staging, máscara Bernoulli, H2D, matemáticas GPU, D2H y validaciónCPU; sin restar sobrecostes. GPU PCG64 por semilla/época/lote/capa; no se afirma identidad de máscaras/orden históricos.

| Identidad | Resultado | Tratamiento |
|---|---|---|
| EXP-010 |FC-A completó15épocas y test; falló CSV antes de guardar modelo |Intento incompleto excluido; error y captura preservados. |
| EXP-011 |FC-A1 completó15épocas, test, checkpoint y monitor; falló solo Styler por jinja2 ausente |Se reutiliza SIN volver a entrenar. El notebook fallido íntegro queda archivado. |
| EXP-012 |Solo nueve corridas pendientes:FC-A2–5 y FC-B1–5 |Completadas; su configuración también referencia FC-A1 guardada. |

La colección analizada contiene exactamente cinco corridas completas por modelo,150épocas y14,400actualizaciones. Incluyendo el intento incompletoEXP-010 se ejecutaron165épocas FC durante el desarrollo, sin ocultarlo ni incorporarlo a la media. La segunda interrupción fue de presentación: no ocasionó repetir su entrenamiento. Se eliminaron Styler/tabulate de la presentación para trabajar sin instalar dependencias. Un primer análisis por glob detectó el intento CPU06; se corrigió a los cinco RUN explícitos antes de calcular estadísticas. No se alteraron mediciones para corregir estas incidencias.

## Resultados y límites

| Entrenamiento | CPU media±DE(s),n=5 | CUDA media±DE(s),n=5 | Speedup | Reducción |
|---|---|---|---|---|
| CNN-B histórica |805.649726±2.722684 |418.607683±1.916308 |1.924594× |48.040982% |
| FC-A nueva |50.715660±1.264624 |15.122541±0.332283 |3.353647× |70.181712% |
| FC-B nueva |102.441082±3.275205 |21.061995±0.201404 |4.863788× |79.439895% |

[Análisis completo con tablas,CSV,estadísticas,calidad,recursos y cuatro gráficas PNG/SVG](../../resultados/comparacion/ampliacion-fc/resultados.md). Speedup por fase entrenamiento/test conservado; el test sigueCPU y su cociente no se atribuye a inferenciaGPU. Intervalos CUDA anidados no se suman otra vez a tiempo pared ni se llaman tiempo activo exclusivo.

**Calidad:**FC-A test53.065%CPU frente a51.724%CUDA; FC-B44.291%CPU frente a39.962%CUDA. Hay reducción de tiempo, no mejora predictiva. Arquitectura/estado/pasos equivalentes no implican la misma trayectoria con distinto shuffle/Dropout y redondeo. No se reentrenó buscando una métrica más favorable. Las cinco GPU son repetibles desde la misma semilla, no cinco inicializaciones independientes.

**Recursos:** capturas reales en primera corrida por red:FC-A50%GPU/111MiB en su muestra seleccionada yFC-B32%GPU/187MiB; consultar JSON originales para UTC. Monitorización por segundoCPU12lógicos/GPU/VRAM de todas las corridas. Son ventanas reales con lectura viva de psutil/nvidia-smi e ImageGrab, no gráficos rotulados como captura. Su coste en primera corrida no se descontó. No se fuerza saturación ni se certifica ocupaciónSM. La actividad menor queCNN-B es compatible con redes más pequeñas por lote y tiempo de host/validación; sin profiler no se identifica un cuello de microarquitectura de forma concluyente.

## Notebooks, informe y evidencia

- [CUDA ampliado](../../entregables/notebooks/ASL_CUDA.ipynb):92celdas de código,30PNG,0errores; conserva53celdas previas. Sesiones nativas separadas y contadores reiniciados explícitamente, sin falsificarlos.150líneasEPOCH FC nativas verificadas. La primera corrida se conserva hasta su entrenamiento exitoso y el cierre se presenta por lectura real de sus archivos; el error de presentación completo está archivado y transcrito en diagnóstico.
- [CPU](../../entregables/notebooks/ASL_Secuencial.ipynb):inalterado,37celdas de código,75épocas y10PNG.
- [DOCX actualizado](../../entregables/Informe_Proyecto_3_CUDA.docx):{pages}páginas revisadas en Word,16tablas y13imágenes; se añaden oportunidades/diseño FC,medición, dos capturas, comparación,calidad yconclusiones. Los seis apartados principales permanecen en orden. Estilos/tema/footer originales preservados byte a byte;152elementos previos XML idénticos en su orden. No se generó PDF.
- [Validación final](../../resultados/preparacion/ampliacion-fc/validacion-final.json):184artefactos CNN-B intactos,CPUinalterada,datos/pasos/checkpoints/capturas/sourcehashes comprobados.
- [Revisión Word final](../../resultados/preparacion/ampliacion-fc/revision-word-04/revision.json):XPS/PNG para inspección local,sin exportar PDF. Se corrigieron únicamente añadidos de maquetación:tabla de cinco columnas,identificadores compactos ysaltos de página;contenido previo/estilos inalterados.
- [Configuración de la colección](../../configuraciones/EXP-012.json),[contexto estático](../../resultados/paralelo/EXP-012/contexto/contexto.json),[notebook nativo nueve corridas](../../resultados/paralelo/EXP-012/FC_CUDA.ejecutado.ipynb),[primera FC-A archivada](../../resultados/paralelo/EXP-011/FC_CUDA.ejecutado.ipynb).

## Procedimiento y comandos realmente usados

Desde la raíz del repositorio. Preparadores/ejecutores rechazan destinos existentes; para repetir toda la colección se necesita otra identidad y adaptar las referencias de recuperación, no sobrescribir evidencia. Los comandos siguientes documentan la ejecución realizada, **no** deben invocarse de nuevo sobre estos destinos.

```powershell
.\\PCUDA\\Scripts\\python.exe -X utf8 scripts/preparar_ampliacion_fc.py
.\\PCUDA\\Scripts\\python.exe -X utf8 scripts/validar_cuda_fc.py
.\\PCUDA\\Scripts\\python.exe -X utf8 scripts/preparar_experimento_fc.py
.\\PCUDA\\Scripts\\python.exe -X utf8 scripts/ejecutar_fc_notebook.py
.\\PCUDA\\Scripts\\python.exe -X utf8 scripts/corregir_exportacion_fc.py
.\\PCUDA\\Scripts\\python.exe -X utf8 scripts/preparar_experimento_fc_v2.py
.\\PCUDA\\Scripts\\python.exe -X utf8 scripts/ejecutar_fc_notebook_v2.py
.\\PCUDA\\Scripts\\python.exe -X utf8 scripts/continuar_fc_sin_reentrenar.py
.\\PCUDA\\Scripts\\python.exe -X utf8 scripts/preparar_experimento_fc_v3.py
.\\PCUDA\\Scripts\\python.exe -X utf8 scripts/ejecutar_fc_notebook_v3.py
.\\PCUDA\\Scripts\\python.exe -X utf8 scripts/resumir_ampliacion_fc.py
.\\PCUDA\\Scripts\\python.exe -X utf8 scripts/integrar_notebook_fc.py
python -X utf8 scripts/ampliar_informe_fc.py
python -X utf8 scripts/revisar_docx_word.py --entrada entregables/Informe_Proyecto_3_CUDA.docx --destino resultados/preparacion/ampliacion-fc/revision-word-04
.\\PCUDA\\Scripts\\python.exe -X utf8 scripts/capturar_contexto.py --destino resultados/paralelo/EXP-012/contexto
.\\PCUDA\\Scripts\\python.exe -X utf8 scripts/verificar_ampliacion_fc.py
```

Los tres ejecutores se lanzaron con Start-Process -WindowStyle Hidden ystdout/stderr redirigidos a controlador-stdout.log/controlador-stderr.log de suEXP; PID guardado. Sus notebooks ejecutan el entrenamiento directamente por celdas medianteNotebookClient, no un subproceso opaco de entrenamiento. El monitor de captura es un subproceso de visualización, no un segundo entrenador. Invocaciones y UTC reales en estado-notebook.json y metadatos de celda. Los logs de los intentos excluidos no se borran. Análisis, integración y revisión de informe no entrenan.

Las validaciones no certifican reconstrucción en otra máquina ni100%de la rúbrica:PDF y contribuciones humanas verificables siguen a cargo del equipo; saturación literalGPU no acreditada. Se conserva esa reserva sin imponer más campañas al usuario.
'''
    (ROOT/'docs/trazabilidad/ampliacion-fc.md').write_text(text,encoding='utf8')
    append('docs/trazabilidad/bitacora.md',f'''## Cierre BIT-014 — {datetime.now(timezone.utc).isoformat()}

Finalizadas FC-A/FC-B: cinco corridas completas por red,150épocas/14,400pasos nuevos válidos, medias15.122541s/21.061995s yspeedup3.353647×/4.863788× contraCPUhistórica. EXP-010incompleto excluido;EXP-011primeraFC-A guardada reutilizada sin repetir;EXP-012nueve restantes. Cuatro pruebas oráculo aprobadas. Dos capturas vivas,logs,CSV,cuatrográficas de comparación,checkpoints ysalidas nativas. NotebookCUDA92celdas/30PNG/0errores conserva53celdas previas;CPUinalterada y184artefactosCNN-B intactos. DOCX{pages}páginas/16tablas/13imágenes mantiene152elementos previos yestilos/tema/footer del usuario. WordXPS/PNG revisado,sinPDF. MétricaspredictivasFCmenores documentadas. [Procedimiento,evidencias,comandos,incidencias ylímites](ampliacion-fc.md).''')
    append('docs/trazabilidad/cambios.md','## Cierre CAM-012\n\nAmpliación FC implementada y medida, código aditivo src/asl_cuda/fc.py; fuentes CNN-B congeladas intactas. Notebook GPU y DOCX actualizados conservando identidades previas y diseño del usuario. [Trazabilidad completa](ampliacion-fc.md).')
    append('docs/trazabilidad/decisiones.md','## Cierre DEC-027\n\nAceptada e implementada para FC-A y FC-B. Se reutilizaron CPU/CNN-B previas y la primera FC-A ya guardada. Diez corridas completas FC, sin reoptimizar la métrica predictiva. Uso de GPU demostrado con capturas reales; no se fuerza saturación. [Resultados y reservas](ampliacion-fc.md).')
    for path in ['docs/trazabilidad/implementacion-cuda.md','docs/trazabilidad/estrategia-paralelizacion.md','docs/trazabilidad/oportunidades-paralelizacion.md','docs/trazabilidad/diseno-paralelo.md','docs/trazabilidad/reproducibilidad.md','docs/trazabilidad/procedimientos.md','docs/trazabilidad/notebooks-entrega.md','docs/trazabilidad/informe-final.md']:
        append(path,'## Ampliación posterior vigente: FC-A y FC-B\n\nEl alcance inicial CNN-B se amplió con dos entrenamientos completos adicionales en CUDA. Se conservaron las mediciones y evidencias anteriores. [Diseño,procedimiento,decisiones,resultados,notebooks e informe actualizados](ampliacion-fc.md). Los conteos/hashes de la entrega anterior descritos arriba son antecedentes; la entrega actual CUDA tiene92celdas de código y30PNG, y el DOCX16tablas/13imágenes con el diseño editado por el usuario.')
    replace('README.md','CPU; el original permanece intacto. El alcance medido es CNN-B, no todo el notebook.',
        'CPU; el original permanece intacto. La ampliación posterior incorpora también FC-A y FC-B; no se aceleró todo el notebook.')
    replace('README.md','13páginas,12tablas y7figuras;la exportación PDF queda a cargo del usuario.',
        f'{pages}páginas,16tablas y13imágenes, conservando el estilo editado por el usuario; la exportación PDF queda a su cargo.')
    append('README.md','[Ampliación final FC-A/FC-B: implementación, resultados y evidencias](docs/trazabilidad/ampliacion-fc.md). FC-A15.123s vs50.716sCPU(3.354×), FC-B21.062s vs102.441sCPU(4.864×), cinco observaciones por modelo. Notebook CUDA actualizado con salidas nativas de las tres redes; CPU/CNN-B previas no se volvieron a ejecutar.')
    replace('docs/trazabilidad/README.md','Estado: **pasos 1–6 documentados para CNN-B; cinco corridas CPU EXP-002 y cinco\nCUDA EXP-003; informe final DOCX revisado; PDF por usuario y entrega global pendientes**.',
        'Estado: **pasos1–6 y ampliación CUDA CNN-B/FC-A/FC-B documentados y medidos; CPU/CNN-B históricas conservadas; DOCX actualizado manteniendo diseño del usuario. PDF por usuario y entrega global pendientes**.')
    append('docs/trazabilidad/README.md','## Ampliación vigente\n\n[FC-A yFC-B implementadas,cinco mediciones por red,capturas reales yentregables actualizados](ampliacion-fc.md). Sustituye el alcance inicial exclusivamenteCNN-B, sin invalidar ni sobrescribir sus evidencias.')
    replace('docs/trazabilidad/entrega.md','13páginas revisadas,con12tablas,7figuras y los seis apartados en orden.',f'{pages}páginas revisadas,con16tablas,13imágenes y los seis apartados en orden, más la ampliaciónFC.')
    append('docs/trazabilidad/entrega.md','[Ampliación FC completada y entregables vigentes](ampliacion-fc.md): dos notebooks separados, GPU con92celdas/30PNG/0errores. PDF no generado por instrucción del usuario.')
    replace('entregables/notebooks/README.md','31celdas de código ejecutadas,\n  15épocas nuevasCNN-B,1440actualizaciones,4imágenes y0errores.',
        '92celdas de código de sesiones nativas identificadas, CNN-B más cinco corridas FC-A y cinco FC-B;30imágenes y0errores. Conserva las53celdas anteriores, con150épocas FC añadidas.')
    replace('entregables/notebooks/README.md','89c14f3184c36f7b63a53016dc20427c6b131ccc9196849d5e36ee0163e39595',nb['notebook_cuda_sha256'])
    append('entregables/notebooks/README.md','La ampliación FC reutiliza la primera FC-A ya guardada y ejecuta solo nueve pendientes. No se reentrenó CPU/CNN-B. Los contadores reinician en cada sesión, explícitamente identificada; el cierre de presentación recuperado lee archivos reales sin repetir cálculos. [Procedimiento y validación actuales](../../docs/trazabilidad/ampliacion-fc.md).')
    append('resultados/README.md','## Ampliación CUDA FC-A/FC-B\n\n[Resultados completos](comparacion/ampliacion-fc/resultados.md), CSV ycuatrográficas, cinco observaciones por red. EXP-012 referencia FC-A1 guardada enEXP-011 y ejecuta nueve pendientes. EXP-010incompleto se conserva excluido. [Trazabilidad](../docs/trazabilidad/ampliacion-fc.md). CPU/CNN-B previas no se repitieron.')
    append('docs/trazabilidad/requisitos.md','## Actualización de evidencia: ampliación FC\n\nREQ-04/05/06/07/08/10/12 se amplían con FC-A/FC-B: código CUDA adicional, perfil/hilos/bloques/grids documentados, cinco mediciones por red, comparación CPU histórica, calidad y límites explícitos. REQ-09 añade dos capturas vivasGPU, sin certificar saturación literal. REQ-11 mantiene dos notebooks separados y DOCX actualizado;PDF por usuario pendiente. No cambia el estado de contribuciones humanas. [Evidencia yvalidación](ampliacion-fc.md).')
    print('Documentación actualizada;',pages,'páginas Word')
if __name__=='__main__':main()
