# Presentación visual del proyecto

CAM-013,DEC-028,BIT-015. CierreUTC:2026-11-18T22:13:05.400198+00:00. Responsable:Codex por solicitud del usuario.

Entregables:[PowerPoint](../../entregables/Presentacion_Proyecto_3_CUDA.pptx),[guion de exposición](../../entregables/presentacion/Guion_de_exposicion.md),[vista general](../../entregables/presentacion/Vista_general.png),[instrucciones](../../entregables/presentacion/README.md).

## Contenido y correspondencia con la guía

| Diapositivas | Cobertura |
|---|---|
|1–3 |Portada,equipo,ruta de seis pasos,dataset real yparticiones |
|4–5 |Paso1:flujo original,operaciones yrepeticiones |
|6–7 |Paso2:protocoloCPU,cinco observaciones ycuelloCNN-B |
|8 |Paso3:selección/exclusiones yjustificación por operación |
|9–11 |Paso4:flujoCPU/GPU,arquitecturas,threads/blocks/grids |
|12–16 |Paso5:stack,intervalo medido,pruebas ycapturas realesCPU/GPU/FC |
|17–24 |Paso6:tiempos,DE,speedup,cinco repeticiones,fases,desglose,curvas,accuracy ydiscusión técnica |
|25–26 |Trazabilidad yconclusiones de las tres redes |
|27–31 |Anexos:hardware/memoria,14fasesCPU,testCPU,recursos,referencias/glosario |

Se muestran13gráficas nativas editables ytablas de datos reales. Cada diapositiva tiene notas para el expositor. Se evita llenar las láminas con párrafos. Los promedios,DE yspeedup se recalcularon solo desdeCSVguardados y coinciden contolerancia1e-9;no se ejecutaron modelos.39fuentes originales identificadas porSHA-256. Se copian cinco capturas originales yseis imágenes delmanifiesto sin alterarlas.

## Procedimiento y decisiones de presentación

Se leyó entregables/SKILL-P.md completo yelPDFnormativo. Los archivos auxiliares indicados por esa habilidad no estaban instalados;se creó un flujo equivalente conpptxgenjs,validaciónOOXMLypython-pptx,renderrealconPowerPoint16local yPillowpara contacto. Se usa tema central,layouts,placeholders,cinco secciones ycoloresCPU/GPUconsistentes. Las matrices/curvas son gráficas nativas conhojas embebidas;capturas yfotografías se incorporan comoimágenes. SinPDFnuevo. Sinconsultasweb porque losvalores yfuentes técnicas ya están documentados enelproyecto.

Se comprobó quepptxgenjs noestaba disponible;se instaló v4.0.1 con19paquetes únicamente endata/presentacion-tooling. No se alteraron PCUDA ni modelos. Configuraciónexacta deNode archivada en losmanifiestos deherramientas. El scriptgenerador contiene la aplicación detema yla corrección de una referencia axId extra no declarada en gráficos2D emitida porlalibrería;no altera series niresultados.

Primera validación estructural detectó elaxId extra antes deabrir PowerPoint. Se corrigió en elgenerador. Primera revisión completa(revision-02) renderizó31láminas ydetectóuntexto desbordado en7;inspección visual detectó solapamiento de tabla/estadísticas en18. Se amplió la primera caja yse separaron filas/estadísticas en18. Revision-03validóyrenderizó31:sinobjetos fuera,sindesbordes,13gráficas,31notas. Seinspeccionaron las cuatro hojas decontacto y en tamaño completo7,18y23. No se modificó elinformePDF/DOCX actualdelusuario ni losnotebooks:hashes previos yposteriores iguales.

## Comandos realizados desde raíz

```powershell
node -e "try{console.log(require.resolve('pptxgenjs'))}catch(e){console.log('NO_PPTXGENJS')}"
npm.cmd install --prefix data/presentacion-tooling pptxgenjs
.\PCUDA\Scripts\python.exe -X utf8 scripts/preparar_presentacion.py
node scripts/crear_presentacion_cuda.cjs
python -X utf8 scripts/revisar_presentacion_cuda.py --revision revision-03
.\PCUDA\Scripts\python.exe -X utf8 scripts/finalizar_presentacion_cuda.py
```

Generación yrevisiones previas conservadas comoactividad,con resultados de revisión-02 yrevisión-03. Las primeras31láminas de cada revisión se renderizan solo paraQA,no se agregan experimentos ni se reabren notebooks para calcular. [Validaciónfinal](../../resultados/preparacion/presentacion-v01/revision-03/validacion.json),[textoextraído](../../resultados/preparacion/presentacion-v01/revision-03/texto-diapositivas.md),[identidadesprevias](../../resultados/preparacion/presentacion-v01/identidades-previas.json),[fuentes](../../entregables/presentacion/fuentes.csv).

La presentaciónes unresumen,distingue entrenamiento de test/preparación y noafirma speedup delpipeline adaptativo. ReferenciaCPUhistórica,RNGdistinto ycalidadFCinferior quedan visibles. ActividadGPUno se renombraocupación porSM. Lasnotas indican que las capturas adicionalesn=1 no se mezclan con lascampañasn=5. No se declara cumplimiento de contribucionesindividuales ni saturaciónliteral pendiente.


Auditoría documental final:las cuatro copias históricas de Markdown se renombraron .md.bak sin cambiar sus bytes; sus enlaces son relativos a su ubicación original yno documentación activa. El primer control detectó esos enlaces de las copias. El manifiesto inicial artefactos.csv se conserva comoantecedente; artefactos-v02.csv registra elcierre actual ylas rutas de snapshots corregidas. Ninguna diapositiva odato experimental se cambió.
