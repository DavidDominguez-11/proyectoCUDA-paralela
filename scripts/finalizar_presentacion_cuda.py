"""Document and seal the finished presentation; existing experiments stay untouched."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,csv,shutil,statistics
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'entregables/presentacion'
QA=ROOT/'resultados/preparacion/presentacion-v01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def append(relative,text):
    with (ROOT/relative).open('a',encoding='utf8') as f:f.write('\n\n'+text.strip()+'\n')
def main():
    validation=json.loads((QA/'revision-03/validacion.json').read_text(encoding='utf8'))
    assert validation['estado']=='aprobado'
    assert sha(ROOT/'entregables/Presentacion_Proyecto_3_CUDA.pptx')==validation['pptx_sha256']
    data=json.loads((OUT/'datos-presentacion.json').read_text(encoding='utf8'))
    for row in data['compare']:
        values={origin:[float(x['duracion_s']) for x in data['observations'] if x['modelo']==row['modelo'] and x['origen']==origin and x['fase']==row['fase']] for origin in ['CPU','CUDA']}
        for origin,key,sd in [('CPU','cpu_media_s','cpu_de_s'),('CUDA','cuda_media_s','cuda_de_s')]:
            assert len(values[origin])==5
            assert abs(statistics.mean(values[origin])-float(row[key]))<1e-9
            assert abs(statistics.stdev(values[origin])-float(row[sd]))<1e-9
        assert abs(float(row['speedup'])-float(row['cpu_media_s'])/float(row['cuda_media_s']))<1e-12
    assert abs(statistics.mean(data['commonruns'])-float(data['common']['media_s']))<1e-8
    for row in csv.DictReader((OUT/'fuentes.csv').open(encoding='utf8',newline='')):
        assert sha(ROOT/row['ruta_relativa'])==row['sha256']
    initial=json.loads((QA/'identidades-previas.json').read_text(encoding='utf8'))
    assert all(sha(ROOT/p)==m['sha256'] for p,m in initial.items())
    # Readable contact sheet, derived exclusively from the final PowerPoint render.
    canvas=Image.new('RGB',(1280,8*210),'#e6ecef');draw=ImageDraw.Draw(canvas)
    for i in range(31):
        p=QA/'revision-03'/f'diapositiva-{i+1:02d}.png'
        im=Image.open(p);im.thumbnail((312,176));x=(i%4)*320+4;y=(i//4)*210+25
        canvas.paste(im,(x,y));draw.text((x,y-20),str(i+1).zfill(2),fill='#15222D')
    canvas.save(OUT/'Vista_general.png')
    for name in ['package.json','package-lock.json']:
        shutil.copy2(ROOT/'data/presentacion-tooling'/name,OUT/('tooling-'+name))
    (OUT/'README.md').write_text('''# Presentación del Proyecto3CUDA

[Abrir PowerPoint editable](../Presentacion_Proyecto_3_CUDA.pptx) · [Guion para exponer](Guion_de_exposicion.md) · [Vista general](Vista_general.png)

31diapositivas en16:9:26principales y5anexos. Diagramas del flujo original yCPU/GPU,operaciones repetitivas,referenciaCPU,selección,diseñoCUDA,implementación,pruebas,capturas reales,tiempos,speedup,variabilidad,curvas,calidad,limitaciones yconclusiones. Tablas y13gráficas nativas editables,31notas del presentador. Las capturas se copian sin alterar y las muestrasASL pertenecen a los archivos reales del manifiesto. No contiene imágenes generadas ni benchmarks nuevos.

## Uso durante la exposición

- Exponer las diapositivas1–26 y conservar27–31 para preguntas.
- Usar Vista del moderador para leer las notas; el guionMarkdown contiene la misma explicación.
- Resultados principales:17–23. Evidencia de dispositivos:14–16. Conclusión:26.
- En17 las escalas son diferentes por arquitectura; comparar CPU/CUDA dentro del panel.
- En18 cada columna es la secuencia de cinco repeticiones de su campaña; una misma fila no significa par CPU/GPU simultáneo.
- En20 se muestran proporciones normalizadas y una tabla en segundos; no sumar los componentes otra vez al total.
- En23 separar la rapidez de la calidad: lasFC bajaron1.34/4.33puntos porcentuales frente aCPUhistórica.

## Fuentes y reproducción

Datos usados en datos-presentacion.json;39archivos originales conSHA-256 en fuentes.csv. Guion/notas en estructura-y-notas.json y Guion_de_exposicion.md. Generador:scripts/crear_presentacion_cuda.cjs,preparación:scripts/preparar_presentacion.py. Las dependenciasNode se añadieron únicamente a data/presentacion-tooling;PCUDA permanece intacto. Se adjuntan manifiestospackage.json/package-lock para fijar las versiones usadas.

El PPTX incorpora las imágenes y hojas de datos; puede abrirse sin ejecutar ningún notebook ni disponer deldataset. Para reconstruir desde este repositorio ya preparado:

```powershell
node scripts/crear_presentacion_cuda.cjs
python -X utf8 scripts/revisar_presentacion_cuda.py --revision revision-NUEVA
```

El generador recompone elPPTX:conservar una copia antes si se hacen ediciones manuales. El preparador inicial rechaza destinos existentes. La revisión no exportaPDF:abre conPowerPointlocalenlectura,sacaPNG yverificaarchivo,relaciones,ejes,colores,límites ynotas. Los auxiliares sugeridos porSKILL-P.md no estaban incluidos;se realizaron controles equivalentes locales. El generador corrige una referencia de eje serie no declarada quepptxgenjs4.0.1 emite en gráficos2D. No se modifican los datos al hacerlo.

## Reservas importantes

CPUhistórica,nocinco pares alternados;shuffle/Dropout distintos,misma arquitectura/datos/estado inicial/pasos. Preparación1→6workers esCPU;testcontinúaCPU. ActividadGPUnoequivaleaocupaciónSMnisaturaciónVRAM. “Secuencial”es la referencia del proyecto,conparalelismo internoCPUhabilitado. El contenido diferenciamediciones,cálculosderivados ehipótesis. Se conservan los documentos ynotebooks previos sinningúncambio.
''',encoding='utf8')
    (OUT/'Vista_previa.md').write_text('# Vista general de las31diapositivas\n\n[PresentaciónPowerPoint](../Presentacion_Proyecto_3_CUDA.pptx) · [Guion](Guion_de_exposicion.md)\n\n![Vista de la presentación](Vista_general.png)\n',encoding='utf8')
    now=datetime.now(timezone.utc).isoformat()
    doc=f'''# Presentación visual del proyecto

CAM-013,DEC-028,BIT-015. CierreUTC:{now}. Responsable:Codex por solicitud del usuario.

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
node -e "try{{console.log(require.resolve('pptxgenjs'))}}catch(e){{console.log('NO_PPTXGENJS')}}"
npm.cmd install --prefix data/presentacion-tooling pptxgenjs
.\\PCUDA\\Scripts\\python.exe -X utf8 scripts/preparar_presentacion.py
node scripts/crear_presentacion_cuda.cjs
python -X utf8 scripts/revisar_presentacion_cuda.py --revision revision-03
.\\PCUDA\\Scripts\\python.exe -X utf8 scripts/finalizar_presentacion_cuda.py
```

Generación yrevisiones previas conservadas comoactividad,con resultados de revisión-02 yrevisión-03. Las primeras31láminas de cada revisión se renderizan solo paraQA,no se agregan experimentos ni se reabren notebooks para calcular. [Validaciónfinal](../../resultados/preparacion/presentacion-v01/revision-03/validacion.json),[textoextraído](../../resultados/preparacion/presentacion-v01/revision-03/texto-diapositivas.md),[identidadesprevias](../../resultados/preparacion/presentacion-v01/identidades-previas.json),[fuentes](../../entregables/presentacion/fuentes.csv).

La presentaciónes unresumen,distingue entrenamiento de test/preparación y noafirma speedup delpipeline adaptativo. ReferenciaCPUhistórica,RNGdistinto ycalidadFCinferior quedan visibles. ActividadGPUno se renombraocupación porSM. Lasnotas indican que las capturas adicionalesn=1 no se mezclan con lascampañasn=5. No se declara cumplimiento de contribucionesindividuales ni saturaciónliteral pendiente.
'''
    (ROOT/'docs/trazabilidad/presentacion.md').write_text(doc,encoding='utf8')
    append('docs/trazabilidad/bitacora.md',f'''## Cierre BIT-015 — {now}

Presentación editable de31diapositivas(26principales+5anexos),13gráficas nativas,31notas yguionMD. Reutilizadas39fuentes,fotosASL ycinco capturas reales. Medias/DE/speedup contrastados;ningún modelo ejecutado. PowerPoint abrióyrenderizó todas,sin desbordes tras correcciones7/18. Documentos/notebooks previos conservadosporSHA-256. [Procedimiento,evidencia yentrega](presentacion.md).''')
    append('docs/trazabilidad/cambios.md','## Cierre CAM-013\n\nAñadidos PPTX,guion,fuentes,recursos,validación yvista previa. Informe actual delusuario,PDF ynotebooks inalterados. [Detalle](presentacion.md).')
    append('docs/trazabilidad/decisiones.md','## Cierre DEC-028\n\nPresentación visual creada:datos reales,notas separadas,gráficas/tablas editables,anexos yverificación conPowerPoint. ReferenciaCPUhistórica yreservas de calidad/recursos visibles. [Detalle](presentacion.md).')
    append('docs/trazabilidad/README.md','## Presentación visual\n\n[Presentación del proyecto yguion de exposición](presentacion.md):31diapositivas conresultados detresredes,capturas,diagramas y13gráficas editables;sin repetirentrenamientos ni modificar elinforme.')
    append('README.md','[Presentación PowerPoint del proyecto](entregables/Presentacion_Proyecto_3_CUDA.pptx) · [Guion para exponer](entregables/presentacion/Guion_de_exposicion.md) · [Vista general](entregables/presentacion/Vista_general.png).')
    append('entregables/README.md','## Presentación visual\n\n[PowerPoint editable](Presentacion_Proyecto_3_CUDA.pptx),31diapositivas;[guion](presentacion/Guion_de_exposicion.md) y[vista general](presentacion/Vista_general.png). Gráficas/tablas editables,capturas reales ynotas en todas las láminas.')
    paths={ROOT/'entregables/Presentacion_Proyecto_3_CUDA.pptx',ROOT/'scripts/preparar_presentacion.py',ROOT/'scripts/crear_presentacion_cuda.cjs',ROOT/'scripts/revisar_presentacion_cuda.py',Path(__file__).resolve(),ROOT/'docs/trazabilidad/presentacion.md'}
    paths.update(p for p in OUT.rglob('*') if p.is_file())
    paths.update(p for p in QA.rglob('*') if p.is_file())
    seal=OUT/'artefactos.csv';paths.discard(seal)
    with seal.open('x',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['ruta_relativa','bytes','sha256']);writer.writeheader()
        for p in sorted(paths):writer.writerow(dict(ruta_relativa=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    print('CIERRE APROBADO',len(paths),'artefactos;31diapositivas;sinentrenamientos;documentosprevios intactos.')
if __name__=='__main__':main()
