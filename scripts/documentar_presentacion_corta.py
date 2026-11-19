"""Close the short deck with provenance and a preview; no experiment execution."""
from pathlib import Path
from datetime import datetime,timezone
import json,csv,hashlib
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'entregables/presentacion-corta'
QA=ROOT/'resultados/preparacion/presentacion-corta-v01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def append(p,t):
    with (ROOT/p).open('a',encoding='utf8') as f:f.write('\n\n'+t.strip()+'\n')
def main():
    v=json.loads((QA/'revision-01/validacion.json').read_text(encoding='utf8'));assert v['estado']=='aprobado'
    mapping=json.loads((OUT/'estructura-y-notas.json').read_text(encoding='utf8'))
    prior=json.loads((QA/'identidades-previas.json').read_text(encoding='utf8'))
    assert all(sha(ROOT/p)==m['sha256'] for p,m in prior.items())
    assert sha(ROOT/mapping['salida'])==v['pptx_sha256']
    canvas=Image.new('RGB',(1280,4*210),'#e6ecef');draw=ImageDraw.Draw(canvas)
    for i in range(15):
        im=Image.open(QA/'revision-01'/f'diapositiva-{i+1:02d}.png');im.thumbnail((312,176))
        x=(i%4)*320+4;y=(i//4)*210+25;canvas.paste(im,(x,y));draw.text((x,y-20),str(i+1).zfill(2),fill='#15222D')
    canvas.save(OUT/'Vista_general.png')
    (OUT/'README.md').write_text('''# Presentación corta:15diapositivas

[Abrir PowerPoint corto](../Presentacion_Proyecto_3_CUDA_Corta.pptx) · [Guion corto](Guion_corto.md) · [Vista general](Vista_general.png)

La presentación completa de31diapositivas se conserva íntegra. Esta es una segunda versión con el diseño del archivo actual delusuario,7gráficas y7tablas editables,notas ycapturas reales. No se volvieron a ejecutar modelos ni se modificaron informes/notebooks.

| Nueva | Contenido |
|---|---|
|1 |Portada yspeedup de las tres redes |
|2 |Datos reales yparticiones |
|3 |Paso1:operaciones yrepeticiones |
|4–5 |Paso2:protocoloCPU ycuello de rendimiento |
|6 |Paso3:selección de operaciones |
|7–8 |Paso4:CPU/GPU yjerarquíaCUDA |
|9 |Paso5:implementación yfrontera de medición |
|10–11 |Evidencia realCPU yGPU |
|12–14 |Paso6:tiempos/speedup,desglose ycalidad |
|15 |Conclusión |

La versión larga sirve comorespaldo para arquitecturas completas,repeticiones individuales,curvas,pruebas ydetalles dehardware. El guion corto adapta su referencia final a esa versión completa;lasnotas originales de cada diapositiva se preservan enelPPTX. La renumeración1–15 se realiza mediantePowerPoint.

Se creó unacopia conPowerPoint, se eliminaron únicamente las16láminas excluidas yse guardó enotrarchivo. Textos,tablas,gráficas yestilos de las láminas conservadas no se reescribieron. La validación abrióyrenderizó las15láminas,comprobórelaciones,ejes,colores,notas ylímite de objetos;sin desbordes. SHA-256confirmaque la presentación larga,PDF,DOCXynotebooks permanecen idénticos. No seexportóPDF.
''',encoding='utf8')
    now=datetime.now(timezone.utc).isoformat()
    doc='''# Segunda presentación: versión corta

DEC-029,CAM-014,BIT-016. Se creó [Presentacion_Proyecto_3_CUDA_Corta.pptx](../../entregables/Presentacion_Proyecto_3_CUDA_Corta.pptx),con15diapositivas,7gráficas nativas,7tablas y15notas;[guion](../../entregables/presentacion-corta/Guion_corto.md) y[vista general](../../entregables/presentacion-corta/Vista_general.png).

Origen:archivo actual editado por elusuario,de31láminas. Se conservan las1,3,5,6,7,8,9,11,12,14,15,17,20,23,26. La selección mantiene los seis pasos,ambas evidencias principalesCPU/GPU ylas comparaciones de tiempo,desglose yexactitud. Los anexos ycurvas permanecen en laversión completa. No se entrenó ni se ejecutó ningún notebook.

El archivo completo se abrió enlectura yse usó SaveCopyAs para crearotrarchivo. Se cerró eloriginal;soloenlacopia se eliminaron16láminas. Se verificóconSHA-256que eloriginal ylosotrosentregables no cambiaron. PowerPoint conservó temas,layouts,notas,gráficas embebidas ytablas;renumerólos campos1–15. La comprobación textual inicial se ajustó para excluir únicamente elplaceholder de número de diapositiva,que debía cambiar;elcontenido restanteesidéntico aloriginal. La opción --finalizar permitiócontinuar la documentación sinregenerar/copiar otra vez elarchivo.

Se aplicaron las instruccioneslocales entregables/SKILL-P.md,continuando conPowerPointlocal yvalidaciónOOXML equivalentes a los auxiliares noincluidos. [Validación](../../resultados/preparacion/presentacion-corta-v01/revision-01/validacion.json):15láminasrenderizadas,sindesbordes,nirelacionesrotas,niejesindefinidos,nielementosfuera. Se inspeccionaron las doshojasdecontacto ylalámina5atamañocompleto. [Identidadesprevias](../../resultados/preparacion/presentacion-corta-v01/identidades-previas.json),[mapaynotas](../../entregables/presentacion-corta/estructura-y-notas.json).

Comandos ejecutados desde raíz:

```powershell
python -X utf8 scripts/crear_presentacion_corta.py
python -X utf8 scripts/crear_presentacion_corta.py --finalizar
python -X utf8 scripts/revisar_presentacion_corta.py
python -X utf8 scripts/documentar_presentacion_corta.py
```

Los ejecutores soloeditan/renderizan documentos;no hacenmediciones de rendimiento ni alteran experimentos. Mantienen las reservas deCPUhistórica,RNGdistinto,validaciónCPU yexactitudFCmenor. La versióncompletaactualdelusuario se conserva byteabyte;no se regenera desde unaversión anterior delgenerador.
'''
    (ROOT/'docs/trazabilidad/presentacion-corta.md').write_text(doc,encoding='utf8')
    append('docs/trazabilidad/bitacora.md',f'''## Cierre BIT-016 — {now}

Presentación corta final:15láminas,7gráficas,7tablas ynotas;PowerPointvalidadoyrenderizado sin desbordes. Se preservaexactamenteelPPTX31delusuario,SHA-256:{mapping['origen_sha256']}. TambiénPDF/DOCX/notebooks intactos. [Procedimiento](presentacion-corta.md).''')
    append('docs/trazabilidad/cambios.md','## Cierre CAM-014\n\nAñadida versión corta independiente;presentaciónlarga yotrosentregables intactos. [Detalle](presentacion-corta.md).')
    append('docs/trazabilidad/decisiones.md','## Cierre DEC-029\n\nSeleccionadas15láminasesenciales;seconservaelestiloactualdelusuario yla versióncompleta comorespaldo. [Validación](presentacion-corta.md).')
    append('docs/trazabilidad/README.md','## Presentación corta independiente\n\n[Versión de15diapositivas](presentacion-corta.md),sin modificarla de31. Incluye los seis pasos ylos resultadosprincipales.')
    append('README.md','[Presentación corta:15diapositivas](entregables/Presentacion_Proyecto_3_CUDA_Corta.pptx) · [Guion corto](entregables/presentacion-corta/Guion_corto.md). La versión de31se conserva sin cambios.')
    paths={ROOT/mapping['salida'],ROOT/'docs/trazabilidad/presentacion-corta.md'}
    paths.update(p for p in OUT.rglob('*') if p.is_file())
    paths.update(p for p in QA.rglob('*') if p.is_file())
    paths.update((ROOT/'scripts').glob('*presentacion_corta.py'))
    seal=OUT/'artefactos.csv';paths.discard(seal)
    with seal.open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['ruta_relativa','bytes','sha256']);w.writeheader()
        for p in sorted(paths):w.writerow(dict(ruta_relativa=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    print('CERRADO:',len(paths),'artefactos. Original31intacto;versióncorta15validada.')
if __name__=='__main__':main()
