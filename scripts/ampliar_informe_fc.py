"""Add FC results to the user's styled DOCX; preserve every prior body element."""
from pathlib import Path
from copy import deepcopy
import json,csv,hashlib,zipfile,re
from lxml import etree
from docx import Document
from docx.text.paragraph import Paragraph
from docx.shared import Inches,Pt
from docx.enum.text import WD_BREAK
from docx.oxml import OxmlElement
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'resultados/preparacion/ampliacion-fc'
RESULTS=ROOT/'resultados/comparacion/ampliacion-fc'
TARGET=ROOT/'entregables/Informe_Proyecto_3_CUDA.docx'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return list(csv.DictReader(p.open(encoding='utf8',newline='')))

def main():
    previous=BASE/'Informe_Proyecto_3_CUDA.docx'
    if sha(previous)!=sha(TARGET):
        last=json.loads((BASE/'integracion-informe.json').read_text(encoding='utf8'))
        assert sha(TARGET)==last['despues_sha256'],'DOCX edited since backup: preserve latest before merging'
        if not (BASE/'Informe_ampliado_maqueta01.docx').exists():(BASE/'Informe_ampliado_maqueta01.docx').write_bytes(TARGET.read_bytes())
    doc=Document(previous)
    tables=list(doc.tables)
    old=Document(previous)
    original_elements=[etree.tostring(e,method='c14n') for e in old._element.body if e.tag.rsplit('}',1)[-1]!='sectPr']
    paragraphs=doc.paragraphs
    body=next(p for p in paragraphs if p.text.startswith('El programa clasifica'))
    heading=next(p for p in paragraphs if p.text.startswith('6.3 Calidad'))
    caption=next(p for p in paragraphs if p.style.name=='Caption' and p.text)
    imagepara=next(p for p in paragraphs if p._p.xpath('.//w:drawing'))
    def clonep(template,text):
        text=re.sub(r'(?<=\d)(?=[a-záéíóúñ])',' ',text)
        text=re.sub(r'(?<=[a-záéíóúñ])(?=\d)',' ',text)
        for a,b in [('CPU128','CPU 128'),('CPU32','CPU 32'),('4GiB','4 GiB'),('MiB',' MiB')]:text=text.replace(a,b)
        node=deepcopy(template._p)
        for child in list(node):
            if child.tag.rsplit('}',1)[-1]!='pPr':node.remove(child)
        p=Paragraph(node,doc._body);run=p.add_run(text)
        if template.runs and template.runs[0]._r.rPr is not None:run._r.insert(0,deepcopy(template.runs[0]._r.rPr))
        return p
    def para(anchor,text,template=body):
        p=clonep(template,text);anchor.addprevious(p._p);return p
    def title(anchor,text):
        p=para(anchor,text,heading)
        if text.startswith(('5.3 ','5.4 ','5.5 ','6.4 ','6.5 ','6.6 ')):
            p.paragraph_format.page_break_before=True
        return p
    def pagebreak(anchor):
        # New subsection headings own their page break; no empty break paragraph.
        pass
    def figure(anchor,path,label,width=6.5):
        p=clonep(imagepara,'');p.add_run().add_picture(str(path),width=Inches(width));anchor.addprevious(p._p)
        para(anchor,label,caption)
    def replace_cell(cell,text):
        template=cell.paragraphs[0];p=clonep(template,str(text))
        for child in list(cell._tc):
            if child.tag.rsplit('}',1)[-1]!='tcPr':cell._tc.remove(child)
        cell._tc.append(p._p)
    def table(anchor,headers,rows,template):
        element=deepcopy(template._tbl)
        from docx.table import Table
        t=Table(element,doc._body)
        headerrow=deepcopy(t.rows[0]._tr);datarow=deepcopy(t.rows[1]._tr)
        for row in list(t.rows):element.remove(row._tr)
        element.append(headerrow)
        for cell,value in zip(t.rows[0].cells,headers):replace_cell(cell,value)
        for values in rows:
            element.append(deepcopy(datarow))
            for cell,value in zip(t.rows[-1].cells,values):replace_cell(cell,value)
        anchor.addprevious(element)
    anchor4=next(p._p for p in paragraphs if p.text.startswith('4. Diseño'))
    title(anchor4,'Ampliación de oportunidades: FC-A y FC-B')
    para(anchor4,'Además de CNN-B, se seleccionaron las dos redes totalmente conectadas del laboratorio. Sus productos matriciales, activaciones ReLU, normalización, Dropout, softmax, cálculo de gradientes y Adam se ejecutan en CUDA. La dependencia entre capas y actualizaciones se conserva; el paralelismo ocurre dentro de las matrices, características y elementos de cada lote. La preparación de archivos y la validación permanecen en CPU.')
    anchor5=next(p._p for p in paragraphs if p.text.startswith('5. Implementación'))
    title(anchor5,'Extensión del diseño a las redes totalmente conectadas')
    para(anchor5,'FC-A: Flatten → BatchNormalization → Dense128/ReLU → Dropout0.3 → Dense64/ReLU → Dropout0.3 → Dense29/softmax. FC-B: Flatten → BatchNormalization → Dense512/ReLU → Dropout0.4 → Dense256/ReLU → Dropout0.4 → Dense128/ReLU → Dropout0.4 → Dense29/softmax. Las imágenes se aplanan a12,288valores. Se conserva BatchNormalization con momentum0.99 y epsilon0.001, distintos del momentum0.9 de CNN-B.')
    table(anchor5,['Operación FC','Block','Grid / asignación'],[
        ['Matrices Dense y gradientes','Interno de cuBLAS','SGEMM FP32 PEDANTIC; jerarquía interna gestionada por NVIDIA.'],
        ['BN parcial de entrada','(32,8,1)','(1,384,1) para128 o20 muestras y12,288características.'],
        ['Cierre BN por característica','(256,1,1)','(48,1,1); momentum de la capa original; sin compartir estado entre características.'],
        ['ReLU,Dropout,BN yAdam','(256,1,1)','ceil(E/256), máximo4096bloques; recorrido grid-stride.'],
        ['Softmax/pérdida','(32,1,1)','Un bloque por muestra:128 o20bloques.']
    ],tables[7])
    para(anchor5,'FC-A tiene1,607,709parámetros entrenables y usa49.46MiB de buffers CUDA propios; FC-B tiene6,484,509 y usa124.65MiB. El perfil FC elimina el workspace im2col de convoluciones. Pesos, gradientes y estados Adam permanecen en GPU; la validación CPU recibe los pesos mediante D2H cada época. Estas cifras de buffers propios no representan la VRAM total del proceso/contexto.')
    data=json.loads((RESULTS/'resumen.json').read_text(encoding='utf8'))
    comparison=data['comparacion'];quality=data['calidad'];config=json.loads((ROOT/'configuraciones/EXP-012.json').read_text(encoding='utf8'))
    firsts={}
    for name in ['FC-A','FC-B']:
        item=next(x for x in config['runs'] if x['modelo']==name and x['repeticion']==1)
        firsts[name]=ROOT/item.get('ruta',f"resultados/paralelo/EXP-012/{item['run_id']}")
    anchor6=next(p._p for p in paragraphs if p.text=='6. Análisis y conclusiones')
    pagebreak(anchor6)
    title(anchor6,'5.3 Implementación y medición adicional de FC-A y FC-B')
    para(anchor6,'Se realizaron cinco entrenamientos nuevos por cada arquitectura FC:15épocas,96lotes por época,1440actualizaciones, batch128 y último lote20. Se reutilizaron los mismos17,400archivos, particiones, tensores y estados iniciales completos de la referencia CPU, verificados por SHA-256. No se repitieron las campañas CPU ni CNN-B. El entorno PCUDA y las dependencias se conservaron sin instalaciones adicionales.')
    para(anchor6,'El calentamiento de cada corrida comprendió lotes128+20 en GPU y evaluación CPU con lotes128/32; después se verificó la restauración de pesos, normalización, Adam y contador. Se midió tiempo pared sincronizado, incluyendo preparación del lote, máscaras, transferencias H2D/D2H y validación CPU en cada época. La preparación de imágenes, inicialización/compilación, guardado de modelos y elaboración de gráficas quedaron fuera del intervalo de entrenamiento. El test final CPU se midió por separado.')
    para(anchor6,'Las pruebas nuevas verificaron forward, gradientes, estado BatchNormalization, Adam y restauración para lotes20 y128 de ambas FC. La mayor diferencia absoluta de probabilidades frente al oráculo CPU fue1.67×10⁻⁶ y de la primera actualización Adam6.32×10⁻⁶. Son comprobaciones numéricas locales; la aleatoriedad interna de Dropout/shuffle sigue siendo distinta a la CPU histórica. Un intento incompleto por exportación quedó excluido; la corrida ya guardada que solo falló al presentar una tabla se recuperó sin reentrenarla.')
    table(anchor6,['Red / RUN','Entrenamiento (s)','Test CPU (s)','Test accuracy','Adam pasos'],[
        [m['modelo'][-1]+str(m['repeticion']),f"{json.loads((ROOT/m.get('ruta',f'resultados/paralelo/EXP-012/{m["run_id"]}')/'metricas.json').read_text(encoding='utf8'))['entrenamiento_s']:.4f}",
         f"{json.loads((ROOT/m.get('ruta',f'resultados/paralelo/EXP-012/{m["run_id"]}')/'metricas.json').read_text(encoding='utf8'))['evaluacion_cpu_s']:.4f}",
         f"{json.loads((ROOT/m.get('ruta',f'resultados/paralelo/EXP-012/{m["run_id"]}')/'metricas.json').read_text(encoding='utf8'))['test']['accuracy']*100:.3f}%",'1440'] for m in config['runs']
    ],tables[8])
    para(anchor6,'A1–A5 corresponden a FC-A; B1–B5 a FC-B. Son cinco repeticiones por arquitectura.')
    for i,name in enumerate(['FC-A','FC-B']):
        pagebreak(anchor6);title(anchor6,'5.'+str(4+i)+' Evidencia real durante entrenamiento '+name)
        run=firsts[name];picture=next(run.glob('captura-*.png'))
        meta=json.loads(picture.with_suffix('.json').read_text(encoding='utf8'));sample=meta['muestra_original']
        para(anchor6,f"Captura original de la ventana de monitor en vivo: {sample['utc']}, época{sample['epoca']}, actividad GPU{float(sample['gpu_util_pct']):.0f}%, VRAM{float(sample['gpu_mem_used_mib']):.0f}/4096MiB y CPU del proceso e hijos{float(sample['cpu_procesos_pct']):.1f}%. El monitor lee psutil y nvidia-smi y muestra los12procesadores lógicos. Actividad agregada no equivale a ocupación por SM ni a porcentaje de núcleos activos.")
        figure(anchor6,picture,f'Figura {8+i}. Captura real de recursos durante el entrenamiento CUDA de {name}.',6.4)
        epochs=read(run/'epocas.csv');last=epochs[-1]
        para(anchor6,f"Registro real de cierre: {name}, época15, pasos acumulados{last['pasos_acumulados']}, train_loss={float(last['train_loss']):.6f}, train_accuracy={float(last['train_accuracy']):.6f}, val_loss={float(last['val_loss']):.6f}, val_accuracy={float(last['val_accuracy']):.6f}. Checkpoint guardado con1440pasos Adam.")
        para(anchor6,'Las cinco corridas conservan muestras de recursos por segundo, logs de nvidia-smi,15registros de época, tablas de tiempos, transferencias, configuración y hashes. La captura se tomó durante la primera repetición y su coste no se descontó. Los notebooks incluyen las salidas nativas y las imágenes. No se afirma saturación total de VRAM ni ocupación por SM.')
    boundary=Paragraph(OxmlElement('w:p'),doc._body)
    boundary.paragraph_format.page_break_before=True
    boundary.paragraph_format.space_before=Pt(0);boundary.paragraph_format.space_after=Pt(0)
    boundary.paragraph_format.line_spacing=1;boundary.add_run().font.size=Pt(1)
    anchor6.addprevious(boundary._p)
    ref=next(p._p for p in paragraphs if p.text=='Referencias')
    # Insert BEFORE the user's original reference-page break. Inserting after
    # it and then breaking again would create an empty page. Preserve that old
    # paragraph unchanged so it still separates the final content/references.
    prior=ref.getprevious()
    for _ in range(3):
        if prior is None:break
        if any(e.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}type')=='page' for e in prior.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}br')):
            ref=prior;break
        prior=prior.getprevious()
    pagebreak(ref);title(ref,'6.4 Comparación ampliada: tres redes con CUDA')
    para(ref,'La extensión incorpora FC-A y FC-B al entrenamiento CUDA ya medido para CNN-B. Cada media se calcula con cinco observaciones. Las cifras previas de CNN-B permanecen vigentes; las nuevas observaciones FC se comparan exclusivamente con sus respectivas fases CPU históricas. DE indica desviación estándar muestral. Speedup=mediaCPU/mediaCUDA; reducción=(1−mediaCUDA/mediaCPU)×100.')
    table(ref,['Fase','CPU media ± DE(s)','CUDA media ± DE(s)','Speedup','Reducción'],[
        [r['fase'].replace('entrenamiento_','Fit ').replace('evaluacion_','Test CPU '),
         f"{r['cpu_media_s']:.3f} ± {r['cpu_de_s']:.3f}",f"{r['cuda_media_s']:.3f} ± {r['cuda_de_s']:.3f}",f"{r['speedup']:.3f}×",f"{r['reduccion_pct']:.2f}%"] for r in comparison
    ],tables[10])
    para(ref,'La fase test sigue ejecutándose en CPU después de exportar los pesos: su cociente de tiempos no mide aceleración de inferencia GPU. Se incluyen sus tiempos por transparencia y no se atribuye esa diferencia a CUDA.')
    figure(ref,RESULTS/'01_fc_tiempos.png','Figura10. Tiempos FC: barras media±DE y puntos de cada corrida; validación y transferencias incluidas.')
    pagebreak(ref);title(ref,'6.5 Interpretación del rendimiento y calidad de las FC')
    figure(ref,RESULTS/'02_tres_redes_speedup.png','Figura11. Aceleración del entrenamiento de CNN-B, FC-A y FC-B; cociente entre medias de cinco corridas.')
    fit=[r for r in comparison if r['fase'].startswith('entrenamiento_')]
    cpu=sum(r['cpu_media_s'] for r in fit);gpu=sum(r['cuda_media_s'] for r in fit)
    para(ref,f'La suma de las tres fases de entrenamiento seleccionadas es{cpu:.3f}s en CPU y{gpu:.3f}s con CUDA: cociente agregado{cpu/gpu:.3f}× y reducción{100*(1-gpu/cpu):.2f}%. Es una suma derivada de medias compatibles de campañas distintas, no una medición continua del pipeline original completo. Las fases no modificadas, como otras arquitecturas, augmentation, PCA y Random Forest, conservan sus mediciones y su ejecución en CPU.')
    figure(ref,RESULTS/'03_fc_desglose.png','Figura12. Desglose anidado del tiempo FC: intervalos de stream, transferencias, validación y resto host.')
    para(ref,'SGEMM concentra muchos productos y sumas independientes en GPU. Las matrices FC evitan im2col y no requieren las convoluciones de CNN-B. El tiempo total aún incluye lanzamientos y sincronización por lote, transferencia de parámetros y validación CPU. Por ello la actividad agregada puede ser menor que en CNN-B sin invalidar la reducción de tiempo. Los eventos delimitan intervalos del stream y pueden incluir huecos del host; no son mediciones de actividad exclusiva por kernel.')
    pagebreak(ref);title(ref,'6.6 Calidad y cierre de la ampliación')
    table(ref,['Modelo','Accuracy CPU','Accuracy CUDA','Loss CPU','Loss CUDA'],[
        [name,f"{next(r for r in quality if r['modelo']==name and r['origen']=='CPU')['accuracy_media_pct']:.3f}%",
         f"{next(r for r in quality if r['modelo']==name and r['origen']=='CUDA')['accuracy_media_pct']:.3f}%",
         f"{next(r for r in quality if r['modelo']==name and r['origen']=='CPU')['loss_media']:.5f}",
         f"{next(r for r in quality if r['modelo']==name and r['origen']=='CUDA')['loss_media']:.5f}"] for name in ['FC-A','FC-B']
    ],tables[10])
    figure(ref,RESULTS/'04_fc_calidad.png','Figura13. Evolución de validación CPU/GPU en las dos redes FC: cinco trayectorias y media por época.')
    para(ref,'Las FC CUDA obtuvieron menor exactitud de test que sus referencias CPU; esta ampliación no demuestra una mejora predictiva. Se reporta sin equipararla con speedup: arquitectura, datos, inicialización y número de actualizaciones coinciden, pero el orden interno de lotes, máscaras Dropout y redondeo no son idénticos a la CPU histórica. Las pruebas numéricas locales no certifican igualdad de toda la trayectoria. Las cinco repeticiones GPU reinician el mismo estado y RNG, por lo que evalúan repetibilidad temporal; no cinco inicializaciones independientes del aprendizaje.')
    para(ref,'El alcance final implementado comprende el entrenamiento completo de tres redes: CNN-B, FC-A y FC-B. La preparación de archivos/imágenes se mantiene en CPU con múltiples workers, mientras las operaciones matemáticas y optimización se ejecutan con CUDA Python y cuBLAS. La validación/test se declara en CPU. Esta distribución reduce el tiempo observado de las fases seleccionadas y conserva evidencia real de dispositivos, transferencias y resultados, sin afirmar optimización de todo el laboratorio ni saturación total del hardware.')
    # Preserve original element XML and all style definitions, headers and footers.
    destination=BASE/'Informe_Proyecto_3_CUDA_ampliado.docx';doc.save(destination)
    after=Document(destination)
    actual=[etree.tostring(e,method='c14n') for e in after._element.body]
    index=0
    for element in actual:
        if index<len(original_elements) and element==original_elements[index]:index+=1
    assert index==len(original_elements),('Original body changed',index,len(original_elements))
    # python-docx normalizes some XML even when no style edits were requested.
    # Restore the original protected package parts byte-for-byte: additions only
    # reference pre-existing styles and do not require any new style definition.
    with zipfile.ZipFile(previous) as a,zipfile.ZipFile(destination) as b:
        protected=[p for p in a.namelist() if p=='word/styles.xml' or p.startswith('word/header') or p.startswith('word/footer') or p.startswith('word/theme/')]
        normalized=[p for p in protected if a.read(p)!=b.read(p)]
        payload=[(info,a.read(info.filename) if info.filename in protected else b.read(info.filename)) for info in b.infolist()]
    temporary=destination.with_suffix('.tmp.docx')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for info,content in payload:archive.writestr(info,content)
    temporary.replace(destination)
    with zipfile.ZipFile(previous) as a,zipfile.ZipFile(destination) as b:
        assert all(a.read(p)==b.read(p) for p in protected),'User styles/header/footer changed'
    assert etree.tostring(old.sections[0]._sectPr)==etree.tostring(after.sections[0]._sectPr)
    TARGET.write_bytes(destination.read_bytes())
    report=dict(antes_sha256=sha(previous),despues_sha256=sha(TARGET),elementos_previos_preservados=index,
        estilos_headers_footers_temas_preservados=protected,partes_normalizadas_restauradas=normalized,tablas_antes=len(old.tables),tablas_despues=len(after.tables),
        imagenes_antes=len(old.inline_shapes),imagenes_despues=len(after.inline_shapes),pdf_generado=False)
    (BASE/'integracion-informe.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8');print(report)

if __name__=='__main__':main()
