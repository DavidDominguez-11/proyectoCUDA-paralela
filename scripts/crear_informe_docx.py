"""Create the final editable course report from real evidence. Never export PDF."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path
import shutil
import sys

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]


def readable(text):
    """Normalize Spanish typography in narrative/table text, never in raw logs."""
    text = str(text)
    protected = {}
    def protect(match):
        key = '<IDENTIDAD' + chr(65+len(protected)) + '>'
        protected[key] = match.group()
        return key
    text = re.sub(r'RUN-\d{8}T\d{6}Z-(?:cpu|gpu)-\d{2}|\b(?:FP32|C\d{2}|G\d{2}|im2col|col2im|float32|int64)\b', protect, text)
    text = re.sub(r'([a-záéíóúñ])(?=\d)', r'\1 ', text)
    text = text.replace('float 32', 'float32').replace('int 64', 'int64').replace('im 2col', 'im2col').replace('col 2im', 'col2im')
    text = re.sub(r'(CPU|GPU|CUDA|FP32|BN|SM|DE|IC)(?=[a-záéíóúñ])', r'\1 ', text)
    text = re.sub(r'(?<=[a-záéíóúñ])(CPU|GPU|CUDA|CNN|BN|FP32|SM)', r' \1', text)
    text = re.sub(r'(CUDA|GTX|CPU|GPU|VRAM|BN|CNN-B)(?=\d)', r'\1 ', text)
    text = re.sub(r'(?<=\d)(?=[A-Za-zÁÉÍÓÚáéíóúñ])', ' ', text)
    text = re.sub(r'(\d) e([−+-]?\d)', r'\1e\2', text)
    text = re.sub(r'(?<=[a-záéíóúñ])(VRAM|Host|Adam|PCIe)', r' \1', text)
    text = re.sub(r'(?<=[%×])(?=[A-Za-záéíóúñ])', ' ', text)
    text = re.sub(r'([,;:])(?=[A-Za-zÁÉÍÓÚáéíóúñ])', r'\1 ', text)
    text = re.sub(r';(?=\S)', '; ', text)
    text = re.sub(r'([A-Za-záéíóúñ]):(?=[^/\s])', r'\1: ', text)
    text = re.sub(r'(?<=\S)\(', ' (', text)
    text = re.sub(r'(?<=aprox\.)(?=\d)', ' ', text)
    text = re.sub(r'(?<=\d) sde\b', ' s de', text)
    text = text.replace('ocupaciónpor', 'ocupación por').replace('globalbitabit', 'global bit a bit')
    for key, value in protected.items():
        text = text.replace(key, value)
    return text


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def rows(path):
    with path.open(encoding='utf8', newline='') as handle:
        return list(csv.DictReader(handle))


def load(path):
    return json.loads(path.read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destino', type=Path, required=True)
    args = parser.parse_args()
    out = args.destino.resolve()
    out.mkdir(parents=True, exist_ok=False)
    assets = out / 'figuras'
    assets.mkdir()
    sources = set()

    def source(relative):
        path = ROOT / relative
        sources.add(path)
        return path

    gpu = 'resultados/paralelo/EXP-003'
    cpu = 'resultados/secuencial/EXP-002'
    analysis = 'resultados/comparacion/paso-6-revision-02'
    for entry in rows(source(gpu + '/artefactos-experimento.csv')):
        path = ROOT / entry['ruta_relativa']
        if path.stat().st_size != int(entry['bytes']) or sha(path) != entry['sha256']:
            raise ValueError('Sealed evidence changed: ' + str(path))
    comparison = rows(source(analysis + '/comparacion_fases.csv'))
    stats = rows(source(analysis + '/estadisticas.csv'))
    obs = rows(source(analysis + '/observaciones.csv'))
    common = rows(source(cpu + '/tiempos_referencia_comun.csv'))
    csummary = rows(source(cpu + '/resumen.csv'))
    components = rows(source(analysis + '/desglose_entrenamiento.csv'))
    resources = rows(source(gpu + '/recursos_resumen.csv'))
    quality = rows(source(gpu + '/calidad_comparada.csv'))
    source('docs/Proyecto 3 CUDA.pdf')
    source('docs/trazabilidad/revision-pasos-1-5.md')
    source('docs/trazabilidad/analisis-original.md')
    source('docs/trazabilidad/diseno-paralelo.md')
    source('docs/trazabilidad/implementacion-cuda.md')
    source('configuraciones/EXP-002.json')
    source('configuraciones/EXP-003.json')

    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Inches(8.2677), Inches(11.6929)
    section.top_margin, section.bottom_margin = Inches(.65), Inches(.60)
    section.left_margin, section.right_margin = Inches(.73), Inches(.73)
    section.header_distance, section.footer_distance = Inches(.25), Inches(.25)
    section.different_first_page_header_footer = True
    styles = doc.styles
    normal = styles['Normal']
    normal.font.name, normal.font.size = 'Calibri', Pt(10.5)
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.03
    normal.paragraph_format.widow_control = True
    for name, size in [('Title', 27), ('Subtitle', 14), ('Heading 1', 17), ('Heading 2', 12), ('Heading 3', 11)]:
        style = styles[name]
        style.font.name, style.font.size = 'Calibri', Pt(size)
        style.font.color.rgb = RGBColor.from_string('173B57')
        style.paragraph_format.space_before = Pt(9)
        style.paragraph_format.space_after = Pt(6)
        style.paragraph_format.keep_with_next = True
    for name in ('Caption',):
        styles[name].font.name, styles[name].font.size = 'Calibri', Pt(9)
        styles[name].font.color.rgb = RGBColor.from_string('425563')
        styles[name].paragraph_format.space_after = Pt(6)
    styles['List Bullet'].paragraph_format.space_after = Pt(4)
    styles['List Bullet'].font.size = Pt(10.5)
    lang = OxmlElement('w:lang')
    lang.set(qn('w:val'), 'es-GT')
    normal.element.get_or_add_rPr().append(lang)
    header = section.header.paragraphs[0]
    header.text = 'CC3069  ·  PROYECTO 3                                      ASL / CUDA PYTHON'
    header.runs[0].font.size = Pt(8)
    header.runs[0].font.color.rgb = RGBColor.from_string('536778')
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer.add_run('Proyecto 3  |  ').font.size = Pt(8)
    field = OxmlElement('w:fldSimple')
    field.set(qn('w:instr'), 'PAGE')
    footer._p.append(field)
    doc.core_properties.title = 'Proyecto 3: Mejorando algoritmos con CUDA'
    doc.core_properties.subject = 'Informe de rendimiento y análisis: reconocimiento ASL con CNN-B'
    doc.core_properties.author = 'David Domínguez; Jonathan Diaz; Luis Padilla'
    doc.core_properties.keywords = 'CUDA Python, ASL, CNN, CPU, GPU, rendimiento, speedup'
    md = []
    table_number = figure_number = 0

    def p(text, style=None):
        text = readable(text)
        paragraph = doc.add_paragraph(text, style)
        md.append(text + '\n')
        return paragraph

    def h(text, level=1):
        text = readable(text)
        doc.add_heading(text, level=level)
        md.append('#' * (level+1) + ' ' + text + '\n')

    def page():
        doc.add_page_break()
        md.append('\n<!-- Salto de página -->\n')

    def bullet(text):
        p(text, 'List Bullet')

    def table(title, headers, data, widths=None, font=9):
        nonlocal table_number
        table_number += 1
        title = readable(title)
        headers = [readable(v) for v in headers]
        data = [[readable(v) for v in row] for row in data]
        cap = doc.add_paragraph(f'Tabla {table_number}. {title}', 'Caption')
        cap.paragraph_format.keep_with_next = True
        grid = doc.add_table(rows=1, cols=len(headers))
        grid.alignment = WD_TABLE_ALIGNMENT.CENTER
        grid.autofit = False
        if widths is None:
            widths = [6.80 / len(headers)] * len(headers)
        for cell, text in zip(grid.rows[0].cells, headers):
            cell.text = str(text)
        for data_row in data:
            for cell, text in zip(grid.add_row().cells, data_row):
                cell.text = str(text)
        for column, width in zip(grid.columns, widths):
            column.width = Inches(width)
        for i, row in enumerate(grid.rows):
            rowPr = row._tr.get_or_add_trPr()
            no_split = OxmlElement('w:cantSplit')
            rowPr.append(no_split)
            if i == 0:
                repeat = OxmlElement('w:tblHeader')
                rowPr.append(repeat)
            for cell, width in zip(row.cells, widths):
                cell.width = Inches(width)
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                props = cell._tc.get_or_add_tcPr()
                shade = OxmlElement('w:shd')
                shade.set(qn('w:fill'), '173B57' if i == 0 else ('EEF3F7' if i % 2 else 'FFFFFF'))
                props.append(shade)
                margins = OxmlElement('w:tcMar')
                for side, amount in [('top', 45), ('bottom', 45), ('left', 70), ('right', 70)]:
                    edge = OxmlElement('w:' + side)
                    edge.set(qn('w:w'), str(amount)); edge.set(qn('w:type'), 'dxa')
                    margins.append(edge)
                props.append(margins)
                for para in cell.paragraphs:
                    para.paragraph_format.space_after = Pt(1)
                    para.paragraph_format.line_spacing = 1
                    for run in para.runs:
                        run.font.size = Pt(font)
                        if i == 0:
                            run.bold = True
                            run.font.color.rgb = RGBColor(255, 255, 255)
        md.extend([f'**Tabla {table_number}. {title}**\n',
                   '| ' + ' | '.join(headers) + ' |',
                   '| ' + ' | '.join('---' for _ in headers) + ' |',
                   *['| ' + ' | '.join(str(v).replace('\n', '; ') for v in row) + ' |' for row in data], ''])
        doc.add_paragraph().paragraph_format.space_after = Pt(0)

    def image(relative, name, caption, width=6.6):
        nonlocal figure_number
        figure_number += 1
        caption = readable(caption)
        original = source(relative)
        target = assets / name
        shutil.copyfile(original, target)
        picture = doc.add_paragraph()
        picture.alignment = WD_ALIGN_PARAGRAPH.CENTER
        picture.paragraph_format.space_after = Pt(3)
        picture.paragraph_format.keep_with_next = True
        picture.add_run().add_picture(str(target), width=Inches(width))
        doc.add_paragraph(f'Figura {figure_number}. {caption}', 'Caption')
        md.append(f'![Figura {figure_number}. {caption}](figuras/{name})\n')

    def code(lines):
        para = doc.add_paragraph()
        para.paragraph_format.space_after = Pt(5)
        para.paragraph_format.line_spacing = 1
        shade = OxmlElement('w:shd'); shade.set(qn('w:fill'), 'EEF3F7')
        para._p.get_or_add_pPr().append(shade)
        run = para.add_run('\n'.join(lines))
        run.font.name, run.font.size = 'Consolas', Pt(8.2)
        md.append('```text\n' + '\n'.join(lines) + '\n```\n')

    # Page 1 — cover and self-contained executive summary.
    p('UNIVERSIDAD DEL VALLE DE GUATEMALA').runs[0].bold = True
    p('Facultad de Ingeniería · Departamento de Ciencias de la Computación')
    p('CC3069 · Computación Paralela y Distribuida · Semestre II, 2026')
    p('Proyecto 3\nMejorando algoritmos\ncon CUDA', 'Title')
    p('Informe de rendimiento y análisis\nReconocimiento de señas ASL con CNN-B', 'Subtitle')
    p('David Domínguez · 23712\nJonathan Diaz · 23873\nLuis Padilla · 23663')
    p('Guatemala, noviembre de 2026')
    h('Resumen', 2)
    p('Se analizó el laboratorio de reconocimiento ASL y se seleccionó CNN-B como principal '
      'oportunidad de aceleración. Se mantuvo la preparación de archivos en CPU y se implementó '
      'la matemática del entrenamiento con CUDA Python, kernels Numba-CUDA y cuBLAS. Se compararon '
      'cinco ejecuciones CPU y cinco GPU con las mismas 17,400 imágenes, arquitectura, pesos iniciales '
      'y 15 épocas, incluyendo transferencias y validación CPU en el tiempo de entrenamiento.')
    table('Resultado principal del entrenamiento CNN-B', ['CPU, media ± DE', 'Variante CUDA, media ± DE', 'Speedup', 'Reducción'],
          [['805.650 ± 2.723 s', '418.608 ± 1.916 s', '1.9246×', '48.04%']], [1.95, 2.25, 1.15, 1.45], 10)
    p('La comparación utiliza una referencia CPU histórica: el orden y las máscaras aleatorias no son '
      'idénticos entre backends. El resultado acredita una reducción local del tiempo de CNN-B, no del '
      'laboratorio completo. Las capturas demuestran uso real de CPU y GPU; no demuestran saturación '
      'total de memoria ni ocupación por multiprocesador. La exactitud de prueba GPU fue 98.62%.')
    p('Palabras clave: CUDA Python, ASL, CNN, CPU/GPU, transferencias, speedup.')

    # Page 2 — exact required section title, original operations and repetitions.
    page(); h('1. Análisis de la solución secuencial existente')
    p('El programa clasifica imágenes almacenadas de manos en 29 clases: A–Z, del, nothing y space. '
      'Sus partes principales son adquisición/exploración, preparación, entrenamiento, evaluación '
      'y selección, y predicción sobre fotografías propias. El laboratorio procesa archivos, no vídeo en vivo.')
    operations = [
        ('1', 'Configuración y recepción', 'Importar librerías, recuperar ASL Alphabet de caché e identificar 29 clases.'),
        ('2', 'Exploración e inventario', 'Contar 87,000 imágenes disponibles e inspeccionar formato y muestras.'),
        ('3', 'Muestreo y partición', 'Seleccionar 600 imágenes/clase; separar train/val/test 70/15/15 con semilla 42.'),
        ('4', 'Preparación', 'Abrir JPEG, convertir RGB, redimensionar a 64×64 y normalizar float32/255; codificar etiquetas.'),
        ('5', 'Persistencia de datos', 'Conservar arreglos preparados y el mapeo clase–índice; revisar transformación visual.'),
        ('6', 'Entrenamiento CNN', 'Construir CNN-A y CNN-B; ejecutar forward, pérdida, backward, Adam y validación.'),
        ('7', 'Diagnóstico CNN', 'Calcular accuracy/loss, predicciones, matriz de confusión y reporte de clasificación.'),
        ('8', 'Otras alternativas', 'Entrenar FC-A/FC-B; ajustar PCA de 150 componentes y dos Random Forest.'),
        ('9', 'Aumento de imágenes', 'Entrenar una nueva CNN seleccionada con rotación, zoom, traslación y contraste.'),
        ('10', 'Selección y guardado', 'Comparar métricas, elegir modelo final, guardar y recargar redes y estimadores.'),
        ('11', 'Fotografías propias', 'Preparar 15 imágenes del equipo y obtener etiquetas mediante predict y argmax.'),
        ('12', 'Presentación', 'Mostrar predicciones y aciertos; comparar modelos y discutir generalización.')]
    table('Descomposición del recorrido original', ['No.', 'Operación identificada', 'Descripción de su funcionamiento'], operations, [.35, 1.62, 4.83], 9)
    table('Operaciones repetitivas', ['Operación repetitiva', 'Datos que procesa', 'Cantidad de repeticiones'], [
        ('Leer/RGB/resize', 'Archivos seleccionados', '17,400 por carga completa, no por época.'),
        ('Normalizar y etiquetar', '12,288 valores por imagen', '213,811,200 valores y 17,400 etiquetas por carga.'),
        ('Forward/backward/Adam', 'Lotes, pesos y gradientes', '96 pasos/época; 95 lotes de128 y uno de20.'),
        ('Entrenar modelos', 'Train:12,180 imágenes', '5 fit originales; hasta15 épocas por modelo.'),
        ('Validar', 'Val:2,610 imágenes', 'Una pasada por época y por red.'),
        ('Capas convolucionales', 'Mapas y filtros3×3', '2 capas/lote CNN-A;5 capas/lote CNN-B.'),
        ('Ajustar PCA / bosques', 'Vectores y150 componentes', '1 ajustePCA;2 bosques,100+300 árboles.'),
        ('Inferir fotos propias', '15 fotografías,6 modelos', '15 predicciones finales y90 al comparar modelos.')], [1.65, 1.82, 3.33], 8.8)
    p('Los niveles anteriores están anidados y no se suman. En el historial original CNN-B terminó '
      'en 11 épocas por parada temprana; la medición controlada posterior usa 15 épocas fijas en '
      'ambas variantes. “Secuencial” designa la versión CPU: sus librerías pueden usar varios hilos.')

    # Page 3 — experimental CPU conditions and all repetitions.
    page(); h('2. Medición del tiempo de ejecución secuencial')
    h('Datos, entorno y condiciones controladas', 2)
    p('Se reutilizó ASL Alphabet, versión local 1, sin descargar otra vez (Akash, s. f.). De 87,000 '
      'imágenes se conservaron 17,400: 12,180 de entrenamiento, 2,610 de validación y 2,610 de prueba. '
      'No se detectaron duplicados exactos entre las particiones. SHA-256 de archivos, etiquetas y '
      'tensores permitió comprobar que la carga CPU/GPU coincidía. Esto no acredita independencia por persona o sesión.')
    table('Entorno utilizado', ['Componente', 'Configuración registrada'], [
        ('CPU y RAM', 'AMD Ryzen5 4600H;6 núcleos físicos/12 lógicos;15.42GiB RAM.'),
        ('Entorno', 'Windows;PCUDA;Python3.14.3;NumPy2.5.5;TensorFlow2.22.0rc0.'),
        ('Paralelismo CPU', 'TensorFlow intra-op6/inter-op1;BLAS6;RandomForest n_jobs=−1.'),
        ('Preparación base', 'Un worker;Pillow RGB/resize BICUBIC;64×64;float32/255.'),
        ('Entrenamiento base', '15 épocas fijas;batch128;sin EarlyStopping ni restauración de mejor época.')], [1.35, 5.45], 9.4)
    h('Procedimiento de medición', 2)
    p('Cada repetición se ejecutó en un proceso nuevo: verificar identidades, construir/calentar '
      'modelos, restaurar pesos y optimizador, cargar imágenes y medir las fases con perf_counter_ns. '
      'La validación dentro de fit está incluida. Se excluyeron imports, calentamiento, EDA, gráficas '
      'y escritura final de checkpoints. La caché del sistema no se vació y los archivos se verificaron '
      'antes de medir; no se trata de disco frío. Se conservaron todas las observaciones completas.')
    # Take the continuous pipeline directly from each original timing CSV.
    cpu_runs = load(source(cpu + '/serie.json'))['runs']
    cpu_pipeline = []
    for name in cpu_runs:
        original_rows = rows(source(cpu + '/' + name + '/tiempos.csv'))
        cpu_pipeline.append(float(next(r for r in original_rows if r['fase']=='pipeline_total')['duracion_s']))
    cpu_obs = [r for r in obs if r['dispositivo']=='cpu']
    table('Cinco repeticiones CPU: alcances distintos expresados en segundos',
          ['Repetición', 'Referencia común\n14 fases', 'CNN-B fit', 'Pipeline adaptativo'],
          [[f'C{i+1}', f"{float(common[i]['duracion_s']):.3f}", f"{float(cpu_obs[i]['entrenamiento_s']):.3f}", f'{cpu_pipeline[i]:.3f}'] for i in range(5)],
          [1.1, 2.0, 1.5, 2.2], 9.4)
    p('La referencia común promedió 1119.928±2.723s (media±DE). El pipeline adaptativo promedió '
      '1858.617±314.366s, pero cambió de CNN aumentada en C04: es un promedio descriptivo de ramas '
      'diferentes, no una base homogénea para speedup. CNN-B base se ejecutó en las cinco corridas '
      'y su fit de 805.650±2.723s es la referencia compatible empleada después.')

    # Page 4 — CPU workload decomposition and measured variability.
    page(); h('2.1 Resultados CPU por fase y factores de rendimiento', 2)
    groups = [('Preparación', ['preparacion_cpu']), ('CNN-A:fit + test', ['entrenamiento_CNN-A','evaluacion_CNN-A']),
              ('CNN-B:fit + test', ['entrenamiento_CNN-B','evaluacion_CNN-B']),
              ('FC-A:fit + test', ['entrenamiento_FC-A','evaluacion_FC-A']),
              ('FC-B:fit + test', ['entrenamiento_FC-B','evaluacion_FC-B']), ('PCA', ['pca']),
              ('RF-A:fit + validación', ['entrenamiento_RF-A','validacion_RF-A']),
              ('RF-B:fit + validación', ['entrenamiento_RF-B','validacion_RF-B'])]
    grouped = [[label, f"{sum(float(next(r for r in csummary if r['fase']==name)['media_s']) for name in names):.3f}"] for label,names in groups]
    table('Distribución de las 14 fases comunes CPU (n=5)', ['Grupo de fases', 'Suma de medias (s)'], grouped, [4.8,2], 9.2)
    image(cpu+'/graficas/referencia_comun_repeticiones.png', '01_referencia_cpu.png',
          'Referencia CPU común de14 fases en cinco procesos. El panel inferior amplía las diferencias respecto a la media; el ICt95% es nominal.', 5.15)
    p('CNN-B fit representa 71.94% de los1119.928s comunes. La variación común fue pequeña '
      '(CV0.243%;ICt95%[1116.546,1123.309]s); el total adaptativo tuvo CV16.91% por cambiar de rama. '
      'También pueden influir caché, hilos internos, cargas del sistema y estado térmico. No se '
      'aisló experimentalmente el efecto individual de esos factores.')

    # Page 5 — selection and dependencies.
    page(); h('3. Identificación de oportunidades de paralelización')
    p('Se priorizó la matemática del entrenamiento de CNN-B por su predominio temporal y el '
      'gran número de operaciones regulares sobre lotes y mapas de características. No se eligió '
      'la arquitectura por obtener mejor test en una nueva búsqueda; se fijó por prioridad computacional.')
    table('Operaciones seleccionadas y justificación', ['Operación', 'Simultaneidad aprovechable', 'Beneficio esperado'], [
        ('Convolución forward', 'Salidas por imagen,píxel y filtro.', 'Distribuir muchos productos y sumas.'),
        ('Gradientes convolucionales', 'Contribuciones de posiciones/lotes y filtros.', 'Acelerar el trabajo repetido del backward.'),
        ('Capas Dense', 'Elementos de productos matriciales.', 'Usar multiplicación matricial especializada.'),
        ('ReLU y Dropout', 'Transformación independiente por elemento.', 'Procesar activaciones/gradientes simultáneamente.'),
        ('MaxPooling', 'Ventanas y canales independientes.', 'Reducir ventanas en paralelo y conservar índice de máximo.'),
        ('BatchNormalization', 'Agregaciones por canal y aplicación por elemento.', 'Paralelizar estadísticas y transformación.'),
        ('Softmax y pérdida', 'Filas del lote y clases dentro de cada fila.', 'Reducir máximos/sumas y calcular gradiente estable.'),
        ('Adam', 'Actualización por parámetro una vez listo el gradiente.', 'Actualizar millones de parámetros con trabajo uniforme.')],
          [1.5,2.6,2.7], 9.5)
    h('Operaciones que permanecieron en CPU', 2)
    p('Apertura y decodificación de JPEG, conversión RGB, resize, normalización de entrada, '
      'etiquetado, organización de lotes, máscaras aleatorias, validación, test, métricas de informe '
      'y persistencia. La preparación permite concurrencia entre archivos sin alterar los tensores. '
      'PCA, Random Forest, otras redes y las fotografías propias no se trasladaron a GPU en este alcance.')
    h('Dependencias que limitan el paralelismo', 2)
    p('Las capas necesitan las activaciones anteriores. La pérdida precede al backward y Adam '
      'requiere los gradientes completos antes de actualizar pesos. El siguiente lote usa esos '
      'pesos nuevos; por ello no se entrenan épocas independientemente. BatchNormalization y '
      'gradientes necesitan reducciones: la independencia por muestra no elimina las agregaciones.')
    p('La oportunidad era una hipótesis de rendimiento en esta etapa. La selección no garantizaba '
      'un speedup mínimo ni que cada operación pequeña fuera más rápida en GPU de manera aislada.')

    # Page 6 — complete realized hierarchy and mathematical contract.
    page(); h('4. Diseño del algoritmo paralelo')
    p('El flujo diseñado separa CPU(preparación) → H2D → GPU(entrenamiento) → D2H → '
      'CPU(validación/test). Se usa un proceso, una GPU y un stream, sin solapamiento. Los estados '
      'del modelo permanecen en GPU durante entrenamiento; cada época exporta pesos/BN para validar en CPU.')
    p('Arquitectura: Conv32–ReLU–BN → Conv32–ReLU–Pool–Dropout0.25 → Conv64–ReLU–BN '
      '→ Conv64–ReLU–Pool–Dropout0.25 → Conv128–ReLU–BN–Pool–Dropout0.30 → Flatten8192 '
      '→ Dense256–ReLU–BN–Dropout0.40 → Dense29–Softmax. Hay2,245,245 parámetros '
      'entrenables y960 valores móvilesBN. El orden Conv→ReLU→BN se conserva.')
    table('Distribución efectiva de trabajo entre threads, blocks y grids', ['Operación', 'Block', 'Grid / asignación'], [
        ('im2col/col2im,activaciones,pooling,Dropout,Adam,control de finitud', '256 threads', 'min(4096,ceil(E/256));recorrido grid-stride.'),
        ('Reducción por canal BN/bias', '32×8=256', '(ceil(L/2048),ceil(C/32));canales contiguos por warp.'),
        ('Fusión de parciales por canal', '256 threads', 'C blocks;reducción de parciales y estadísticas.'),
        ('Softmax/pérdida', '32 threads', 'b blocks:una fila/imagen por block;29 clases útiles.'),
        ('Métricas / contador', '256 /1', 'Un block;agregación del lote / avance de paso.'),
        ('Productos grandes SGEMM', 'Backend NVIDIA', 'cuBLAS administra su jerarquía interna;no se atribuyen grids propios.')],
          [2.4,1.05,3.35], 9.2)
    p('E=número de elementos;L=filas reducidas;C=canales;b=128 o20. Numba-CUDA permite '
      'declarar esta jerarquía explícitamente al lanzar los kernels (NVIDIA, s. f.-c). Para una '
      'convolución se organiza im2col,SGEMM y gather backward; la biblioteca cuBLAS implementa '
      'álgebra lineal sobre memoria GPU (NVIDIA, s. f.-a). No se confunde im2col con leer o preparar archivos JPEG.')
    h('Secuencia, memoria y decisiones del diseño', 2)
    p('Por lote: copiar datos/máscaras;ejecutar forward y pérdida;calcular backward;actualizar '
      'todos los parámetros con Adam;verificar finitud. Tras96 lotes,exportar modelo y validar. '
      'Se conservan b real en la pérdida,padding SAME,MaxPool2×2,BN(momentum0.9,epsilon0.001), '
      'Adam(lr0.001,beta0.9/0.999,epsilon1e−7) y precisiónFP32 sin mixed precision/TF32.')
    p('Workspace im2col reutilizable576MiB;tope previsto1408MiB. La reserva real de buffers '
      'propios fue1,207,125,096bytes(aprox.1151.2MiB),dentro del límite. El diseño tiled16×16 '
      'se conservó como referencia;antes de congelar las corridas se eligió SGEMM por admisión '
      'preliminar≈0.60 frente a≈0.26s/lote (n=1, no resultado final). Se reutilizó CPUhistórica '
      'compatible;no se impuso una campaña nueva ni igualdad de máscaras de backends distintos.')

    # Page 7 — implementation execution, all GPU runs, admission.
    page(); h('5. Implementación con CUDA Python')
    p('Se conservó el notebook original y se creó una copia documentada con el flujo CUDA y '
      'salidas reales de la última corrida. Los kernels propios se implementaron con numba-cuda0.30.4 '
      '(Numba0.67.0);SGEMM usa cuBLAS mediante ctypes sobre los mismos buffers y stream. '
      'La GPU fue GTX1650,compute capability7.5,driver591.59 y toolkitCUDA13.1. TensorFlow '
      'es una compilación CPU y se usa solo como referencia,validación/test y persistencia.')
    p('La preparación utiliza ThreadPoolExecutor con6workers y hasta12tareas pendientes. Un '
      'coordinador conserva el orden;los SHA-256 de X/y coincidieron con CPU. Se mantuvieron '
      'datos,pesos iniciales,15épocas,batch128 y1440actualizaciones. Seed43;shuffle PCG64 '
      'y máscaras Bernoulli con probabilidades originales. No reproducen exactamente el RNG de TensorFlow histórico.')
    h('Ejecución y temporización', 2)
    p('Cada prueba fue un proceso nuevo. Se calentaron128+20 muestras train,128val y32test '
      'y se verificó la restauración de pesos,Adam,BN y contador. El cronómetro de pared '
      'sincronizado incluye staging,máscaras,H2D,cómputo,D2H y validaciónCPU por época. '
      'Los eventos CUDA se registran por grupo;no sustituyen la pared ni son tiempo activo puro de kernels.')
    gpu_obs = [r for r in obs if r['dispositivo']=='gpu']
    table('Cinco ejecuciones CUDA completas: segundos y test real', ['RUN', 'Preparación CPU', 'Entrenamiento', 'Test CPU', 'Accuracy test'],
          [[f'G{i+1:02}',*[f"{float(r[k]):.3f}" for k in ('preparacion_cpu_s','entrenamiento_s','evaluacion_cpu_s')], '98.621%'] for i,r in enumerate(gpu_obs)],
          [.6,1.6,1.6,1.35,1.65], 9.5)
    p('Media de entrenamiento418.608±1.916s;CV0.458%;ICt95%nominal[416.228,420.987]s. '
      'El semiancho relativo0.568%cumple el criterio previo≤5%;no fue necesario ampliar a7/9 '
      'repeticiones. No se excluyeron corridas completas por ser lentas o por su accuracy.')
    h('Comprobaciones de corrección y evidencias conservadas', 2)
    p('Aprobaron51pruebas de operadores,78comparaciones de adjuntos locales con activaciones '
      'idénticas y10pasos sintéticos de integración,contador y guardado/recarga. Se conservaron '
      'diagnósticos estrictos de red completa fallidos:se observaron cambios de rutas MaxPooling '
      'cerca de empatesFP32. Las pruebas locales respaldan la implementación;no certifican '
      'igualdad global bit a bit ni corrección absoluta por obtener alta accuracy.')
    p('Las cinco corridas terminaron con código0 y contadorAdam1440. Los checkpoints tienen '
      'contenido numérico canónico idéntico en este entorno. El monitoreo por segundo registró '
      'CPU por procesador lógico,GPU,VRAM,temperatura y fase. Los apartados siguientes presentan '
      'las capturas originales y extractos de logs;no son simulaciones del rendimiento.')

    # Page 8 — original CPU screenshot, genuine hardware and logs.
    page(); h('5.1 Evidencia de ejecución: preparación en CPU', 2)
    p('Captura de la ventana real del monitor alimentado por psutil y nvidia-smi durante la '
      'cuarta ejecución CUDA. Muestra CPU por12procesadores lógicos y baja actividadGPU '
      'mientras la CPU prepara las imágenes. La muestra corresponde al17/11/2026,23:39:47UTC.')
    image('resultados/evidencias/EXP-003/20261117T233947Z-cpu.png','02_captura_cpu.png',
          'PreparaciónCPU,G04(RUN-20261117T233923Z-gpu-04). El árbol del proceso usa225.2%CPU;la GPU registra1%. Captura original conservada.', 6.1)
    p('Un porcentaje de proceso superior a100% resulta de sumar actividad de varios hilos; '
      'no significa más de100%de un solo núcleo. Las barras por procesador son actividad del '
      'sistema y no atribución exclusiva del proyecto. Seis workers estaban configurados. La '
      'captura pertenece a preparación CPU de la variante optimizada;no es una captura del fit CPUhistórico.')
    h('Registro real de la referencia CPU', 3)
    hardware = load(source(cpu+'/'+cpu_runs[0]+'/hardware.json'))
    code(['CPU: AMD Ryzen 5 4600H; 6 físicos / 12 lógicos',
          'TensorFlow: is_cuda_build=false; dispositivo efectivo CPU:0',
          'TensorFlow intra-op=6, inter-op=1; BLAS=6'])
    cpu_log = source(cpu+'/'+cpu_runs[0]+'/stdout.log').read_text(encoding='utf8')
    selected = [line for line in cpu_log.splitlines() if line.startswith(('RESULTADO preparacion_cpu:', 'RESULTADO entrenamiento_CNN-B:', 'RESULTADO evaluacion_CNN-B:'))]
    code(selected)
    p('Extractos literales del log de C01:preparación17.963012s,fitCNN-B803.847403s '
      'y evaluación2.444951s. El hardware efectivoCPU y los registros por procesador '
      'respaldan la ejecución base. No se fabricó una captura retroactiva de ese entrenamiento.')

    # Page 9 — GPU original screenshot, real metrics and exact final logs.
    page(); h('5.2 Evidencia de ejecución: entrenamiento en GPU', 2)
    p('El dispositivo se reconoció como NVIDIA GeForce GTX1650 con4096MiB,compute '
      'capability7.5 y simulador desactivado. La siguiente captura fue tomada durante '
      'entrenamientoCNN-B,época11,de la tercera repetición;no corresponde a reposo ni a una nueva ejecución.')
    image('resultados/evidencias/EXP-003/20261117T233706Z-gpu.png','03_captura_gpu.png',
          'G03(RUN-20261117T233146Z-gpu-03),muestra23:37:05.912UTC,captura23:37:06.954UTC. ActividadGPU88%,VRAM1213/4096MiB y CPU del proceso98.9%.', 6.1)
    table('Monitoreo de entrenamientoGPU por repetición', ['RUN', 'Actividad media', 'Máxima', 'VRAM máxima', 'T máxima'],
          [[f'G{i+1:02}',f"{float(r['gpu_util_media_pct']):.2f}%",f"{float(r['gpu_util_max_pct']):.0f}%",f"{float(r['vram_max_mib']):.0f}MiB",f"{float(r['temperatura_max_c']):.0f}°C"] for i,r in enumerate(resources)],
          [.65,1.7,1.1,1.8,1.55], 9)
    p('La actividad alta demuestra usoGPU;1213MiB equivale a29.6%de capacidadVRAM, '
      'por lo que no demuestra saturación total. Actividad agregada y ocupación porSM '
      'son métricas diferentes;la segunda no se midió. nvidia-smi permite registrar métricas '
      'de monitorización,cuya interpretación debe respetar su definición (NVIDIA,s. f.-b).')
    final_log = source(gpu+'/RUN-20261117T234658Z-gpu-05/stdout.log').read_text(encoding='utf8')
    real_lines = [line for line in final_log.splitlines() if 'evaluacion_final_cpu' in line or 'completada' in line]
    code(real_lines)
    p('Log literal deG05:tras1440pasos se evalúa enCPU y termina la corrida. Los cinco '
      'procesos registran salida0. Las capturas y los logs documentan fases diferentes '
      'y complementarias;no se presentan como mediciones simultáneas de una única corrida.')

    # Page 10 — comparison and formulas with all replicate structure.
    page(); h('6. Análisis y conclusiones')
    table('Comparación de fases equivalentes, n=5 por variante', ['Fase', 'CPU media(s)', 'Variante CUDA(s)', 'Speedup', 'Reducción'],
          [[r['descripcion'],f"{float(r['media_cpu_s']):.3f}",f"{float(r['media_variante_cuda_s']):.3f}",f"{float(r['speedup']):.4f}×",f"{float(r['reduccion_pct']):.2f}%"] for r in comparison],
          [2.0,1.2,1.4,1.1,1.1], 9.3)
    p('S=mediaCPU/mediaCUDA;ahorro=mediaCPU−mediaCUDA;reducción(%)=100×(1−mediaCUDA/mediaCPU). '
      'Se calcula un cociente de medias,no una media de cocientes de corridas emparejadas artificialmente.')
    p('Entrenamiento:S=805.6497257/418.6076830=1.924593739×. El ahorro es387.042s '
      '(6.451min),una reducción48.041%. Se pasa de13.427 a6.977min incluyendo copias '
      'y validación. Un speedup1.925×no significa192.5%menos tiempo.')
    image(analysis+'/graficas/01_tiempos_por_fase.png','04_comparacion_fases.png',
          'Cinco observaciones por variante,puntos individuales y media±DEmuestral. Cada panel tiene escala propia con origen cero. La suma es derivada de tres fases.', 6.5)
    p('CPU/GPU:DEentrenamiento2.723/1.916s y CV0.338/0.458%. Incluso el máximoGPU '
      'es menor que el mínimoCPU. Esto respalda estabilidad local,sin eliminar sesgos '
      'temporales/térmicos. Los ICt95%de medias son nominales y no son intervalos del speedup.')

    # Page 11 — graph and the two first mandatory discussion questions.
    page(); h('6.1 Speedup por fase y discusión técnica', 2)
    image(analysis+'/graficas/02_speedup_por_fase.png','05_speedup.png',
          'Cocientes de medias;1×indica mismo tiempo. La preparación mejora enCPU. El testCPU registra0.932×y no muestra aceleración.', 6.6)
    h('¿Por qué procesar imágenes/archivos con varios núcleos CPU y delegar la matemática a GPU?', 3)
    p('Los archivos requieren acceso al almacenamiento,decodificación JPEG y transformaciones '
      'de entrada. El paralelismo entre archivos permite usar6workersCPU con orden fijo;en esta '
      'prueba la preparación bajó17.325→8.034s sin cambiar los tensores. Esa mejora2.156× '
      'pertenece aCPU. Después,las convoluciones y productos densos tienen muchas operaciones '
      'regulares sobre lotes,adecuadas para distribuir entre hilosGPU y una biblioteca matricial '
      'especializada. Mantener archivos enCPU y matemática de entrenamiento enGPU coincide '
      'con ese reparto observado;no demuestra que toda preparación posible sea siempre más rápida enCPU.')
    h('¿Cómo CUDA permitió reducir o cambiar el tiempo de procesamiento?', 3)
    p('Numba-CUDA distribuye im2col/col2im,activaciones,pooling,BN,Dropout,pérdida '
      'y Adam;cuBLAS realiza SGEMM con buffersGPU y el mismo stream. Reutilizar memoria '
      'evita reservarla en cada operación. La combinación reduce el fit a418.608s '
      'frente a805.650sCPU,aun contandoH2D/D2H y validaciónCPU. La suma de preparación, '
      'fit y test baja825.297→429.134s:1.9232×y48.00%menos tiempo. Ese total incluye '
      'también la mejoraCPU y no equivale a todo el notebook original.')
    p('La referenciaCPU usa librerías con varios hilos;la comparación incluye cambios '
      'de backend y orquestación. Los datos no permiten atribuir un porcentaje causal '
      'preciso a cada kernel,al hardware aislado o a la reutilización de buffers.')

    # Page 12 — overhead and third required discussion question.
    page(); h('6.2 Transferencias y factores que limitan CUDA', 2)
    image(analysis+'/graficas/03_desglose_entrenamiento.png','06_desglose.png',
          'Desglose contable del entrenamiento. EventosCUDA y validación son subintervalos;el residuo se calcula por diferencia. No es tiempo activo exclusivo de kernels.', 6.6)
    table('Componentes incluidos en418.608s medios de entrenamiento', ['Componente', 'Media(s)', 'Parte del total'],
          [[r['descripcion'],f"{float(r['media_s']):.3f}",f"{float(r['porcentaje_entrenamiento']):.3f}%"] for r in components],
          [4.0,1.4,1.4], 9.2)
    h('¿Qué factores justifican o limitan la utilización de CUDA?', 3)
    p('La justifican el volumen de convoluciones/productos matriciales,1440actualizaciones '
      'y el dominio temporal deCNN-B. La limitan transferencias,staging,generaciónCPU de '
      'máscaras,llamadas y sincronizaciónHost,exportación por época,validaciónCPU,dependencias '
      'entre capas/pasos y memoria disponible. H2D+D2H suman3.528s(0.843%de la pared); '
      'validaciónCPU y residuo suman74.265s(17.741%). No se suman otra vez al entrenamiento.')
    p('Los eventos pueden contener huecos de envíoHost;el residuo no es una fase '
      'perfilada exclusivamente de Python. Por eso no se infiere un límite formal de '
      'Amdahl,ocupaciónSM ni ancho de bandaPCIe. No hay speedup de transferencias '
      'porque CPUhistórica no tiene una fase equivalente medida. El testCPU subió '
      '0.170sde media;su corta duración y mayor variabilidad impiden asegurar una regresión general.')

    # Page 13 — quality, final conclusions, APA7 references.
    page(); h('6.3 Calidad, conclusiones y alcance de los resultados', 2)
    image(gpu+'/graficas/03_curvas_aprendizaje.png','07_curvas.png',
          'Evolución de entrenamientoGPU y validaciónCPU durante15épocas. Las cinco ejecuciones coinciden numéricamente en este entorno y usan las mismas claves aleatorias.', 6.5)
    p('TestGPU:98.620689%(2574/2610aciertos),loss0.061565;validación final98.582375%. '
      'CPUhistóricaCNN-B:92.874–98.889%. No se concluye mejora de calidad por usarGPU '
      'ni equivalencia de trayectorias. La prueba usa el test interno,no vídeo,test oficial '
      'de28imágenes ni fotos propias. No hay separación acreditada por persona/sesión.')
    bullet('El entrenamientoCNN-B se aceleró1.9246×y redujo48.04%su tiempo,incluyendo copias y validación.')
    bullet('El repartoCPU/GPU permitió reducir la suma comparable48.00%;la mejora de preparación esCPU y el test no mejoró.')
    bullet('Paralelismo tensorial y SGEMM especializado son explicaciones coherentes con código y tiempos;su contribución individual no se aisló.')
    bullet('Los resultados son locales:CPUhistórica no alternada,RNG/shuffle distintos y condiciones térmicas no igualadas. No se certifica generalización a otro hardware.')
    bullet('UsoGPU y fin correcto están documentados;no se certifican saturación literal,ocupaciónporSM ni igualdad globalbitabit. Se conservaron pruebas fallidas y límites.')
    h('Referencias', 2)
    references = [
        'Akash. (s. f.). ASL alphabet [Conjunto de datos]. Kaggle. https://www.kaggle.com/datasets/grassknoted/asl-alphabet',
        'NVIDIA. (s. f.-a). cuBLAS library (CUDA Toolkit 13.1) [Documentación]. https://docs.nvidia.com/cuda/archive/13.1.0/cublas/index.html',
        'NVIDIA. (s. f.-b). NVIDIA System Management Interface [Documentación]. Recuperado el 17 de noviembre de 2026, de https://docs.nvidia.com/deploy/nvidia-smi/index.html',
        'NVIDIA. (s. f.-c). Writing CUDA kernels. En Numba CUDA. Recuperado el 17 de noviembre de 2026, de https://nvidia.github.io/numba-cuda/user/kernels.html']
    for reference in references:
        paragraph = p(reference)
        titles = ('ASL alphabet', 'cuBLAS library (CUDA Toolkit 13.1)', 'NVIDIA System Management Interface', 'Writing CUDA kernels')
        title = next(t for t in titles if t in reference)
        left, right = reference.split(title, 1)
        paragraph.clear()
        paragraph.add_run(left)
        paragraph.add_run(title).italic = True
        paragraph.add_run(right)
        paragraph.paragraph_format.left_indent = Inches(.5)
        paragraph.paragraph_format.first_line_indent = Inches(-.5)
        paragraph.paragraph_format.space_after = Pt(4)
        for run in paragraph.runs:
            run.font.size = Pt(8.8)

    destination = out / 'Informe_Proyecto_3_CUDA.docx'
    doc.save(destination)
    (out / 'informe-rendimiento.md').write_text('# Proyecto 3: Mejorando algoritmos con CUDA\n\n'+'\n'.join(md), encoding='utf8')
    sources.add(Path(__file__).resolve())
    with (out / 'fuentes.csv').open('x',encoding='utf8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['ruta_relativa','bytes','sha256'])
        writer.writeheader()
        writer.writerows(dict(ruta_relativa=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(sources))
    info=dict(fecha_utc=datetime.now(timezone.utc).isoformat(),comando=[sys.executable,*sys.argv],cwd=str(Path.cwd()),
              python=sys.version,python_docx=importlib.metadata.version('python-docx'),
              responsable='Codex por solicitud del usuario',autores_confirmados=['David Domínguez23712','Jonathan Diaz23873','Luis Padilla23663'],
              docx_sha256=sha(destination),tablas=table_number,figuras=figure_number,
              secciones_obligatorias=6,entrenamiento_ejecutado=False,pdf_generado=False,
              etapa='generacion; revision visual Word posterior',fuentes=len(sources))
    (out/'generacion.json').write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps(info,ensure_ascii=False))


if __name__ == '__main__':
    main()
