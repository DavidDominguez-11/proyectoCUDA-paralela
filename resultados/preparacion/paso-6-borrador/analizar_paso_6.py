"""Derive step-6 tables/figures from sealed CPU/CUDA evidence; never train."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics as st
import sys

ROOT = Path(__file__).resolve().parents[1]
PHASES = ('preparacion_cpu_s', 'entrenamiento_s', 'evaluacion_cpu_s', 'suma_comparable_s')
LABELS = ('Preparación CPU', 'Entrenamiento CNN-B', 'Evaluación final CPU', 'Suma de tres fases')
COLORS = ('#0072B2', '#D55E00')


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf8'))


def read_csv(path):
    with path.open(encoding='utf8', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, data):
    with path.open('x', encoding='utf8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def table(header, data):
    return '\n'.join(['| ' + ' | '.join(header) + ' |',
                      '| ' + ' | '.join('---' for _ in header) + ' |',
                      *['| ' + ' | '.join(str(v) for v in row) + ' |' for row in data]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destino', type=Path, required=True)
    args = parser.parse_args()
    out = args.destino.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    inputs = set()

    def source(path):
        inputs.add(path)
        return path

    cpu = ROOT / 'resultados/secuencial/EXP-002'
    gpu = ROOT / 'resultados/paralelo/EXP-003'
    cseries = load(source(cpu / 'serie.json'))
    gseries = load(source(gpu / 'serie.json'))
    config = load(source(gpu / 'configuracion.json'))
    if len(cseries['runs']) != 5 or len(gseries['corridas']) != 5 or any(r['codigo'] != 0 for r in gseries['corridas']):
        raise ValueError('Expected the five closed CPU and five successful CUDA runs')
    if sha(gpu / 'configuracion.json') != gseries['configuracion_sha256']:
        raise ValueError('GPU configuration identity changed')
    # Verify full closed evidence and each CPU run before calculating anything.
    for path, base in [(source(gpu / 'artefactos-experimento.csv'), ROOT),
                       *[(source(cpu / name / 'artefactos.csv'), cpu / name) for name in cseries['runs']]]:
        for row in read_csv(path):
            target = base / row['ruta_relativa']
            if target.stat().st_size != int(row['bytes']) or sha(target) != row['sha256']:
                raise ValueError('Sealed evidence changed: ' + str(target))
    records = []
    for device, folder, names in [('cpu', cpu, cseries['runs']),
                                  ('gpu', gpu, [r['run_id'] for r in gseries['corridas']])]:
        for name in names:
            timing = source(folder / name / 'tiempos.csv')
            values = {r['fase']: float(r['duracion_s']) for r in read_csv(timing)}
            record = dict(dispositivo=device, run_id=name, fuente=timing.relative_to(ROOT).as_posix(),
                          preparacion_cpu_s=values['preparacion_cpu'],
                          entrenamiento_s=values['entrenamiento_CNN-B'],
                          evaluacion_cpu_s=values['evaluacion_CNN-B'],
                          validacion_anidada_s=values['validacion_anidada_CNN-B'])
            record['suma_comparable_s'] = sum(record[p] for p in PHASES[:3])
            if device == 'gpu':
                record.update(h2d_eventos_s=sum(values[p] for p in ('h2d_inicial', 'h2d_lotes')),
                              d2h_eventos_s=sum(values[p] for p in ('d2h_control', 'd2h_modelo', 'd2h_optimizador')),
                              computo_intervalos_stream_s=sum(values[p] for p in ('forward_loss', 'backward', 'adam_metricas')))
                record['administracion_y_staging_s'] = record['entrenamiento_s'] - sum(record[p] for p in
                    ('h2d_eventos_s', 'd2h_eventos_s', 'computo_intervalos_stream_s', 'validacion_anidada_s'))
            records.append(record)
    # Uniform CSV schema; empty fields do not imply measured zero CPU transfers.
    keys = list(dict.fromkeys(k for r in records for k in r))
    records = [{k: r.get(k, '') for k in keys} for r in records]
    write_csv(out / 'observaciones.csv', records)
    summaries = []
    comparison = []
    for phase, label in zip(PHASES, LABELS):
        group = {}
        for device in ('cpu', 'gpu'):
            values = [r[phase] for r in records if r['dispositivo'] == device]
            mean, sd = st.mean(values), st.stdev(values)
            half = 2.7764451051977987 * sd / math.sqrt(5)  # t(0.975,df=4)
            group[device] = dict(dispositivo=device, fase=phase, n=5, media_s=mean,
                                 de_muestral_s=sd, mediana_s=st.median(values), min_s=min(values), max_s=max(values),
                                 cv_pct=100 * sd / mean, ic95_nominal_inferior_s=mean-half,
                                 ic95_nominal_superior_s=mean+half)
            summaries.append(group[device])
        c, g = group['cpu']['media_s'], group['gpu']['media_s']
        comparison.append(dict(fase=phase, descripcion=label, n_cpu=5, n_gpu=5, media_cpu_s=c,
                               media_variante_cuda_s=g, ahorro_s=c-g, speedup=c/g, reduccion_pct=100*(1-g/c)))
    # Cross-check existing figures/tables without changing their closed inputs.
    previous = read_csv(source(gpu / 'speedup.csv'))
    for row in comparison:
        old = next(r for r in previous if r['fase'] == row['fase'])
        if not math.isclose(row['speedup'], float(old['speedup']), abs_tol=1e-12):
            raise ValueError('Derived speedup does not match archived result')
    write_csv(out / 'estadisticas.csv', summaries)
    write_csv(out / 'comparacion_fases.csv', comparison)
    component_keys = ('computo_intervalos_stream_s', 'h2d_eventos_s', 'd2h_eventos_s',
                      'validacion_anidada_s', 'administracion_y_staging_s')
    component_labels = ('Cómputo: intervalos del stream', 'H2D: eventos', 'D2H: eventos',
                        'Validación CPU anidada', 'Residuo de pared / staging / control')
    train_mean = comparison[1]['media_variante_cuda_s']
    components = [dict(componente=key, descripcion=label,
                       media_s=st.mean(r[key] for r in records if r['dispositivo'] == 'gpu'),
                       porcentaje_entrenamiento=100*st.mean(r[key] for r in records if r['dispositivo'] == 'gpu')/train_mean)
                  for key, label in zip(component_keys, component_labels)]
    if not math.isclose(sum(r['media_s'] for r in components), train_mean, abs_tol=1e-8):
        raise ValueError('Non-additive decomposition')
    write_csv(out / 'desglose_entrenamiento.csv', components)
    quality = read_csv(source(gpu / 'calidad_comparada.csv'))
    resources = read_csv(source(gpu / 'recursos_resumen.csv'))
    source(ROOT / 'docs/Proyecto 3 CUDA.pdf')
    source(ROOT / 'docs/trazabilidad/revision-pasos-1-5.md')
    source(ROOT / 'resultados/auditoria/pasos-1-5-revision-02/verificacion.json')
    source(ROOT / 'configuraciones/EXP-002.json')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    figures = out / 'graficas'
    figures.mkdir()

    def save(fig, name):
        fig.savefig(figures / (name + '.png'), dpi=180)
        fig.savefig(figures / (name + '.svg'))
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), layout='constrained')
    for ax, phase, label in zip(axes.flat, PHASES, LABELS):
        for j, device in enumerate(('cpu', 'gpu')):
            values = [r[phase] for r in records if r['dispositivo'] == device]
            summary = next(r for r in summaries if r['fase'] == phase and r['dispositivo'] == device)
            ax.scatter([j + delta for delta in (-.12, -.06, 0, .06, .12)], values,
                       color=COLORS[j], s=29, alpha=.8, zorder=3)
            ax.errorbar(j+.20, summary['media_s'], yerr=summary['de_muestral_s'],
                        fmt='D', color=COLORS[j], capsize=4, markersize=6, zorder=4)
            ax.annotate(f"{summary['media_s']:.3f} s", (j, max(values)), xytext=(0, 13),
                        textcoords='offset points', ha='center', fontsize=10)
        upper = max(r[phase] for r in records)
        ax.set(xticks=(0, 1), xticklabels=('CPU histórica', 'Variante CUDA'), ylabel='Tiempo de pared (s)',
               title=label + (' (derivada)' if phase == PHASES[-1] else ''), ylim=(0, upper*1.24), xlim=(-.4, 1.5))
        ax.grid(axis='y', alpha=.18)
    handles = [Line2D([], [], marker='o', ls='', color='#555555', label='Cada punto: un RUN (n = 5 por variante)'),
               Line2D([], [], marker='D', ls='', color='#555555', label='Diamante: media; barra: ± DE muestral')]
    fig.suptitle('CNN-B: comparación de tiempos por fase\nEscalas verticales distintas; origen cero en cada panel', fontsize=14)
    fig.legend(handles=handles, loc='outside lower center', ncols=2, frameon=False)
    save(fig, '01_tiempos_por_fase')

    fig, ax = plt.subplots(figsize=(10.5, 5.4), layout='constrained')
    for j, row in enumerate(comparison):
        ax.barh(j, row['speedup'], color=COLORS[1] if j in (1, 3) else COLORS[0], height=.57)
        ax.text(row['speedup']+.035, j, f"{row['speedup']:.3f}×  |  reducción {row['reduccion_pct']:.2f}%", va='center', fontsize=10)
    ax.axvline(1, ls='--', color='#333333', lw=1.1)
    ax.set(yticks=range(4), yticklabels=('Preparación CPU (1 → 6 workers)', 'Entrenamiento CNN-B (incluye copias)',
                                       'Evaluación final (CPU en ambas)', 'Suma de tres fases (derivada)'),
           xlabel='Speedup = media CPU / media variante CUDA (×)', xlim=(0, 3.32),
           title='CNN-B: aceleración observada por fase\nLínea discontinua: 1×, mismo tiempo; valores < 1× indican mayor tiempo')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=.16)
    fig.text(.5, -.002, 'Cocientes de medias de cinco RUN por variante; no se emparejan corridas ni se calculan IC del cociente.', ha='center', fontsize=9)
    save(fig, '02_speedup_por_fase')

    fig, ax = plt.subplots(figsize=(11, 5.2), layout='constrained')
    palette = ('#D55E00', '#0072B2', '#56B4E9', '#009E73', '#777777')
    for j, (row, color) in enumerate(zip(components, palette)):
        ax.barh(j, row['media_s'], color=color, height=.60)
        ax.text(row['media_s']+3, j, f"{row['media_s']:.3f} s  ({row['porcentaje_entrenamiento']:.2f}%)", va='center')
    ax.set(yticks=range(5), yticklabels=component_labels, xlim=(0, 445), xlabel='Tiempo medio (s)',
           title=f'Desglose contable del entrenamiento de la variante CUDA: {train_mean:.3f} s\nIntervalos de eventos y pared; no representa tiempo activo exclusivo de kernels')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=.16)
    save(fig, '03_desglose_entrenamiento')

    per_run = table(('Variante', 'RUN', 'Preparación (s)', 'Entrenamiento (s)', 'Test CPU (s)', 'Suma comparable (s)'),
                    [('CPU' if r['dispositivo']=='cpu' else 'CUDA', r['run_id'],
                      *[f'{r[p]:.3f}' for p in PHASES]) for r in records])
    phase_table = table(('Fase', 'CPU media ± DE (s)', 'Variante CUDA media ± DE (s)', 'Ahorro (s)', 'Speedup', 'Reducción (%)'),
                        [(row['descripcion'],
                          *[f"{next(s for s in summaries if s['fase']==row['fase'] and s['dispositivo']==d)['media_s']:.3f} ± "
                            f"{next(s for s in summaries if s['fase']==row['fase'] and s['dispositivo']==d)['de_muestral_s']:.3f}" for d in ('cpu', 'gpu')],
                          f"{row['ahorro_s']:.3f}", f"{row['speedup']:.6f}×", f"{row['reduccion_pct']:.3f}") for row in comparison])
    spread_table = table(('Variante', 'n', 'Mediana (s)', 'Mín–máx (s)', 'CV (%)', 'ICt95% nominal de la media (s)'),
                         [('CPU' if s['dispositivo']=='cpu' else 'CUDA', 5, f"{s['mediana_s']:.3f}",
                           f"{s['min_s']:.3f}–{s['max_s']:.3f}", f"{s['cv_pct']:.3f}",
                           f"[{s['ic95_nominal_inferior_s']:.3f}, {s['ic95_nominal_superior_s']:.3f}]")
                          for s in summaries if s['fase']=='entrenamiento_s'])
    breakdown_table = table(('Componente anidado', 'Media (s)', 'Porcentaje del entrenamiento'),
                            [(r['descripcion'], f"{r['media_s']:.3f}", f"{r['porcentaje_entrenamiento']:.3f}%") for r in components])
    quality_table = table(('Variante', 'RUN', 'Accuracy test (%)', 'Loss test'),
                          [('CPU' if r['dispositivo']=='cpu' else 'CUDA', r['run_id'],
                            f"{100*float(r['accuracy']):.3f}", f"{float(r['loss']):.6f}") for r in quality])
    fit, total = comparison[1], comparison[3]
    transfer = sum(r['media_s'] for r in components if r['componente'] in ('h2d_eventos_s', 'd2h_eventos_s'))
    cpu_floor = components[3]['media_s'] + components[4]['media_s']
    report = f'''# Paso 6. Análisis y conclusiones — CNN-B CPU/CUDA

Estado: análisis del paso 6 completado para el alcance medido. **Documento de
trabajo en Markdown; no es el informe final y no se genera PDF en esta etapa.**
Responsable: Codex por solicitud del usuario. DEC-024; CAM-009; BIT-011; PROC-09.
Fuente normativa: [Proyecto 3 CUDA.pdf](../../../docs/Proyecto%203%20CUDA.pdf),
página 2, apartado 6. Elaborado a partir de cinco RUN CPU EXP-002 y cinco CUDA
EXP-003 existentes; no se entrenó ni midió el hardware nuevamente.

## 1. Alcance de la comparación

Se compara CNN-B base: 17,400 imágenes ASL, 29 clases, splits 12,180/2,610/2,610,
RGB 64×64 float32 /255, pesos iniciales preservados, Adam, 15 épocas, batch 128 y
último lote de 20; 96 lotes por época y 1440 actualizaciones. No se usa aumento
de imágenes en este alcance. Los tensores conservan las mismas identidades.

La referencia denominada secuencial es la implementación CPU original bajo el
protocolo fijo documentado; TensorFlow y BLAS usan paralelismo interno (6 hilos),
por lo que no representa un solo núcleo. La variante paralela prepara archivos
con 6 workers CPU y entrena mediante kernels Python Numba-CUDA y SGEMM cuBLAS
FP32 en una GTX 1650 de 4 GiB. Validación por época y test final permanecen CPU.

El tiempo de entrenamiento CUDA es pared sincronizada e incluye preparación de
lotes/máscaras, H2D, forward/backward/Adam, exportación D2H, validación CPU y
administración. El fit CPU también incluye validación. Imports, construcción,
warmup/restauración, EDA, figuras y guardado final quedan fuera. La suma comparable
comprende preparación + entrenamiento CNN-B + test CNN-B; no es el notebook
íntegro ni los totales CPU de 14 fases o de ramas adaptativas.

CPU es una referencia histórica no alternada con GPU. Datos, arquitectura y
volumen coinciden; RNG/shuffle no reproducen máscaras/orden exactos. Horario,
temperatura y preparación 1→6 workers difieren. Se interpreta el resultado como
aceleración observada de implementaciones con carga equivalente, sin certificar
condiciones idénticas ni aislar causalmente el efecto exclusivo del hardware.
Véase la [revisión previa](../../../docs/trazabilidad/revision-pasos-1-5.md).

## 2. Resultados individuales y comparación por fase

Cada fila es un proceso independiente; su posición no establece una pareja
experimental CPU/GPU. Se mantienen todas las cinco observaciones por variante.
La suma es derivada de tres intervalos disjuntos, no una nueva medición continua.

{per_run}

**Medias y desviaciones muestrales**, n=5 por variante. Un ahorro o reducción
negativos indican que la variante CUDA tardó más en esa fase.

{phase_table}

![Tiempos por fase con cinco observaciones y media ± DE](graficas/01_tiempos_por_fase.png)

Figura 1. Cada punto corresponde a un RUN; diamante y barra representan media
y ±DE muestral, no IC. Cada panel tiene escala vertical propia con origen cero.
La fase de preparación es CPU en ambas variantes (1 y 6 workers), y el test
también es CPU. Fuente: [observaciones.csv](observaciones.csv) y
[estadisticas.csv](estadisticas.csv).

## 3. Cálculo e interpretación del speedup

Para cada fase se usan los valores sin redondear:

- `media = suma(tiempos de los 5 RUN) / 5`.
- `speedup S = media_CPU / media_variante_CUDA`.
- `ahorro_s = media_CPU - media_variante_CUDA`.
- `reduccion_pct = 100 × (1 - media_variante_CUDA / media_CPU)`.

Entrenamiento: `S = {fit['media_cpu_s']:.7f} / {fit['media_variante_cuda_s']:.7f}
= {fit['speedup']:.9f}×`. El ahorro medio es **{fit['ahorro_s']:.3f} s**,
aproximadamente **{fit['ahorro_s']/60:.3f} minutos**, y la reducción del tiempo
es **{fit['reduccion_pct']:.3f}%**. Se pasa de {fit['media_cpu_s']/60:.3f} a
{fit['media_variante_cuda_s']/60:.3f} minutos para el entrenamiento completo medido.
Un speedup de 1.925× no significa 192.5% menos tiempo: corresponde a cerca de
48.04% menos tiempo. Los cocientes se calculan entre medias de grupos; no se
promedian ratios de filas emparejadas artificialmente.

![Speedup por fase, incluyendo fases CPU](graficas/02_speedup_por_fase.png)

Figura 2. La línea en 1× indica igualdad de tiempo. El test es 0.932×: no se
aceleró. El speedup de preparación 2.156× es una mejora CPU. El 1.923× de la suma
combina el cambio CPU de preparación con entrenamiento CUDA y test CPU; no se
atribuye por completo al dispositivo GPU. Fuente:
[comparacion_fases.csv](comparacion_fases.csv). No se calculan intervalos de
confianza del cociente ni se presentan barras de DE como si fueran esos intervalos.

La suma de tres fases baja **{total['ahorro_s']:.3f} s ({total['reduccion_pct']:.3f}%)**.
Ese ahorro es casi todo entrenamiento: su contribución al ahorro neto de la
suma es {100*fit['ahorro_s']/total['ahorro_s']:.2f}%. Este porcentaje describe
diferencias de medias, no una descomposición causal experimental.
No se calcula speedup H2D/D2H: la CPU histórica no tiene una transferencia
equivalente medida. Se informa su coste dentro del entrenamiento CUDA.

## 4. Variabilidad y suficiencia de las observaciones

{spread_table}

La baja dispersión y que incluso el máximo GPU sea menor que el mínimo CPU
son consistentes con una reducción local amplia y repetible del tiempo de la
carga fijada. No se efectuó una prueba de significancia ni se atribuye toda la
diferencia a CUDA aislada del backend y demás condiciones. Los ICt95% son
nominales: `media ± 2.776445 × DE / sqrt(5)`, con 4 grados de libertad; suponen
observaciones aproximadamente independientes y error compatible con ese modelo.
No cubren sesgos térmicos, cambios temporales o de implementación.

El semiancho relativo de la media GPU es 0.568%, inferior al objetivo previo
de 5% que condicionaba ampliar a 7/9 repeticiones. Por eso cinco observaciones
son suficientes para ese criterio de precisión local. El test CPU tiene solo
unos 2–3 s y mayor variabilidad relativa en la variante CUDA; no hay evidencia
que permita atribuir su incremento a una causa concreta o asegurar una regresión
general. Todas las estadísticas adicionales están en [estadisticas.csv](estadisticas.csv).

## 5. Coste de transferencias, validación y administración

{breakdown_table}

![Desglose contable del entrenamiento CUDA](graficas/03_desglose_entrenamiento.png)

Figura 3. Las cinco medias suman {train_mean:.3f} s por construcción. Fuente:
[desglose_entrenamiento.csv](desglose_entrenamiento.csv). Los eventos son
intervalos del stream y pueden incluir huecos de envío Host; no equivalen a
tiempo activo puro de kernels. La última fila es un residuo calculado, no una
fase perfilada directamente ni una medida exclusiva de sobrecarga Python.

H2D+D2H suman **{transfer:.3f} s**, el **{100*transfer/train_mean:.3f}%** de la
pared. Están incluidos y no eliminan la mejora observada. No obstante, su pequeño
peso relativo no prueba que PCIe tenga ancho de banda óptimo ni que las copias
sean gratuitas. La implementación usa un único stream sin solapamiento y una
exportación adicional final de modelo incluida de manera conservadora.

La validación CPU ocupa {components[3]['media_s']:.3f} s y el residuo
{components[4]['media_s']:.3f} s: juntos representan
{100*cpu_floor/train_mean:.3f}% de la pared. Muestran trabajo que no desaparece
al acelerar solo los productos matriciales, aunque no son una fracción serial
formal de Amdahl ni un límite inmutable: el residuo incluye varios costes y
posibles huecos Host ya contabilizados en eventos. No se extrapola un speedup
ideal ni se calcula eficiencia dividiendo por el número de núcleos CUDA.

## 6. Posibles causas de las diferencias de rendimiento

| Factor | Evidencia disponible | Interpretación y grado de certeza |
|---|---|---|
| Trabajo matemático abundante y regular | Cinco convoluciones, Dense, BN, pooling, Dropout y Adam; batch128 y 1440pasos; tiempo de entrenamiento menor | La distribución de elementos y reducciones entre hilos/bloques permite aprovechar GPU. Mecanismo coherente con código y tiempos; no se aisló la aportación de cada kernel. |
| Backend de productos matriciales | Serie usa im2col/col2im Numba y SGEMM cuBLAS FP32 pedantic; pesos y buffers residen en GPU | Biblioteca especializada y reutilización de buffers pueden reducir costes. El cambio de implementación forma parte del resultado; no es una comparación del mismo binario en dos dispositivos. |
| Ensayo previo del backend | Admisión sintética n=1: tiled≈0.60s/lote, SGEMM≈0.26s/lote, antes de congelar EXP-003 | Explica la decisión DEC-022; no se usa como speedup final ni prueba estadística independiente. |
| Preparación de archivos en CPU | Se conserva Pillow/RGB/resize/normalización; 6workers con orden fijo frente al bucle previo; tensores coincidentes | Leer/decodificar archivos permite concurrencia CPU sin enviar imágenes sin preparar a GPU. El menor tiempo medido apoya esta elección local; no demuestra que toda preparación posible sea más rápida en CPU. |
| Transferencias y llamadas Host | Eventos H2D/D2H y residuo de pared registrados; máscaras CPU, sincronización por paso y exportación por época | Reducen el beneficio frente a medir únicamente matemática GPU. No hay perfilado que asigne cada segundo del residuo a una función concreta. |
| Validación y test CPU | CPU valida cada época; test final conserva backendCPU | El trabajo CPU limita cuánto cae la pared total. Test no recibe aceleración CUDA; su pequeño aumento es una observación con variabilidad, no una causa demostrada. |
| Dependencias del algoritmo | Capas, forward/backward y Adam requieren orden; actualizaciones no se ejecutan en paralelo entre pasos | Tener más hilos no elimina dependencias. El reparto ocurre dentro de operaciones/lotes, no entrenando épocas independientemente. |
| Recursos del dispositivo | Monitor registra medias≈86–88%, máximos91–94%, VRAM1213/4096MiB, máximos69–70°C | Actividad elevada y uso real compatibles con trabajo GPU sostenido; no prueban ocupación por SM, saturación de VRAM, ausencia de throttling ni cuello exclusivo de cómputo/memoria. |
| Referencia histórica y RNG distinto | CPU yGPU no alternadas, shuffle/máscaras distintos, misma distribución y volumen | Pueden influir en tiempos/trayectorias; no se cuantificó su contribución. Deben acompañar las conclusiones. |

La separación CPU/GPU es coherente con la guía: CPU manipula archivos e imágenes,
organiza memoria Host y orquesta; GPU procesa matrices/tensores del entrenamiento.
Esto evita atribuir a CUDA la lectura de disco y usa el paralelismo masivo donde
hay más operaciones regulares. El resultado local favorece ese reparto; no se
probó una variante que decodifique archivos en GPU para establecer una ley general.

Las funciones/librerías adicionales principales son Numba-CUDA (`cuda.jit`,
grids, streams, buffers y eventos) y cuBLAS (`cublasSgemm_v2`) sobre el mismo
stream. El código Python explícito conserva im2col/col2im, activaciones, BN,
pooling, Dropout, pérdida y Adam. Los grids internos SGEMM pertenecen a NVIDIA;
no se atribuyen al diseño de kernels del equipo. TensorFlow funciona en CPU
para referencia/evaluación y persistencia, sin fallback de entrenamiento oculto.

## 7. Calidad del resultado y límites de corrección

{quality_table}

El test GPU es **98.620689%**, equivalente a 2574/2610 aciertos, con loss
0.061565 en las cinco corridas. Es la partición interna ASL fijada, no vídeo en
vivo, el test oficial de 28 imágenes ni las fotos propias. Mismos pesos, claves
aleatorias y operaciones deterministas hacen coherente la repetición del
resultado GPU; no son cinco semillas distintas para estudiar generalización.

La variación CPU no permite atribuir a CUDA una mejora de accuracy ni afirmar
trayectorias equivalentes. La ausencia de duplicados exactos no acredita
separación por persona/sesión. Las pruebas previas de 51 operadores y 78 adjuntos
locales, más integración sintética, respaldan la implementación bajo sus
condiciones; los diagnósticos globales estrictos fallidos permanecen fallidos,
con sensibilidad documentada de MaxPooling cerca de empates FP32. No se certifica
igualdad bit a bit ni corrección absoluta mediante accuracy elevada.

## 8. Conclusiones del paso 6

1. **Sí se redujo el tiempo de entrenamiento medido de CNN-B:** de 805.650 a
   418.608s en promedio, speedup1.924594× y reducción48.041%. El resultado incluye
   copias CPU↔GPU,staging y validaciónCPU; no depende de omitir esos costes.
2. **La variante completa de tres fases comparables también reduce el tiempo:**
   825.297→429.134s, speedup1.923167×. La preparación mejora por concurrenciaCPU;
   el test continúaCPU y no mejora. No se afirma acelerar todo el laboratorio.
3. **La explicación técnica más consistente** combina paralelismo de operaciones
   tensoriales,kernelsCUDA y SGEMM especializado con reutilización de memoria.
   Transferencias,controlHost,dependencias y validaciónCPU limitan el beneficio.
   Los tiempos no permiten repartir causalmente la ganancia entre esos factores.
4. **El resultado es local y delimitado:** cargaCNN-B fija,FP32,hardware/entorno
   registrados y CPUhistórica. Baja dispersión respalda repetibilidad temporal
   bajo ese protocolo; no certifica condiciones idénticas ni el mismo resultado
   en otro equipo,otras semillas u otro tamaño de datos.
5. **La calidad observada es adecuada como resultado descriptivo del test fijado**,
   sin demostrar superioridad de GPU en aprendizaje ni equivalencia global
   estricta. UsoGPU está acreditado;saturación literal y ocupaciónporSM no.

Estas conclusiones cubren el apartado6. El informe final,PDF,contribuciones del
equipo y revisión de entrega siguen pendientes; no se generan en esta actividad.

## 9. Procedimiento reproducible y fuentes

Desde la raíz del proyecto,con PCUDA existente:

```powershell
.\\PCUDA\\Scripts\\python.exe -X utf8 scripts/analizar_paso_6.py --destino resultados/comparacion/paso-6
```

El script rechaza carpetas existentes;para regenerar se debe usar otro nombre
bajo `resultados/comparacion/`. Verifica sellos,lee tiempos originales,recalcula
tablas y dibuja figuras PNG/SVG. No descarga datos,no importa el modelo,no
entrena,no inicializa CUDA ni produce PDF. Redondeo solo al presentar;CSV y
cálculos conservan precisión de las fuentes.

- [Script de análisis](../../../scripts/analizar_paso_6.py).
- [Fuentes con bytes y SHA-256](fuentes.csv),[procedencia y versiones](procedencia.json)
  y [manifiesto de derivados](artefactos.csv).
- [Resultados CPU](../../secuencial/EXP-002/resultados.md) y
  [CUDA](../../paralelo/EXP-003/resultados.md);configuraciones y RUN enlazados allí.
- [Auditoría previa aprobada](../../auditoria/pasos-1-5-revision-02/verificacion.json).
- [Capturas y su procedencia](../../evidencias/EXP-003/README.md) y
  [recursos GPU resumidos](../../paralelo/EXP-003/recursos_resumen.csv).
- Figuras vectoriales: [tiempos](graficas/01_tiempos_por_fase.svg),
  [speedup](graficas/02_speedup_por_fase.svg),[desglose](graficas/03_desglose_entrenamiento.svg).

Las entradas experimentales se preservan;estos archivos son cálculos derivados,
no corridas nuevas. El sello de derivados se verifica al cerrar la actividad;
la revisión visual/documental posterior se registra en BIT-011.
'''
    (out / 'analisis-y-conclusiones.md').write_text(report, encoding='utf8')
    source(Path(__file__).resolve())
    fingerprints = [dict(ruta_relativa=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(inputs)]
    write_csv(out / 'fuentes.csv', fingerprints)
    provenance = dict(inicio_utc=started, fin_utc=datetime.now(timezone.utc).isoformat(),
                      tipo='analisis_derivado_sin_nuevo_entrenamiento', responsable='Codex por solicitud del usuario',
                      comando=[sys.executable, *sys.argv], cwd=str(Path.cwd()), python=sys.version,
                      paquetes={p: importlib.metadata.version(p) for p in ('matplotlib', 'numpy', 'pillow')},
                      decisiones=['DEC-024'], cambios=['CAM-009'], experimentos=['EXP-002', 'EXP-003'],
                      n_cpu=5, n_gpu=5, entrenamiento_ejecutado=False, pdf_generado=False,
                      jerarquia_incertidumbre='Puntos RUN; barras DE muestral; ICt95 nominal solo media, no cociente',
                      integridad_exp003_entradas=184, figuras=3, fuentes=len(fingerprints))
    (out / 'procedencia.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding='utf8')
    write_csv(out / 'artefactos.csv', [dict(ruta_relativa=p.relative_to(out).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
                                      for p in sorted(out.rglob('*')) if p.is_file()])
    print(json.dumps(dict(estado='completado', destino=str(out), speedup=fit['speedup'],
                          reduccion_pct=fit['reduccion_pct'], ahorro_s=fit['ahorro_s'], figuras=3, pdf=False), ensure_ascii=False))


if __name__ == '__main__':
    main()
