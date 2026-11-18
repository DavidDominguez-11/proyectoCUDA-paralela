"""Audit existing artifacts only: no model imports, training or CUDA execution."""
import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def rows(path):
    with path.open(encoding='utf8', newline='') as source:
        return list(csv.DictReader(source))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destino', type=Path, required=True)
    out = parser.parse_args().destino.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(inicio_utc=datetime.now(timezone.utc).isoformat(),
                  tipo='verificacion_estatica_y_recalculo_sobre_evidencias',
                  entrenamiento_ejecutado=False, controles=[])
    checks = report['controles']

    def check(name, condition, **details):
        checks.append(dict(control=name, aprobado=bool(condition), **details))

    def manifest(path, base):
        entries = rows(path)
        failures = []
        for entry in entries:
            relative = entry.get('ruta_relativa', entry.get('ruta'))
            target = base / relative
            if not target.is_file() or target.stat().st_size != int(entry['bytes']) or sha(target) != entry['sha256']:
                failures.append(relative)
        check(path.relative_to(ROOT).as_posix(), not failures, entradas=len(entries), fallos=failures)

    try:
        cpu = ROOT / 'resultados/secuencial/EXP-002'
        gpu = ROOT / 'resultados/paralelo/EXP-003'
        cfg = read(ROOT / 'configuraciones/EXP-003.json')
        cseries, gseries = read(cpu / 'serie.json'), read(gpu / 'serie.json')
        check('cinco_procesos_gpu_completos', len(gseries['corridas']) == 5 and all(r['codigo'] == 0 for r in gseries['corridas']))
        manifest(gpu / 'artefactos-experimento.csv', ROOT)
        for folder, names in ((cpu, cseries['runs']), (gpu, [r['run_id'] for r in gseries['corridas']])):
            for name in names:
                manifest(folder / name / 'artefactos.csv', folder / name)
        for relative, digest in cfg['codigo']['fuentes'].items():
            check('fuente_congelada:' + relative, sha(ROOT / relative) == digest, sha256=digest)
        check('configuracion_gpu_congelada', sha(ROOT / 'configuraciones/EXP-003.json') == sha(gpu / 'configuracion.json') == gseries['configuracion_sha256'])
        check('pdf_normativo', sha(ROOT / 'docs/Proyecto 3 CUDA.pdf') == '2dd8efc1ffb36d1b0908fbb400808a713b99e1c1322f71843e3e22198f72eb6d')
        check('notebook_original', sha(ROOT / 'LAB3-DATASCIENCE/Lab3_ASL_Deep_Learning.ipynb') == cfg['codigo']['notebook_original_sha256'])
        split = rows(cpu / 'manifest_splits.csv')
        counts = Counter(r['split'] for r in split)
        check('conteos_y_separacion_exacta', counts == dict(train=12180, val=2610, test=2610)
              and len({r['ruta_relativa'] for r in split}) == len(split)
              and len({r['sha256'] for r in split}) == len(split), conteos=dict(counts),
              clases=dict(Counter(r['clase'] for r in split)))
        mismatches = [r['ruta_relativa'] for r in split if sha(Path(cfg['datos']['ruta']) / r['ruta_relativa']) != r['sha256']]
        check('17400_archivos_reales_sin_cambios', not mismatches, fallos=mismatches)
        tensor_ref = read(cpu / cseries['runs'][0] / 'tensores.json')
        raw = {}
        for variant, folder, names in (('cpu', cpu, cseries['runs']), ('gpu', gpu, [r['run_id'] for r in gseries['corridas']])):
            for name in names:
                run = folder / name
                tensors = read(run / 'tensores.json')
                check('tensores:' + name, all(tensors[s][k] == tensor_ref[s][k]
                      for s in ('train', 'val', 'test') for k in ('images_sha256', 'labels_sha256')))
                measurements = rows(run / 'tiempos.csv')
                fit = next(r for r in measurements if r['fase'] == 'entrenamiento_CNN-B')
                raw[name] = float(fit['duracion_s'])
                check('carga:' + name, int(fit['epocas']) == 15 and int(fit['pasos']) == 1440 and int(fit['batch_size']) == 128)
                initial = read(cpu / (name if variant == 'cpu' else cseries['runs'][0]) / 'estados-iniciales.json')['CNN-B']
                check('estado_inicial:' + name, sha(ROOT / initial['archivo']) == initial['archivo_sha256'])
                if variant == 'gpu':
                    epochs = rows(run / 'epocas.csv')
                    metric = read(run / 'metricas.json')
                    model = read(run / 'modelo-guardado.json')
                    check('modelo_y_epocas:' + name, len(epochs) == 15 and int(epochs[-1]['pasos_acumulados']) == 1440
                          and metric['pasos'] == 1440 and sha(ROOT / model['ruta']) == model['sha256'])
                    check('estado_gpu_importado:' + name, read(run / 'hardware.json')['initial_checkpoint_sha256'] == initial['archivo_sha256'])
                    check('accuracy_test_coherente:' + name, math.isclose(metric['test']['accuracy'] * 2610, round(metric['test']['accuracy'] * 2610), abs_tol=.001), accuracy=metric['test']['accuracy'])
        stats = rows(gpu / 'resumen.csv')
        for variant in ('cpu', 'gpu'):
            values = [raw[r] for r in raw if ('-gpu-' in r) == (variant == 'gpu')]
            recorded = next(r for r in stats if r['dispositivo'] == variant and r['fase'] == 'entrenamiento_s')
            check('estadisticas_entrenamiento:' + variant,
                  math.isclose(statistics.mean(values), float(recorded['media_s']), abs_tol=1e-9)
                  and math.isclose(statistics.stdev(values), float(recorded['desviacion_muestral_s']), abs_tol=1e-9),
                  media_s=statistics.mean(values), de_s=statistics.stdev(values))
        derived = rows(gpu / 'comparacion_cnn_b.csv')
        check('tabla_derivada_desde_crudos', all(math.isclose(float(r['entrenamiento_s']), raw[r['run_id']], abs_tol=1e-9) for r in derived))
        for r in rows(gpu / 'speedup.csv'):
            check('speedup:' + r['fase'], math.isclose(float(r['speedup']), float(r['media_cpu_s']) / float(r['media_gpu_s']), abs_tol=1e-12))
        for label, expected in (('operadores', 51), ('adjuntos', 78)):
            evidence = read(ROOT / cfg['validacion'][label])
            check('pruebas_archivadas:' + label, evidence['estado'] == 'aprobado' and len(evidence['pruebas']) == expected
                  and all(p['aprobado'] for p in evidence['pruebas']), pruebas=len(evidence['pruebas']))
        nbpath = ROOT / 'notebooks/Lab3_ASL_CUDA_con_salidas.ipynb'
        nb = read(nbpath)
        outputs = [o for c in nb['cells'] for o in c.get('outputs', [])]
        text = ''.join(''.join(o.get('text', [])) for o in outputs)
        lastlog = (gpu / gseries['corridas'][-1]['run_id'] / 'stdout.log').read_text(encoding='utf8')
        lines = [line for line in lastlog.splitlines() if line.startswith('EPOCH ')]
        check('notebook_salidas_reales', len(lines) == 15 and all(line in text for line in lines)
              and not any(o.get('output_type') == 'error' for o in outputs)
              and sha(nbpath) == read(ROOT / 'resultados/notebook-salidas/validacion.json')['notebook_sha256'], epocas_reales=len(lines))
        report['estado'] = 'aprobado' if all(c['aprobado'] for c in checks) else 'con_inconsistencias'
    except Exception as error:
        report.update(estado='error_auditoria', error=repr(error))
    report.update(fin_utc=datetime.now(timezone.utc).isoformat(), script_sha256=sha(Path(__file__)))
    (out / 'verificacion.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf8')
    print(json.dumps(dict(estado=report['estado'], controles=len(checks), fallos=[c for c in checks if not c['aprobado']], error=report.get('error')), ensure_ascii=False))
    return 0 if report['estado'] == 'aprobado' else 1


if __name__ == '__main__':
    raise SystemExit(main())
