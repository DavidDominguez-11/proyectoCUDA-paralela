"""Resume only pending FC runs, reusing the already saved real first FC-A run."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
first=next((ROOT/'resultados/paralelo/EXP-011').glob('RUN-*-fc-a-01'))
assert (first/'modelo-guardado.json').is_file()
assert json.loads((first/'estado.json').read_text())['fase']=='completada'
source=(ROOT/'scripts/preparar_experimento_fc_v2.py').read_text(encoding='utf8')
source=source.replace('EXP-011','EXP-012').replace('preparar_experimento_fc_v2.py','preparar_experimento_fc_v3.py').replace('ejecutar_fc_notebook_v2.py','ejecutar_fc_notebook_v3.py')
source=source.replace(".style.hide(axis='index')",'')
source=source.replace('for repetition in range(1,6):',"for repetition in range(2 if name=='FC-A' else 1,6):")
source=source.replace('    runs=[]',"    runs=[dict(modelo='FC-A',repeticion=1,run_id="+repr(first.name)+",ruta="+repr(first.relative_to(ROOT).as_posix())+",reutilizado_sin_reentrenar=True)]")
source=source.replace('all_results=[]',"all_results=[dict(run_id="+repr(first.name)+",**json.loads((ROOT/"+repr(first.relative_to(ROOT).as_posix()+'/metricas.json')+").read_text(encoding='utf8')))]")
source=source.replace("15épocas y1440actualizaciones nuevas en total", "15épocas y1440actualizaciones por modelo y corrida")
(ROOT/'scripts/preparar_experimento_fc_v3.py').write_text(source,encoding='utf8')
runner=(ROOT/'scripts/ejecutar_fc_notebook_v2.py').read_text(encoding='utf8').replace('EXP-011','EXP-012')
(ROOT/'scripts/ejecutar_fc_notebook_v3.py').write_text(runner,encoding='utf8')
for rel in ['scripts/resumir_ampliacion_fc.py','scripts/integrar_notebook_fc.py']:
    text=(ROOT/rel).read_text(encoding='utf8').replace('EXP-011','EXP-012').replace(".style.hide(axis='index')",'')
    text=text.replace("run=BASE/item['run_id']", "run=ROOT/item.get('ruta',(BASE/item['run_id']).relative_to(ROOT).as_posix())")
    text=text.replace("(BASE/item['run_id']/'metricas.json')", "((ROOT/item.get('ruta',(BASE/item['run_id']).relative_to(ROOT).as_posix()))/'metricas.json')")
    # Markdown rendering without optional tabulate dependency.
    text=text.replace("def table(rows):return pd.DataFrame(rows).to_markdown(index=False,floatfmt='.6f')", "def table(rows):\n    keys=list(rows[0]);fmt=lambda v:f'{v:.6f}' if isinstance(v,float) else str(v)\n    return '| '+' | '.join(keys)+' |\\n| '+' | '.join(['---']*len(keys))+' |\\n'+'\\n'.join('| '+' | '.join(fmt(row.get(k,'')) for k in keys)+' |' for row in rows)")
    (ROOT/rel).write_text(text,encoding='utf8')
note=dict(estado='entrenamiento_completado_presentacion_fallida',causa='Pandas Styler exige jinja2 ausente en PCUDA; no se instala nada.',
    reutilizado=first.relative_to(ROOT).as_posix(),decision='Conservar y reutilizar esta corrida completa sin volver a entrenarla; EXP-012 ejecuta únicamente las nueve pendientes. Presentar DataFrame nativo sin Styler.')
(ROOT/'resultados/paralelo/EXP-011/continuacion.json').write_text(json.dumps(note,indent=2,ensure_ascii=False),encoding='utf8')
for name in ['bitacora.md','cambios.md','decisiones.md']:
    with (ROOT/'docs/trazabilidad'/name).open('a',encoding='utf8') as f:
        f.write('\n\nLa primera corrida FC-A EXP-011 terminó entrenamiento, test, checkpoint y monitor. Solo falló presentación Styler por jinja2 ausente. **Se reutiliza sin repetirla**. EXP-012 ejecutará únicamente nueve pendientes (FC-A2–5,FC-B1–5) y mostrará DataFrame nativo sin dependencias adicionales. Se preservan notebook y error originales.\n')
print(json.dumps(note,ensure_ascii=False))
