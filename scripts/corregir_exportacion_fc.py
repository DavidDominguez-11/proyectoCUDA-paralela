"""Preserve the incomplete EXP-010; prepare corrected new campaign EXP-011."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[1]
s=(root/'scripts/preparar_experimento_fc.py').read_text(encoding='utf8').replace('EXP-010','EXP-011').replace('preparar_experimento_fc.py','preparar_experimento_fc_v2.py').replace('ejecutar_fc_notebook.py','ejecutar_fc_notebook_v2.py')
s=s.replace('fase=phase,tipo_medicion=kind,duracion_s=float(seconds),configuracion_sha256=sha(CONFIG),**other',
    "fase=phase,tipo_medicion=kind,duracion_s=float(seconds),configuracion_sha256=sha(CONFIG),**dict({'inicio_utc':'','fin_utc':''},**other)")
(root/'scripts/preparar_experimento_fc_v2.py').write_text(s,encoding='utf8')
r=(root/'scripts/ejecutar_fc_notebook.py').read_text(encoding='utf8').replace('EXP-010','EXP-011')
(root/'scripts/ejecutar_fc_notebook_v2.py').write_text(r,encoding='utf8')
s=(root/'scripts/resumir_ampliacion_fc.py').read_text(encoding='utf8').replace('EXP-010','EXP-011')
(root/'scripts/resumir_ampliacion_fc.py').write_text(s,encoding='utf8')
note=dict(estado='excluido_incompleto',causa='Fallo de exportación CSV: primera fila carecía de inicio_utc/fin_utc presentes en entrenamiento. FC-A completó 15épocas y test; no guardó checkpoint ni cierre. No incorporado a promedios.',
    correccion='EXP-011: todas las filas de tiempos definen el mismo esquema. Código matemático FC y validación sin cambios.',no_repetido='CPU histórica y CNN-B intactos')
(root/'resultados/paralelo/EXP-010/exclusion.json').write_text(json.dumps(note,indent=2,ensure_ascii=False),encoding='utf8')
for name in ['bitacora.md','cambios.md','decisiones.md']:
    with (root/'docs/trazabilidad'/name).open('a',encoding='utf8') as f:
        f.write('\n\nAmpliación FC, corrección documental del intento: EXP-010 completó 15épocas FC-A pero falló al exportar CSV de esquema heterogéneo; excluido por cierre incompleto. Se conserva íntegro con traceback,épocas,métricas y captura. EXP-011 corrige únicamente esquema de filas y retoma campaña nueva. No se ejecutó CPU/CNN-B.\n')
print(json.dumps(note,ensure_ascii=False))
