"""Execute actual CPU then CUDA notebooks, checkpointing cell outputs durably."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'resultados/preparacion/notebooks-entrega'


def stamp():return datetime.now(timezone.utc).isoformat()


def main():
    global BASE
    ap=argparse.ArgumentParser()
    ap.add_argument('--revision',default='notebooks-entrega')
    args=ap.parse_args()
    BASE=ROOT/'resultados/preparacion'/args.revision
    contracts=json.loads((BASE/'ejecuciones.json').read_text(encoding='utf8'))
    specs=ROOT/'data/jupyter/kernels'
    (specs/'pcuda').mkdir(exist_ok=True,parents=True)
    (specs/'pcuda/kernel.json').write_text(json.dumps(dict(argv=[sys.executable,'-m','ipykernel_launcher',
        '--HistoryManager.hist_file=:memory:','-f','{connection_file}'],display_name='PCUDA',language='python')),encoding='utf8')
    os.environ['IPYTHONDIR']=str(ROOT/'data/jupyter/ipython')
    os.environ['JUPYTER_RUNTIME_DIR']=str(ROOT/'data/jupyter/runtime')
    overall=[]
    for item in contracts:
        output=Path(item['salida']);run=Path(item['run'])
        if output.exists():raise FileExistsError(output)
        with (run/'contexto-captura.log').open('x',encoding='utf8') as ctx:
            subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts/capturar_contexto.py'),
                            '--destino',str(run/'contexto')],check=True,cwd=ROOT,stdout=ctx,stderr=subprocess.STDOUT)
        notebook=nbformat.read(item['plantilla'],as_version=4)
        manager=KernelSpecManager(kernel_dirs=[str(specs)],ensure_native_kernel=False)
        km=KernelManager(kernel_name='pcuda',kernel_spec_manager=manager)
        state=dict(inicio_utc=stamp(),tipo=item['tipo'],run_id=item['run_id'],celdas_terminadas=0,estado='ejecutando')
        (run/'invocacion.json').write_text(json.dumps(dict(comando=[sys.executable,*sys.argv],notebook=item['plantilla'],
            metodo='NotebookClient; codigo real separado por fases, sin subproceso de entrenamiento',inicio_utc=stamp()),indent=2),encoding='utf8')
        def save_state():
            nbformat.write(notebook,output)
            (BASE/'estado.json').write_text(json.dumps(state,indent=2),encoding='utf8')
        def completed(cell,cell_index,**kwargs):
            state.update(celdas_terminadas=cell_index+1,actualizado_utc=stamp(),ultima_celda=cell.source[:100])
            save_state()
        client=NotebookClient(notebook,km=km,timeout=14400,resources={'metadata':{'path':str(ROOT)}},on_cell_executed=completed)
        save_state();print('INICIO',item['tipo'],item['run_id'],flush=True)
        try:
            client.execute(cwd=str(ROOT))
            state.update(estado='completada',fin_utc=stamp(),codigo=0)
        except BaseException:
            state.update(estado='fallida',fin_utc=stamp(),codigo=1,error=traceback.format_exc())
            raise
        finally:
            save_state()
            if km.has_kernel:km.shutdown_kernel(now=True)
            # Native cell streams, in execution order; distinguish stderr honestly.
            for stream in ('stdout','stderr'):
                with (run/(stream+'.log')).open('x',encoding='utf8') as f:
                    for cell in notebook.cells:
                        for entry in cell.get('outputs',[]):
                            if entry.output_type=='stream' and entry.name==stream:f.write(entry.text)
            (run/'notebook-ejecucion.json').write_text(json.dumps(state,indent=2),encoding='utf8')
        with (run/'artefactos.csv').open('x',encoding='utf8',newline='') as f:
            import csv
            writer=csv.DictWriter(f,fieldnames=['ruta_relativa','bytes','sha256']);writer.writeheader()
            for path in sorted(run.rglob('*')):
                if path.is_file() and path.name!='artefactos.csv':
                    writer.writerow(dict(ruta_relativa=path.relative_to(run).as_posix(),bytes=path.stat().st_size,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        overall.append(dict(**state,notebook_sha256=hashlib.sha256(output.read_bytes()).hexdigest()))
        (BASE/'completadas.json').write_text(json.dumps(overall,indent=2),encoding='utf8')
        print('FIN',item['tipo'],state['estado'],flush=True)


if __name__=='__main__':main()
