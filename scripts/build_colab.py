"""Build the reproducible notebook and uploadable source bundle (standard library only)."""
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def markdown(source):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': source.splitlines(True)})


def code(source):
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                  'outputs': [], 'source': source.splitlines(True)})


markdown('''# MATMED: validity-first Colab experiment

This notebook runs the existing generator architecture with supervised pretraining,
token-level PPO, a frozen-prior KL anchor, and honest raw-validity measurement.
It **does not establish binding, safety, synthesis feasibility, or drug efficacy**.
The reaction and vision agents remain in the repository; their real-data validation
is not replaced with synthetic labels here.

**Start:** choose Runtime > Change runtime type > GPU. Upload the matching
`matmed_colab_bundle.zip` in the setup cell. This avoids accidentally cloning an
older GitHub revision. The notebook and bundle must come from the same build.
Outputs can be saved to Drive. Cloudflare hosts only the dashboard and results API.
''')
code('''from pathlib import Path
import os, sys, subprocess, zipfile, json
from google.colab import files

uploaded = files.upload()  # Select matmed_colab_bundle.zip only.
bundle = next((name for name in uploaded if name.endswith('.zip')), None)
if bundle is None:
    raise ValueError('Upload matmed_colab_bundle.zip from this repository build')
ROOT = Path('/content/matmed_validity')
ROOT.mkdir(exist_ok=True)
with zipfile.ZipFile(bundle) as archive:
    for member in archive.infolist():
        destination = (ROOT / member.filename).resolve()
        if not destination.is_relative_to(ROOT.resolve()):
            raise ValueError('Unsafe archive path')
    archive.extractall(ROOT)
os.chdir(ROOT)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements-colab.txt'], check=True)
import torch
print('Torch:', torch.__version__, 'GPU:', torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], check=True)
''')
markdown('''## Connect the prototype and preserve results

Deploy `prototype/` using its README first. Add `MATMED_API_URL` and
`MATMED_WRITE_TOKEN` in Colab's Secrets panel and enable notebook access.
Use a different `READ_TOKEN` to sign into the dashboard. Never paste the write
token into frontend code. Connection is optional; local artifacts remain available
if an upload fails. `connector_status.json` records upload success or failure.
''')
code('''USE_DRIVE = True
CONNECT_DASHBOARD = False  # Set True after deploying the Worker and setting Colab Secrets.
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    OUTPUT = Path('/content/drive/MyDrive/MATMED/runs')
else:
    OUTPUT = ROOT / 'runs'
OUTPUT.mkdir(parents=True, exist_ok=True)
if CONNECT_DASHBOARD:
    from google.colab import userdata
    os.environ['MATMED_API_URL'] = userdata.get('MATMED_API_URL').rstrip('/')
    os.environ['MATMED_WRITE_TOKEN'] = userdata.get('MATMED_WRITE_TOKEN')
    from urllib.request import urlopen
    with urlopen(os.environ['MATMED_API_URL'] + '/api/health', timeout=20) as response:
        print(json.load(response))
print('Results directory:', OUTPUT)
''')
markdown('''## Software smoke test
Runs a tiny test fixture, two supervised epochs and one PPO update. Low validity
is expected for this randomly initialized tiny model. A completed smoke test proves
execution and report transport, not research success. Gradient tests above verify
that reward-driven updates change the generator and preserve frozen weights.
''')
code('''subprocess.run([sys.executable, 'validity_pipeline.py', '--mode', 'smoke',
                '--output', str(OUTPUT)], check=True)
reports = sorted(OUTPUT.glob('*/report.json'), key=lambda p: p.stat().st_mtime)
smoke_path = reports[-1]
smoke = json.loads(smoke_path.read_text())
print(json.dumps({'status': smoke['status'], 'mode': smoke['mode'], 'metrics': smoke['metrics']}, indent=2))
print((smoke_path.parent / 'connector_status.json').read_text())
from IPython.display import display, Image
display(Image(filename=str(smoke_path.parent / 'curves.png')))
''')
markdown('''## Real corpus and three-seed experiment

Supply a curated, licensed CSV with a `smiles` column. Record the data source and
version. Invalid rows and duplicates are counted; over-length molecules are excluded,
not truncated. Scaffold groups never cross splits; too few groups is an error.
No online download or synthetic fallback occurs. Source metadata is an audit record,
not automatic verification of the scientific quality of your data.

Start with enough pretraining for the **raw** prior validity to exceed 40%.
Otherwise the run ends with `needs_pretraining`, saves its results, and does not run PPO.
The final aim before multi-objective studies should be substantially higher validity.
The test partition is not used to choose the pretrained checkpoint.
''')
code('''RUN_RESEARCH = False  # Enable only after setting real corpus and provenance.
DATA_CSV = '/content/drive/MyDrive/MATMED/data/molecules.csv'
DATA_SOURCE = ''  # Required: dataset name, release/version, and source identifier.
SEEDS = [42, 123, 999]
PRETRAIN_EPOCHS = 20
PPO_ITERATIONS = 50
if RUN_RESEARCH:
    if not Path(DATA_CSV).is_file() or not DATA_SOURCE.strip():
        raise ValueError('Supply a real CSV and DATA_SOURCE before starting research')
    if not torch.cuda.is_available():
        raise RuntimeError('Select a GPU Colab runtime before the full experiment')
    for seed in SEEDS:
        subprocess.run([sys.executable, 'validity_pipeline.py', '--mode', 'research',
            '--data', DATA_CSV, '--source', DATA_SOURCE, '--seed', str(seed),
            '--epochs', str(PRETRAIN_EPOCHS), '--iterations', str(PPO_ITERATIONS),
            '--batch-size', '64', '--eval-samples', '1024', '--min-validity', '40',
            '--output', str(OUTPUT)], check=True)
else:
    print('Research is disabled. Smoke results are not research evidence.')
''')
markdown('''## Inspect and download evidence
Each run has its own directory: report JSON, CSV history, curves, data manifest,
and checkpoint configuration/vocabulary. Repeated executions never append to old
curves. Compare only runs with matching corpus hashes and settings. Do not treat
episodes as independent experimental replicates or interpret smoke runs as ablations.
''')
code('''import pandas as pd
rows = []
for path in sorted(OUTPUT.glob('*/report.json')):
    r = json.loads(path.read_text())
    rows.append({'run_id': r['run_id'], 'mode': r['mode'], 'seed': r['seed'],
                 'status': r['status'], 'source': r.get('dataset', {}).get('source'),
                 **r['metrics']})
display(pd.DataFrame(rows))
archive_path = Path('/content/matmed_results.zip')
with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in OUTPUT.glob('*/*'):
        if path.suffix in {'.json', '.csv', '.png'}:
            archive.write(path, path.relative_to(OUTPUT))
print('Model checkpoints remain in:', OUTPUT)
files.download(str(archive_path))
''')
notebook = {'nbformat': 4, 'nbformat_minor': 5,
            'metadata': {'colab': {'name': 'MATMED_Validity_First.ipynb'},
                         'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
                         'language_info': {'name': 'python'}, 'accelerator': 'GPU'}, 'cells': cells}
for index, cell in enumerate(cells):
    cell['id'] = f'matmed-{index:02d}'
path = ROOT / 'matmed_validity_colab.ipynb'
path.write_text(json.dumps(notebook, indent=2) + '\n')
artifacts = ROOT / 'artifacts'
artifacts.mkdir(exist_ok=True)
with zipfile.ZipFile(artifacts / 'matmed_colab_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for pattern in ('*.py', '*.txt', '*.md', 'tests/*.py', 'scripts/*.py'):
        for source in sorted(ROOT.glob(pattern)):
            archive.write(source, source.relative_to(ROOT))
    archive.write(path, path.name)
print(path)
print(artifacts / 'matmed_colab_bundle.zip')
with zipfile.ZipFile(artifacts / 'matmed_prototype.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for pattern in ('prototype/package*.json', 'prototype/wrangler.jsonc', 'prototype/README.md',
                    'prototype/src/*', 'prototype/public/*', 'prototype/tests/*'):
        for source in sorted(ROOT.glob(pattern)):
            if source.is_file():
                archive.write(source, source.relative_to(ROOT))
print(artifacts / 'matmed_prototype.zip')
