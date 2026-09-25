"""Run three small software smoke jobs and verify readback from the results API."""
import argparse
import json
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validity_pipeline import run


def main():
    url = os.environ['MATMED_API_URL'].rstrip('/')
    read_token = os.environ['MATMED_READ_TOKEN']
    results = []
    for seed in (42, 123, 999):
        args = argparse.Namespace(mode='smoke', data=None, source='', output='runs/verified',
            seed=seed, epochs=2, iterations=1, batch_size=4, max_len=32, eval_samples=16,
            min_validity=40., device='cpu')
        directory, report = run(args)
        upload = json.loads((directory / 'connector_status.json').read_text())
        if not upload.get('uploaded'):
            raise RuntimeError(f'Upload failed: {upload}')
        request = Request(url + '/api/runs/' + report['run_id'],
                          headers={'Authorization': 'Bearer ' + read_token})
        with urlopen(request, timeout=20) as response:
            remote = json.load(response)
        if remote['metrics'] != report['metrics'] or remote['status'] != 'completed':
            raise AssertionError('Remote report does not match local completed result')
        results.append({'seed': seed, 'report_path': str(directory / 'report.json'),
                        'run_id': report['run_id'], 'status': report['status'],
                        'source_sha256': report['provenance']['source_sha256'],
                        'metrics': report['metrics'], 'api_roundtrip_verified': True})
    artifact = {'software_validation_only': True, 'mode': 'smoke',
                'training': '2 fixture pretraining epochs and 1 PPO iteration per seed',
                'runtime': 'CPU', 'results': results,
                'limitations': ['Not a real-data experiment', 'Not drug-discovery validation',
                                'No claimed improvement in molecular validity']}
    output = Path('artifacts/smoke_results.json')
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, allow_nan=False) + '\n')
    print(f'Verified three authenticated report roundtrips. Results: {output}')


if __name__ == '__main__':
    main()
