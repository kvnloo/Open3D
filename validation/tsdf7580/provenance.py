"""Record immutable build/run identifiers and inspect the actual archived tests."""
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import requests

REPO = 'kvnloo/Open3D'
HEAD = '9e2229ab12ec8a33491484c487105388408c45e7'
MERGE = '203aff22f0a49de34cfe548c39195afcbbe08059'
OUT = Path('results/provenance')
OUT.mkdir(parents=True, exist_ok=True)


def get(path, raw=False):
    url = 'https://api.github.com/repos/' + REPO + '/' + path
    r = requests.get(url, headers={'Authorization': 'Bearer ' + os.environ['GH_TOKEN'],
                                   'Accept': 'application/vnd.github+json'},
                     timeout=90, allow_redirects=False)
    if r.status_code in (301, 302, 303, 307, 308):
        # Signed artifact/log URL: do not forward repository credentials.
        r = requests.get(r.headers['Location'], timeout=120)
    r.raise_for_status()
    return r.content if raw else r.json()


def metadata():
    records = {}
    for kind, artifact, run, head in [
        ('control-wheel', 10798136279, 35964996507, '8f35b3fa05711aba1e57eb44291fb191ebbb0c50'),
        ('candidate-wheel', 10798350959, 35965027760, HEAD),
        ('candidate-build', 10796528351, 35965027760, HEAD)]:
        data = get(f'actions/artifacts/{artifact}')
        assert data['workflow_run']['id'] == run
        assert data['workflow_run']['head_sha'] == head
        assert not data['expired']
        records[kind] = data
    (OUT / 'artifacts.json').write_text(json.dumps(records, indent=2))
    return records


def collect():
    metadata()
    source_path = 'cpp/tests/t/geometry/VoxelBlockGrid.cpp'
    obj = get(f'contents/{source_path}?ref={MERGE}')
    source = base64.b64decode(obj['content'])
    digest = hashlib.sha1(b'blob ' + str(len(source)).encode() + b'\0' + source).hexdigest()
    assert digest == obj['sha']
    (OUT / 'VoxelBlockGrid.cpp').write_bytes(source)
    run = get('actions/runs/35965027760')
    job = get('actions/jobs/107521775350')
    assert job['run_id'] == run['id'] and run['head_sha'] == HEAD
    logs = get('actions/jobs/107521775350/logs', raw=True).decode('utf-8', errors='replace')
    (OUT / 'original-job.log').write_text(logs)
    (OUT / 'run.json').write_text(json.dumps(run, indent=2))
    (OUT / 'job.json').write_text(json.dumps(job, indent=2))
    src_names = re.findall(r'TEST_P\s*\(\s*VoxelBlockGridPermuteDevices\s*,\s*(\w+)\s*\)', source.decode())
    lines = logs.splitlines()
    selected = set()
    for i, line in enumerate(lines):
        if re.search(r'VoxelBlockGrid.*(?:FAILED|Integrate|ExtractTriangleMesh)|VoxelBlockGrid.cpp:|checking out|git log -1|203aff2|9e2229a', line, re.I):
            selected.update(range(max(0, i-2), min(len(lines), i+10)))
    excerpts = '\n'.join(lines[i] for i in sorted(selected))
    (OUT / 'failure-excerpts.txt').write_text(excerpts)
    finding = {'source_commit': MERGE, 'source_blob': digest, 'source_test_names': src_names,
               'log_mentions_separate_mesh_test': bool(re.search(r'VoxelBlockGrid[^\n]*\.ExtractTriangleMesh', logs)),
               'source_has_separate_mesh_test': 'ExtractTriangleMesh' in src_names}
    (OUT / 'source-log-comparison.json').write_text(json.dumps(finding, indent=2))
    print('SOURCE_LOG_COMPARISON', json.dumps(finding))


def inspect():
    candidates = [p for p in Path('build/bin').glob('*') if p.name in ('tests', 'unit_tests') and p.is_file()]
    assert len(candidates) == 1, [str(p) for p in candidates]
    binary = candidates[0].resolve()
    binary.chmod(binary.stat().st_mode | 0o111)
    record = {'binary_path': str(binary), 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest()}
    (OUT / 'binary.json').write_text(json.dumps(record, indent=2))
    subprocess.run(['otool', '-L', str(binary)], stdout=(OUT / 'linked-libraries.txt').open('w'), check=True)
    for label, args, timeout in [('list', ['--gtest_list_tests', '--gtest_filter=*VoxelBlockGrid*'], 120),
                                  ('rerun', ['--gtest_filter=*VoxelBlockGrid*Integrate*:*VoxelBlockGrid*ExtractTriangleMesh*',
                                             '--gtest_output=xml:' + str((OUT / 'cpp-rerun.xml').resolve())], 900)]:
        try:
            r = subprocess.run([str(binary)] + args, capture_output=True, text=True, timeout=timeout)
            (OUT / f'archived-{label}.log').write_text(r.stdout + r.stderr)
            record[label + '_exit'] = r.returncode
            if label == 'list':
                record['binary_has_separate_mesh_test'] = 'ExtractTriangleMesh' in r.stdout
            print('ARCHIVED_CPP', label, r.returncode)
        except subprocess.TimeoutExpired:
            record[label + '_timeout'] = True
        (OUT / 'binary.json').write_text(json.dumps(record, indent=2))
    print('BINARY_IDENTITY', json.dumps(record))
    if record.get('list_exit') != 0:
        raise RuntimeError('Archived binary could not list tests; identity remains unresolved')


if __name__ == '__main__':
    if sys.argv[1] == 'metadata': metadata()
    elif sys.argv[1] == 'collect': collect()
    elif sys.argv[1] == 'inspect': inspect()
    else: raise ValueError(sys.argv[1])
