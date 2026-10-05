"""Offline documentation, migration, and recorded-result integrity checks."""
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'results/qwen_gsm8k_pilot_v1'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12), (a, b)


def check_docs():
    pages = [ROOT/'README.md', *ROOT.glob('docs/*.md'), *ROOT.glob('results/**/*.md')]
    for page in pages:
        text = page.read_text()
        assert text.count('```') % 2 == 0, page
        for target in re.findall(r'\]\(([^\s)]+)\)', text):
            parsed = urlsplit(target.strip('<>'))
            if parsed.scheme or not parsed.path:
                continue
            resolved = (page.parent/unquote(parsed.path)).resolve()
            assert resolved.is_relative_to(ROOT) and resolved.exists(), (page, target)
    for path in ROOT.glob('colabs/*.ipynb'):
        nb = read(path)
        assert nb['nbformat'] == 4
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
                assert not cell.get('outputs')
    return len(pages)


def check_results():
    inventory = read(ARCHIVE/'archive_inventory.json')
    for group in ('original_artifact_sha256', 'generator_export_sha256'):
        for name, expected in inventory[group].items():
            assert sha(ARCHIVE/name) == expected, name
    for name, expected in inventory['migrated_source_sha256'].items():
        assert sha(ROOT/name) == expected, f'Migration baseline changed: {name}; document a new code version explicitly'
    assert read(ARCHIVE/'manifest.json')['status'] == 'complete'
    assert not read(ARCHIVE/'development_lock.json')['advance_to_llama']
    for name, count in [('summary.json', 1319), ('development_summary.json', 747)]:
        summary = read(ARCHIVE/name)
        rows = summary['paired_results']
        indexed = {(r['seed'],r['state'],r['arm'],r['horizon']):r for r in rows}
        assert len(indexed) == len(rows) == 126
        for r in rows:
            native = indexed[r['seed'],r['state'],'native',r['horizon']]
            assert r['metrics']['count'] == count
            close(r['metrics']['exact_match'], r['metrics']['correct']/count)
            for metric, delta in r['paired_delta'].items():
                close(delta, r['metrics'][metric]-native['metrics'][metric])
        for p in summary['primary']:
            by_seed = {str(s):statistics.mean(r['paired_delta']['native_loss'] for r in rows
                       if r['seed']==s and r['arm']==p['arm'] and r['horizon']==20)
                       for s in (789,2026,31415)}
            for seed, delta in by_seed.items():
                close(delta, p['per_seed_mean_state_delta'][seed])
            close(statistics.mean(by_seed.values()),p['mean_loss_delta'])
            close(statistics.stdev(by_seed.values()),p['seed_sd'])
    histories = list((ARCHIVE/'generator_histories').glob('*.json'))
    assert len(histories) == 9
    for path in histories:
        d = read(path)
        assert d['episodes'] == len(d['history']) == 200
    return 252


if __name__ == '__main__':
    print(f'Checked {check_docs()} documents and {check_results()} archived evaluation records, source hashes, and nine generator histories.')
