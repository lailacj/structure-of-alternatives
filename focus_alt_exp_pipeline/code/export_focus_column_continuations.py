"""Export a small, deterministic selection of corrected continuation scores.

Run where the original arrays reside. Only top-five words, experimental words,
and up to four intervening words per experimental-word gap are exported.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arrays', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.arrays / 'vocab_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    words, vocab_sources = [], []
    for part in manifest['sources']:
        path = Path(part['path'])
        values = path.read_text().splitlines()
        assert len(values) == part['count']
        words.extend(values)
        vocab_sources.append({'name': path.name, 'sha256': sha(path), 'count': len(values)})
    lookup = {word.strip().lower(): i for i, word in enumerate(words)}
    assert len(lookup) == len(words) == manifest['total_count']
    groups = {}
    with args.sources.open() as stream:
        for row in csv.DictReader(stream):
            if row['dataset_family'] == 'novel_focus':
                groups.setdefault(row['context_id'], []).append(row)
    assert len(groups) == 16
    records = []
    for context, rows in sorted(groups.items()):
        assert len(rows) == 30 and len({r['prompt_id'] for r in rows}) == 1
        prompt_id = rows[0]['prompt_id']
        path = args.arrays / (prompt_id + '.log_probs.npy')
        digest = sha(path)
        assert {r['neutral_score_array_sha256'] for r in rows} == {digest}
        scores = np.load(path, allow_pickle=False).astype(float)
        assert len(scores) == len(words) and np.isfinite(scores).all() and (scores <= 0).all()
        for r in rows:
            for role in ['query', 'trigger']:
                assert abs(scores[lookup[r[role].strip().lower()]] - float(r[role+'_logprob_sum'])) < 1e-12
        targets = {lookup[r['query'].strip().lower()] for r in rows}
        assert len(targets) == 6
        order = np.argsort(-scores, kind='stable')
        positions = np.empty(len(scores), dtype=int)
        positions[order] = np.arange(len(scores))
        target_positions = sorted(int(positions[i]) for i in targets)
        selected = set(range(5)) | set(target_positions)
        gaps = []
        for lo, hi in zip(target_positions[:-1], target_positions[1:]):
            intervening = list(range(lo+1, min(hi, lo+5)))
            selected.update(intervening)
            gaps.append({'after_position': lo+1, 'before_position': hi+1,
                         'available_words': hi-lo-1, 'shown_positions': [i+1 for i in intervening]})
        candidates = []
        for position in sorted(selected):
            i = int(order[position])
            candidates.append({'word': words[i], 'vocabulary_position': position+1,
                               'log_probability': float(scores[i]), 'probability': float(np.exp(scores[i])),
                               'experimental': i in targets})
        records.append({'context': context, 'prompt_id': prompt_id, 'score_array_sha256': digest,
                        'support_size': len(scores), 'gaps': gaps, 'candidates': candidates})
    json.dump({'source_rows_sha256': sha(args.sources), 'vocabulary_sources': vocab_sources,
               'vocabulary_manifest_sha256': sha(manifest_path),
               'selection': 'Top 5; all 6 experimental words; first up to 4 words between each consecutive experimental pair.',
               'ties': 'Descending log probability, then original vocabulary order.',
               'contexts': records}, sys.stdout, indent=2)
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
