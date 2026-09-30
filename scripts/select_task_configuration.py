#!/usr/bin/env python3
"""Select a declared candidate on the internal task stream, never final scores."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def select(paths):
    records = []
    identity = None
    for path in paths:
        path = Path(path)
        result = json.loads((path/'result.json').read_text())
        record = json.loads((path/'configuration.json').read_text())
        configuration = record['configuration']
        provenance = record['provenance']
        if result.get('status') != 'completed' or provenance.get('stage') != 'validation':
            raise ValueError('Only completed initial-period validation candidates may be selected')
        if result.get('provenance') != provenance or record.get('protocol_sha256') != result.get('protocol_sha256'):
            raise ValueError('Result and configuration provenance disagree')
        current = (provenance['source_commit'], provenance['selection_protocol_sha256'], configuration['method'],
                   provenance['source_sha256'], provenance['selection_features_sha256'])
        if identity is not None and identity != current:
            raise ValueError('Candidate methods, source versions, or validation splits differ')
        if result['protocol_sha256'] != current[1]:
            raise ValueError('Result and declared validation protocol disagree')
        identity = current
        score = float(result['metrics']['nlpd'])
        import math
        if not math.isfinite(score): raise ValueError('Finite validation NLPD required')
        expected = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
        if expected != provenance['configuration_sha256']:
            raise ValueError('Configuration hash mismatch')
        records.append(dict(path=str(path), nlpd=score, configuration_sha256=expected,
            result_sha256=hashlib.sha256((path/'result.json').read_bytes()).hexdigest()))
    if not records:
        raise ValueError('At least one declared validation candidate required')
    winner = min(records, key=lambda r: (r['nlpd'], r['configuration_sha256']))
    return dict(status='selected', criterion='query_weighted_observation_nlpd',
        source_commit=identity[0], selection_protocol_sha256=identity[1], method=identity[2],
        source_sha256=identity[3], selection_features_sha256=identity[4],
        configuration_sha256=winner['configuration_sha256'], winner=winner, candidates=records,
        convergence_qualified=False, main_table_admitted=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(select(a.candidates),indent=2)+'\n')


if __name__=='__main__': main()
