"""Conservative paper-table export with explicit pairing and qualification proof."""
import csv
import math
from pathlib import Path

GATES = ('implementation_parity', 'information_boundary', 'numerical_validity',
         'timing_scope', 'source_provenance')
COLUMNS = ('dataset', 'method', 'seed', 'protocol_sha256', 'hardware_id', 'status',
           'main_table_admitted', 'metric_scale', 'rmse', 'nlpd', 'crps', 'ece', 'online_seconds',
           'seconds_per_query', 'fp64_add_mul_fma_subset', 'counter_status', 'counter_task_coverage')


def table_rows(records):
    """One row per run; never silently pool unmatched masks, hardware, or seeds.

    qualification must contain evidenced gates and paired run identities. A
    Boolean admission field in a result alone is insufficient. Paired methods
    must be provided together so their identities can actually be checked.
    """
    records = list(records)
    identities = {}
    for record in records:
        identity = record.get('run_id')
        if not identity or identity in identities:
            raise ValueError('Unique nonempty run_id required')
        identities[identity] = record
    rows = []
    for record in records:
        result = record['result']
        proof = record.get('qualification', {})
        paired_ids = proof.get('paired_run_ids', [])
        paired = [identities.get(identity) for identity in paired_ids]
        gates = proof.get('gates', {})
        admitted = (result.get('status') == 'completed' and bool(paired) and
            all(isinstance(gates.get(g), dict) and gates[g].get('passed') is True
                and bool(gates[g].get('evidence_sha256')) for g in GATES) and
            bool(record.get('hardware_id')) and bool(result.get('protocol_sha256')) and
            all(p is not None and p['run_id'] != record['run_id'] and
                p['result'].get('status') == 'completed' and p.get('dataset') == record.get('dataset') and
                p.get('seed') == record.get('seed') and p.get('method') != record.get('method') and
                p.get('hardware_id') == record.get('hardware_id') and
                p['result'].get('protocol_sha256') == result.get('protocol_sha256') for p in paired))
        metrics, timing = result.get('metrics_native', result.get('metrics', {})), result.get('timing', {})
        counter = record.get('arithmetic', {})
        value = counter.get('fp64_add_mul_fma') if counter.get('status') == 'measured_subset' else None
        row = dict(dataset=record.get('dataset'), method=record.get('method'), seed=record.get('seed'),
            protocol_sha256=result.get('protocol_sha256'), hardware_id=record.get('hardware_id'),
            status=result.get('status'), main_table_admitted=bool(admitted),
            metric_scale=result.get('metric_scale', 'standardized'),
            **{k: metrics.get(k) for k in ('rmse', 'nlpd', 'crps', 'ece')},
            **{k: timing.get(k) for k in ('online_seconds', 'seconds_per_query')},
            fp64_add_mul_fma_subset=value, counter_status=counter.get('status', 'unavailable'),
            counter_task_coverage=counter.get('task_coverage'))
        rows.append(row)
    return rows


def export_tables(records, directory):
    rows = table_rows(records)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    def rendered(value):
        if value is None or isinstance(value, float) and not math.isfinite(value):
            return 'NA'
        if isinstance(value, float):
            return f'{value:.8g}'
        return str(value)
    with (directory / 'comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows({key: rendered(row[key]) for key in COLUMNS} for row in rows)
    def latex(value):
        return ''.join({'_': r'\_', '%': r'\%', '&': r'\&', '#': r'\#', '$': r'\$',
            '{': r'\{', '}': r'\}', '\\': r'\textbackslash{}'}.get(c, c) for c in rendered(value))
    columns = ('dataset', 'method', 'seed', 'metric_scale', 'rmse', 'nlpd', 'crps', 'online_seconds',
               'fp64_add_mul_fma_subset', 'main_table_admitted')
    lines = [r'\begin{tabular}{llllrrrrrl}', ' & '.join(latex(c) for c in columns) + r' \\', r'\hline']
    lines.extend(' & '.join(latex(row[c]) for c in columns) + r' \\' for row in rows)
    lines += [r'\end{tabular}', '% FP64 subset counts exclude CPU and other arithmetic; NA is unmeasured.',
              '% Rows with main_table_admitted=False are diagnostic and must not enter the main paper table.']
    (directory / 'comparison.tex').write_text('\n'.join(lines) + '\n')
    return rows
