#!/usr/bin/env python3
"""Declare equal-time validation opportunities and collect complete paired studies."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.task_stream.factory import Configuration, METHODS
from scripts.run_task_budget_campaign import atomic_json, sha256, verified_completed, canonical_configuration


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def policy():
    return dict(schema_version=1, name='common-initial-optimizer-time-v1',
        split_seeds=dict(era5=list(range(5)), covid=list(range(5, 10)), pems=[1, 2, 3]),
        methods=list(METHODS), seconds=[60., 180., 540.], initial_iteration_safety_cap=1000000,
        learning_rate=.001, online_iterations=5, model_capacity='fixed tested base capacity; no capacity search',
        criterion='query_weighted_validation_observation_nlpd',
        candidate_count_per_method_split=3, early_stopping=False,
        stopping='first completed synchronized update reaching declared time; at least one update',
        scope='initial_optimizer_setup_compilation_and_completed_updates',
        excluded_from_equal_time_cap=['shared feature/ridge preparation', 'initial filtering/posterior construction',
            'validation updates and prediction', 'scoring', 'serialization and W&B'],
        validation='entire initial-period selection stream; no final-stream scoring',
        randomization='same split-derived training seed for all budget candidates; wall clock can change actual steps',
        convergence_claimed=False, stability_diagnostic='absolute last-two NLPD difference <= 0.01; descriptive only',
        failure_policy='retain failures; incomplete group has no selected configuration; no baseline omission',
        refit='freeze selected actual steps; OH preserves expected row exposures on the larger initial dataset',
        main_table_admitted=False)


def build(prepared_root, templates, compute_root, campaign):
    import numpy as np
    rule = policy(); rule_hash = digest(rule); tasks = []
    # Interleave methods/domains in the first tier so every implementation is
    # exercised early; modulo-three workers retain a deterministic assignment.
    for seconds in rule['seconds']:
        for dataset, seeds in rule['split_seeds'].items():
            for seed in seeds:
                local = Path(prepared_root)/dataset/f'seed{seed}'
                counts = []
                for filename in ['selection-stream.npz', 'stream.npz']:
                    with np.load(local/filename, allow_pickle=False) as arrays:
                        counts.append(int(arrays['initial_steps'])*len(arrays['initial_sites']))
                for method in METHODS:
                    c = json.loads((Path(templates)/dataset/f'{method}.json').read_text())
                    c.update(initial_iterations=rule['initial_iteration_safety_cap'], initial_max_seconds=seconds,
                        initial_expected_passes=None, learning_rate=rule['learning_rate'],
                        online_iterations=rule['online_iterations'], seed=seed, device='cuda')
                    configuration = asdict(Configuration(**c))
                    name = f'{dataset}-seed{seed}-{method}-{int(seconds)}s'
                    tasks.append(dict(id=name, dataset=dataset, split_seed=seed, configuration=configuration,
                        prepared=str(Path(compute_root)/'protocol/task-stream-20260930'/dataset/f'seed{seed}'),
                        output=str(Path(compute_root)/'studies'/campaign/'candidates'/name), stage='validation',
                        study=dict(policy=rule, policy_sha256=rule_hash,
                            selection_initial_rows=counts[0], final_initial_rows=counts[1])))
    validate_manifest(tasks, rule)
    return tasks, rule


def validate_manifest(tasks, rule):
    if rule != policy():
        raise ValueError('Study differs from the declared common-time policy')
    groups = defaultdict(list)
    for task in tasks:
        if task.get('stage') != 'validation' or task['study']['policy_sha256'] != digest(rule):
            raise ValueError('Study identity or validation boundary changed')
        if task['study'].get('policy') != rule:
            raise ValueError('Embedded study policy changed')
        c = task['configuration']
        key = (task['dataset'], task['split_seed'], c['method'])
        if (c['initial_iterations'] != rule['initial_iteration_safety_cap']
                or c.get('initial_expected_passes') is not None or c['learning_rate'] != rule['learning_rate']
                or c['online_iterations'] != rule['online_iterations'] or c['seed'] != task['split_seed']):
            raise ValueError('Candidate breaks the shared declared settings')
        groups[key].append(task)
    expected = {(d, s, m) for d, seeds in rule['split_seeds'].items() for s in seeds for m in METHODS}
    if set(groups) != expected:
        raise ValueError('Missing or unexpected dataset/split/method group')
    for group in groups.values():
        if sorted(t['configuration'].get('initial_max_seconds') for t in group) != rule['seconds']:
            raise ValueError('Every group requires all three distinct common budgets')
        reference = None
        for task in group:
            c = dict(task['configuration']); c.pop('initial_max_seconds')
            if reference is not None and c != reference:
                raise ValueError('Capacity or other configuration changed across budget candidates')
            reference = c
    return groups


def collect(tasks, rule, output):
    from scripts.select_task_configuration import select
    from benchmarks.task_stream.refit_budget import frozen_refit_plan
    groups = validate_manifest(tasks, rule)
    summary = dict(policy_sha256=digest(rule), groups=[], complete=False, main_table_admitted=False)
    for (dataset, seed, method), group in groups.items():
        row = dict(dataset=dataset, split_seed=seed, method=method, status='pending', candidates=[])
        paths = []
        for task in group:
            latest = Path(task['output'])/'latest.json'
            if not latest.exists():
                row['candidates'].append(dict(id=task['id'], status='pending')); continue
            terminal = json.loads(latest.read_text()); attempt = Path(terminal['attempt'])
            _, expected_hash = canonical_configuration(task['configuration'])
            valid = (terminal['status'] == 'completed'
                and terminal['proof']['configuration_sha256'] == expected_hash
                and verified_completed(attempt, terminal['proof']))
            row['candidates'].append(dict(id=task['id'], status='completed' if valid else 'failed', attempt=str(attempt)))
            if valid:
                paths.append(attempt/'run')
        if len(paths) == len(group):
            selected = select(paths)
            winner = Path(selected['winner']['path'])
            config = json.loads((winner/'configuration.json').read_text())['configuration']
            record = json.loads((winner/'fit-budget.json').read_text())
            artifacts = json.loads((winner/'artifacts.json').read_text())
            if sha256(winner/'fit-budget.json') != artifacts['fit-budget.json']['sha256']:
                raise ValueError('Selected fit-budget artifact changed')
            selected['refit_budget'] = frozen_refit_plan(config, record,
                group[0]['study']['selection_initial_rows'], group[0]['study']['final_initial_rows'])
            selected['study_policy_sha256'] = digest(rule)
            selected['all_declared_candidates_completed'] = True
            ladder = sorted([(json.loads((p/'configuration.json').read_text())['configuration']['initial_max_seconds'],
                json.loads((p/'result.json').read_text())['metrics']['nlpd']) for p in paths])
            selected['budget_stability_diagnostic'] = dict(last_two_absolute_nlpd_difference=abs(ladder[-1][1]-ladder[-2][1]),
                threshold=.01, passed=abs(ladder[-1][1]-ladder[-2][1]) <= .01, convergence_claimed=False)
            destination = Path(output)/'selections'/f'{dataset}-seed{seed}-{method}.json'
            atomic_json(destination, selected)
            row.update(status='selected_within_declared_budget', selection=str(destination), ladder=ladder,
                       selected_max_seconds=config['initial_max_seconds'])
        elif any(c['status']=='failed' for c in row['candidates']):
            row['status']='failed_candidates_preserved'
        summary['groups'].append(row)
    summary['complete'] = all(r['status']=='selected_within_declared_budget' for r in summary['groups'])
    atomic_json(Path(output)/'study-summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    b = sub.add_parser('build')
    for name in ['prepared-root', 'templates', 'compute-root', 'campaign', 'output']:
        b.add_argument('--'+name, required=True)
    c = sub.add_parser('collect')
    for name in ['manifest', 'policy', 'output']:
        c.add_argument('--'+name, required=True)
    args = parser.parse_args()
    if args.action == 'build':
        tasks, rule = build(args.prepared_root, args.templates, args.compute_root, args.campaign)
        atomic_json(Path(args.output)/'manifest.json', tasks)
        atomic_json(Path(args.output)/'policy.json', rule)
        print(json.dumps(dict(candidates=len(tasks), policy_sha256=digest(rule))))
    else:
        import fcntl
        Path(args.output).mkdir(parents=True, exist_ok=True)
        with (Path(args.output)/'collector.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result = collect(json.loads(Path(args.manifest).read_text()), json.loads(Path(args.policy).read_text()), args.output)
        print(json.dumps(dict(complete=result['complete'], groups=len(result['groups']))))


if __name__ == '__main__':
    main()
