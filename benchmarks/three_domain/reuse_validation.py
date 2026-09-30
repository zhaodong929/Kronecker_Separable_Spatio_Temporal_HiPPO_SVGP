"""Reuse an immutable, synchronized validation result without changing its lineage."""
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def original_validation_directory(previous, result_name):
    """Follow verified copies to their original tracked attempt, checking each copy."""
    previous = Path(previous)
    seen = set()
    while (previous/'reused-validation.json').exists():
        identity = previous.resolve()
        if identity in seen:
            raise ValueError('Cached validation provenance cycle')
        seen.add(identity)
        record = json.loads((previous/'reused-validation.json').read_text())
        if (record.get('status') != 'verified_validation_reused'
                or digest(previous/result_name) != record['result_sha256']):
            raise ValueError('Cached validation copy changed')
        original = Path(record['source_directory'])
        if digest(original/result_name) != record['result_sha256']:
            raise ValueError('Cached validation source changed')
        previous = original
    return previous


def reuse_ohsvgp(previous, destination, spec, compute_root, source_root, result_name):
    dependencies = [Path('scripts/run_covid_ohsvgp_own_theta.py'), Path('scripts/run_traffic_ohsvgp.py'),
                    Path('scripts/run_ohsvgp_cached_prediction.py'), Path('baselines/ohsvgp_prediction.py'),
                    Path('baselines/traffic_protocol_n.py'), Path('benchmarks/three_domain/geometry.py'),
                    Path('benchmarks/three_domain/tracking.py'),Path('baselines/era5_protocol.py'),
                    Path('baselines/covid_long_setting_b/archive.py'),Path('baselines/covid_long_setting_b/protocol.py'),
                    Path('scripts/run_official_ohsvgp_era5.py'),Path('scripts/era5_ncu_ranges.py')]
    for folder in ['stvgp_kronecker',
                   'baselines/external/harrisonzhu508_HIPPOSVGP/hipposvgp']:
        dependencies += [p.relative_to(source_root) for p in (Path(source_root)/folder).rglob('*.py')]
    return reuse_checked(previous,destination,spec,compute_root,source_root,result_name,dependencies)


def reuse_routeb(previous,destination,spec,compute_root,source_root):
    dependencies=[Path('scripts')/name for name in [
        'run_iclr_era5_routeb_batch.py','run_routeb_batch_empirical_bayes.py',
        'run_hipposvgp_era5_routeb.py','run_iclr_era5_routeb_strict_online.py','era5_ncu_ranges.py']]
    dependencies.append(Path('benchmarks/three_domain/tracking.py'))
    dependencies += [p.relative_to(source_root) for p in (Path(source_root)/'stvgp_kronecker').rglob('*.py')]
    return reuse_checked(previous,destination,spec,compute_root,source_root,'result.json',dependencies)


def reuse_checked(previous,destination,spec,compute_root,source_root,result_name,dependencies):
    previous, destination = Path(previous), Path(destination)
    if not (previous/result_name).is_file():
        return None
    previous = original_validation_directory(previous, result_name)
    old_spec = json.loads((previous/'spec.json').read_text())
    ignored = {'source_commit', 'input_files'}
    if {k:v for k,v in old_spec.items() if k not in ignored} != {k:v for k,v in spec.items() if k not in ignored}:
        raise ValueError('Cached validation configuration changed')
    if spec['stage'] != 'validation':
        raise ValueError('Only validation results may be reused')
    old_source = Path(compute_root)/'releases'/old_spec['source_commit']/'source'
    for relative in dependencies:
        if digest(old_source/relative) != digest(Path(source_root)/relative):
            raise ValueError('Cached validation numerical source changed: '+str(relative))
    attempts=[]
    for marker in (previous/'tracking').glob('*/terminal.json'):
        terminal=json.loads(marker.read_text())
        if terminal.get('exit_code') == 0 and terminal.get('status') == 'process_complete' and not terminal.get('tracking_sync_required'):
            attempts.append(marker.parent)
    if len(attempts) != 1:
        raise ValueError('Exactly one successful cached validation attempt is required')
    attempt=attempts[0]
    provenance=json.loads((attempt/'provenance.json').read_text())
    if {Path(k).name:v for k,v in provenance['input_files'].items()} != {Path(k).name:digest(k) for k in spec['input_files']}:
        raise ValueError('Cached validation inputs changed')
    manifest=json.loads((attempt/'artifact_manifest.json').read_text())
    if manifest[result_name]['sha256'] != digest(previous/result_name):
        raise ValueError('Cached validation result hash changed')
    import wandb
    identity=json.loads((attempt/'wandb.json').read_text())
    run=wandb.Api(timeout=30).run(f"harrisonzhu/KronHiPPO-STGP/{identity['id']}")
    if run.state != 'finished' or not list(run.logged_artifacts()):
        raise ValueError('Cached validation has no finished team run and artifact')
    record=dict(status='verified_validation_reused',source_directory=str(previous),
                source_commit=old_spec['source_commit'],source_run=run.url,
                result_sha256=digest(previous/result_name),numerical_source_files=len(dependencies),
                inputs_and_configuration_verified=True,main_table_admitted=False)
    destination.mkdir(parents=True,exist_ok=True)
    (destination/result_name).write_bytes((previous/result_name).read_bytes())
    (destination/'reused-validation.json').write_text(json.dumps(record,indent=2)+'\n')
    return json.loads((previous/result_name).read_text())


def reuse_st_svgp(previous,destination,spec,compute_root,source_root):
    dependencies=[Path(p) for p in [
        'baselines/covid_long_setting_b/adapters/run_st_svgp.py',
        'baselines/covid_long_setting_b/archive.py','baselines/covid_long_setting_b/protocol.py',
        'baselines/st_svgp_filter.py','baselines/bayesnewton_compat.py','baselines/causal_mean.py',
        'baselines/traffic_protocol_n.py','baselines/era5_protocol.py',
        'benchmarks/three_domain/geometry.py','benchmarks/three_domain/tracking.py']]
    return reuse_checked(previous,destination,spec,compute_root,source_root,'task1_validation.json',dependencies)
