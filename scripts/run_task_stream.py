#!/usr/bin/env python3
"""Run one fully specified task-stream candidate in its isolated method environment."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--configuration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['integration', 'validation', 'final', 'ablation'], default='integration')
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--arm', choices=['joint_transfer', 'zero_cross', 'identity_transfer'], default='joint_transfer')
    parser.add_argument('--max-tasks', type=int, help='Integration only; never silently truncate a paper run')
    args = parser.parse_args()
    from benchmarks.task_stream.data import PreparedData, file_hash
    from benchmarks.task_stream.factory import Configuration, FeatureTable, FittedTaskAdapter
    from benchmarks.task_stream.ablations import ARMS, matched_manifest
    from benchmarks.task_stream.protocol import TaskStream
    from benchmarks.task_stream.pipeline import run
    from benchmarks.task_stream.provenance import source_identity, feature_identity, source_revision
    import numpy as np
    config = Configuration(**json.loads(args.configuration.read_text()))
    prepared = PreparedData.load(args.prepared)
    selection_identity = prepared.selection_stream.identity()
    config_hash = hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()
    source_commit = source_revision(ROOT)
    source_sha256 = source_identity(ROOT)
    selection_features_sha256 = feature_identity(prepared.selection_stream.times, prepared.selection_features)
    if args.arm != 'joint_transfer' and args.stage != 'ablation':
        parser.error('Ablation arms must be separate from the main comparison')
    if args.stage in ('final', 'ablation'):
        if args.selection is None:
            parser.error('Final runs require a frozen initial-period selection record')
        selected = json.loads(args.selection.read_text())
        if (selected.get('status') != 'selected' or selected.get('selection_protocol_sha256') != selection_identity
                or selected.get('configuration_sha256') != config_hash
                or selected.get('method') != config.method or selected.get('source_commit') != source_commit
                or selected.get('source_sha256') != source_sha256
                or selected.get('selection_features_sha256') != selection_features_sha256):
            raise ValueError('Selection is stale or belongs to another method, configuration, or split')
    elif args.selection is not None:
        parser.error('A selection record is used only for final/ablation runs')
    if args.output.exists() and any(args.output.iterdir()):
        # W&B supervisor may create only its tracking folder before launching us.
        if set(p.name for p in args.output.iterdir()) != {'tracking'}:
            raise FileExistsError('Refusing to overwrite an existing experiment')
    stream = prepared.selection_stream if args.stage == 'validation' else prepared.stream
    values = prepared.selection_features if args.stage == 'validation' else prepared.features
    if args.max_tasks is not None:
        if args.stage != 'integration' or not 1 <= args.max_tasks <= len(stream.bounds):
            parser.error('max-tasks is a bounded integration diagnostic only')
        end = stream.bounds[args.max_tasks-1][1]
        stream = TaskStream(times=stream.times[:end], targets=stream._targets[:end], coordinates=stream.coordinates,
            visible=stream.visible, hidden=stream.hidden, initial_sites=stream.initial_sites,
            initial_steps=stream.initial_steps, task_steps=stream.task_steps,
            release_previous=stream.release_previous, metadata=dict(stream.metadata, integration_tasks=args.max_tasks))
        values = values[:end]
    adapter = FittedTaskAdapter(config, stream.coordinates, stream.visible, FeatureTable(stream.times, values),
        initial_step=float(stream.times[1]-stream.times[0]), release_previous=stream.release_previous,
        arm=next(a for a in ARMS if a.name == args.arm))
    # Every prediction is materialized on host inside the adapter. Explicitly
    # synchronize the entire method device at phase boundaries as well.
    if config.method in ('kronhippo_svgp', 'ohsvgp'):
        import torch
        synchronize = (lambda: torch.cuda.synchronize(config.device)) if config.device.startswith('cuda') else lambda: None
    elif config.method in ('st_svgp', 'mgpvae'):
        import jax
        devices = jax.devices()
        if config.device.startswith('cuda') and not any(d.platform == 'gpu' for d in devices):
            raise RuntimeError('GPU requested but JAX selected CPU')
        if config.device == 'cpu' and any(d.platform != 'cpu' for d in devices):
            raise RuntimeError('Set JAX_PLATFORMS=cpu for the requested CPU run')
        synchronize = jax.effects_barrier
    else:
        import tensorflow as tf
        from stvgp_kronecker.benchmark_runtime import configure_tensorflow
        configure_tensorflow(tf, device=config.device, dtype='float64')
        synchronize = lambda: tf.constant(0.).numpy()
    provenance = dict(source_commit=source_commit, source_sha256=source_sha256, stage=args.stage,
        selection_features_sha256=selection_features_sha256, selection_protocol_sha256=selection_identity,
        configuration_sha256=config_hash,
        input_files={str(p): file_hash(p) for p in [args.configuration, args.prepared/'stream.npz',
             args.prepared/'selection-stream.npz', args.prepared/'features.npz', args.prepared/'manifest.json']})
    result = run(stream, adapter, output=args.output, synchronize=synchronize,
                 provenance=provenance, configuration=asdict(config))
    if args.stage == 'ablation':
        manifest = matched_manifest(protocol_sha256=stream.identity(), fit_sha256=adapter.fit_sha256,
            source_commit=source_commit, model_config=asdict(config))
        manifest.update(source_sha256=source_sha256, executed_arm=args.arm)
        (args.output/'ablation-manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(status=result['status'], output=str(args.output),
                         main_table_admitted=False, metrics=result['metrics'])))


if __name__ == '__main__':
    main()
