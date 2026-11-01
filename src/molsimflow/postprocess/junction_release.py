"""Incremental chemical-junction release screening from explicit trajectory contracts."""
from __future__ import annotations
import argparse
import csv
import datetime
import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
from molsimflow.io.lammps_dump import iter_lammps_dump_records_selected
from molsimflow.postprocess.trajectory_feature_cache import FeatureCache
from molsimflow.postprocess.junction_release_metrics import frame_metrics
from molsimflow.postprocess.junction_release_events import assemble_series


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_check(source: dict) -> None:
    path = Path(source['path']).resolve(strict=True)
    stat = path.stat()
    if stat.st_ino != source['inode'] or stat.st_size < source['admitted_size']:
        raise ValueError(f"Trajectory was replaced or truncated: {path}")
    with path.open('rb') as stream:
        head = hashlib.sha256(stream.read(source['head_bytes'])).hexdigest()
    if head != source['head_sha256']:
        raise ValueError(f"Trajectory prefix changed: {path}")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError('No rows to export')
    fields = [k for k in rows[0] if not k.startswith('_')]
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def _plot(rows: list[dict], path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True, layout='constrained')
    time = [r['global_ns'] for r in rows]
    epochs = sorted({r['continuity_epoch'] for r in rows})
    for epoch in epochs:
        selected = [r for r in rows if r['continuity_epoch'] == epoch]
        axes[0].plot([r['global_ns'] for r in selected],
                     [r['dx_within_geometry_segment_A']/10 for r in selected], color='#2772a4')
    axes[0].set_ylabel('Bubble dx (nm)')
    for group, label in [('top_left', 'Upper left'), ('top_right', 'Upper right'),
                         ('bottom_left', 'Lower left'), ('bottom_right', 'Lower right')]:
        axes[1].plot(time, [r[group+'_contacts_4p0'] for r in rows], label=label)
        axes[2].plot(time, [r[group+'_site_area_4p0_A2']/100 for r in rows], label=label)
    axes[1].legend(ncol=2, fontsize=8); axes[1].set_ylabel('Contacting N2')
    axes[2].set_ylabel('Site contact area\nproxy (nm2)')
    axes[3].plot(time, [r['min_bubble_surface_distance_A'] for r in rows], label='Any-face minimum gap')
    axes[3].plot(time, [r['bubble_height_q05_q95_A'] for r in rows], label='Bubble height (q05-q95)')
    axes[3].legend(fontsize=8); axes[3].set_ylabel('Distance (A)')
    axes[4].plot(time, [r['largest_cluster_n2_count'] for r in rows]); axes[4].set_ylabel('Main cluster N2')
    axes[4].set_xlabel('Global time (ns)')
    for ax in axes:
        ax.grid(alpha=.2)
        for r in rows:
            if r.get('reseed_boundary'): ax.axvline(r['global_ns'], color='#b94545', ls='--', lw=.8)
    fig.suptitle(title+' | geometric screening; scientific acceptance pending')
    fig.savefig(path, dpi=150); plt.close(fig)


def run_contract(contract_path: Path, output: Path) -> dict:
    contract = json.loads(contract_path.read_text())
    if contract['schema_version'] != 1:
        raise ValueError('Unsupported contract version')
    output.mkdir(parents=True, exist_ok=True)
    with (output/'analysis.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ref_path = Path(contract['reference_npz'])
        if digest(ref_path) != contract['reference_sha256']:
            raise ValueError('Reference hash mismatch')
        with np.load(ref_path, allow_pickle=False) as stored:
            reference = {k: stored[k] for k in stored.files}
        cache = FeatureCache(output/'cache.sqlite', contract['analysis_key'])
        computed = 0; reused = 0; verified = 0; raw = []; coverage = []
        try:
            for source in contract['sources']:
                source_check(source)
                low, high = source['minimum_step'], source['maximum_step']
                expected = list(range(low, high+1, contract['parameters']['stride_steps']))
                if not expected:
                    raise ValueError('Empty source interval')
                previous = cache.steps(source['source_id'])
                needed = set(expected)-previous
                boundary_check = {max(set(expected)&previous)} if set(expected)&previous else set()
                accepted = needed | boundary_check
                if accepted:
                    reader = iter_lammps_dump_records_selected(
                        Path(source['path']).resolve(strict=True),
                        accept_timestep=lambda step: step in accepted,
                        stop_after_timestep=high)
                    observed = set()
                    try:
                        for frame in reader:
                            fingerprint = hashlib.sha256(
                                (str(frame.timestep)+repr(frame.atom_fields)+repr(frame.bounds.tolist())+
                                 '\n'.join(' '.join(r) for r in frame.atom_rows)).encode()).hexdigest()
                            metrics = frame_metrics(frame, reference, contract['parameters'], Path(source['path']))
                            metrics.update(source_id=source['source_id'], source_segment=source['segment_id'])
                            cache.put(source['source_id'], frame.timestep, fingerprint, metrics)
                            observed.add(frame.timestep)
                            computed += int(frame.timestep in needed)
                            verified += int(frame.timestep in boundary_check)
                            if frame.timestep >= max(accepted): break
                    finally:
                        reader.close()
                    if observed != accepted:
                        raise ValueError(f"Missing complete frames: {sorted(accepted-observed)[:8]}")
                source_check(source)
                current = cache.load(source['source_id'], expected)
                reused += len(expected)-len(needed)
                for row in current:
                    row['reseed_boundary'] = bool(source['reseed'] and row['step']==source['native_start_step'])
                raw.extend(current)
                coverage.append(dict(source_id=source['source_id'], first_step=low, last_step=high,
                                     frames=len(current), source_path=source['path'],
                                     append_only_provenance=source))
        finally:
            cache.close()
        steps = [r['step'] for r in raw]
        if contract['mode'] == 'production' and steps != list(range(
                contract['first_step'], contract['last_step']+1, contract['parameters']['stride_steps'])):
            raise ValueError('Lineage frame coverage is not contiguous and unique')
        rows, events = assemble_series(raw, contract['parameters'])
        staging = output/f'.publish-{os.getpid()}'
        staging.mkdir(exist_ok=False)
        _write_csv(staging/'timeseries.csv', rows)
        _write_csv(staging/'episodes.csv', events)
        _plot(rows, staging/'overview.png', contract['label'])
        screening = [e for e in events if e['state']=='DETACHMENT_CANDIDATE_GEOMETRY_ONLY' and e['persistent_screen']]
        summary = dict(status='PASS', mode=contract['mode'], case_id=contract['case_id'], label=contract['label'],
                       processed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       contract_sha256=digest(contract_path), analysis_key=contract['analysis_key'],
                       frames=len(rows), computed_frames=computed, reused_frames=reused,
                       revalidated_cached_boundary_frames=verified,
                       first_global_ns=rows[0]['global_ns'], last_global_ns=rows[-1]['global_ns'],
                       initial={k:v for k,v in rows[0].items() if not k.startswith('_')},
                       final={k:v for k,v in rows[-1].items() if not k.startswith('_')},
                       source_coverage=coverage, reseed_steps=[r['step'] for r in rows if r['reseed_boundary']],
                       persistent_geometric_detachment_candidates=screening,
                       scientific_acceptance='UNREVIEWED', species_context=contract.get('species_context', {}),
                       scope='Complete saved-frame geometric screen. Site area is a reference-plane proxy; water neighbors are not a water film. Cached history assumes append-only source identity, with head and cached-boundary frame revalidation. No full-file trajectory-integrity or causality claim.',
                       artifact_sha256={p.name:digest(p) for p in staging.iterdir()})
        (staging/'SUMMARY.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
        current_dir = output/'current'; current_dir.mkdir(exist_ok=True)
        if (current_dir/'SUMMARY.json').exists():
            with (output/'history.jsonl').open('a') as stream:
                stream.write(json.dumps(json.loads((current_dir/'SUMMARY.json').read_text()))+'\n')
        for path in sorted(staging.iterdir(), key=lambda p:p.name=='SUMMARY.json'):
            os.replace(path, current_dir/path.name)
        staging.rmdir()
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run_contract(args.contract, args.output)
    print(json.dumps({k:report[k] for k in ['status','case_id','frames','computed_frames','reused_frames','first_global_ns','last_global_ns']}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
