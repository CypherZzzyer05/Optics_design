"""Sequential strict-mesh research after the active 5 nm job has finished."""
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

from fdtd_pipeline.config import PROJECT_ROOT
from fdtd_pipeline.pipeline import atomic_json, directory_lock
from compare_double_mesh import read_case


def continue_research():
    state_dir = PROJECT_ROOT/'data/strict_research_workflow'
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir/'status.json'
    state = {'state': 'running', 'steps': [], 'started_at': datetime.now(timezone.utc).isoformat()}
    def step(name, script, *arguments):
        state['current_step'] = name
        state['updated_at'] = datetime.now(timezone.utc).isoformat()
        atomic_json(state_path, state)
        print(f'Workflow: {name}', flush=True)
        subprocess.run([sys.executable, str(PROJECT_ROOT/script), *map(str, arguments)], cwd=PROJECT_ROOT, check=True)
        state['steps'].append({'name': name, 'state': 'completed'})
        atomic_json(state_path, state)
    with directory_lock(state_dir):
        try:
            # This validates successful state, checksum, actual meshes and the
            # full-spatial power check; a stale running manifest is insufficient.
            read_case(PROJECT_ROOT/'data/double_dip_fine5_checked', 'Au_full_difference')
            step('native_audit_5nm', 'audit_double_native.py', '--directory', 'data/double_dip_fine5_checked')
            step('archive_5nm', 'export_double_results.py', '--directory', 'data/double_dip_fine5_checked',
                 '--output', 'research_results/double_dip_fine5_checked')
            step('compare_both_troughs_8nm_5nm', 'compare_double_mesh.py')
            comparison = json.loads((PROJECT_ROOT/'research_results/double_dip_mesh_comparison/comparison.json').read_text(encoding='utf-8'))
            if not comparison['both_dips_stable']:
                raise RuntimeError('Both troughs did not pass the 8/5 nm stability thresholds; inspect before scheduling further work')
            step('complete_six_8nm_Au_Ag_angle_controls', 'explore_double_dips.py', 'run')
            manifest = json.loads((PROJECT_ROOT/'data/double_dip_controls8/manifest.json').read_text(encoding='utf-8'))
            for entry in manifest['cases']:
                step('native_audit_'+entry['id'], 'audit_double_native.py', '--case-id', entry['id'])
            step('archive_six_8nm_controls', 'export_double_results.py')
            step('track_both_troughs_across_angle_and_metal_controls', 'analyze_angle_controls.py')
            step('generate_twelve_real_8nm_seven_parameter_samples', 'generate_dataset.py', 'generate',
                 '--config', 'configs/as2se3_au_fine8.json')
            sampling_args = ('--sampling-config', 'configs/as2se3_au_fine8.json',
                             '--training-config', 'configs/train_as2se3_fine8.json')
            step('native_audit_8nm_random_pilot', 'audit_quadrant_pilot.py', *sampling_args, '--native-only')
            latest_path = PROJECT_ROOT/'runs/as2se3_au_fine8/latest_run.json'
            verified_existing = False
            if latest_path.exists():
                # Reuse training only if the current dataset/checkpoint/CSV audit passes.
                try:
                    step('audit_existing_8nm_training', 'audit_quadrant_pilot.py', *sampling_args)
                    verified_existing = True
                except subprocess.CalledProcessError:
                    pass
            if not verified_existing:
                step('train_seven_output_8nm_pilot', 'train_metaformer.py', '--config', 'configs/train_as2se3_fine8.json')
                step('audit_seven_output_8nm_training', 'audit_quadrant_pilot.py', *sampling_args)
            step('archive_8nm_real_data_and_training', 'export_pilot_results.py', *sampling_args,
                 '--output', 'research_results/seven_parameter_fine8')
            state.update(state='completed', current_step=None, finished_at=datetime.now(timezone.utc).isoformat())
            atomic_json(state_path, state)
        except Exception as error:
            state.update(state='needs_attention', error=str(error), updated_at=datetime.now(timezone.utc).isoformat())
            atomic_json(state_path, state)
            raise


if __name__ == '__main__':
    continue_research()
