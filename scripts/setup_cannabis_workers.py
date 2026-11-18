"""Install or verify the two isolated scientific workers; never starts model training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--chemprop-python', type=Path)
    parser.add_argument('--tox-python', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import chem_repro
    suite = Path(chem_repro.__file__).resolve().parent.parent
    root = args.directory.expanduser().resolve()
    specs = [
        ('chemprop', 'CHEM_REPRO_CHEMPROP_PYTHON', 'cannabis_dmpnn', args.chemprop_python),
        ('open-tox', 'CHEM_REPRO_TOX_PYTHON', 'cannabis_open_tox', args.tox_python),
    ]
    verified, configuration = {}, {}
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)
    environment.pop('PYTHONHOME', None)
    environment['PYTHON_DOTENV_DISABLED'] = '1'
    with tempfile.TemporaryDirectory(prefix='chem-worker-setup-') as temporary:
        for name, variable, dataset, override in specs:
            python = override.expanduser().absolute() if override else root / name / 'bin/python'
            requirements = suite / 'chem_repro/data' / dataset / 'worker_requirements.txt'
            if not python.exists():
                if args.check_only or override:
                    raise RuntimeError(f'Worker Python missing: {python}')
                subprocess.run(['uv', 'venv', str(root / name), '--python', '3.12'], cwd=temporary,
                               env=environment, check=True)
                subprocess.run(['uv', 'pip', 'install', '--python', str(python), '-r', str(requirements)],
                               cwd=temporary, env=environment, check=True)
            # Existing environments are inspected, never silently upgraded or overwritten.
            if dataset == 'cannabis_dmpnn':
                expected = json.loads((requirements.parent / 'scientific_versions.json').read_text())
            else:
                expected = dict(line.split('==', 1) for line in requirements.read_text().splitlines() if line)
            probe = ('import json; from importlib.metadata import version; '
                     f'print(json.dumps({{n:version(n) for n in {list(expected)!r}}}))')
            actual = json.loads(subprocess.check_output([str(python), '-c', probe], cwd=temporary,
                                                       env=environment, text=True, timeout=60))
            if actual != expected:
                raise RuntimeError(f'{name} differs from pinned scientific versions; existing environment unchanged')
            verified[name] = actual
            configuration[variable] = str(python)
    for variable, folder in {
        'CHEM_REPRO_STRUCTURE_RESULTS_DIR': 'dmpnn',
        'CHEM_REPRO_FUSION_RESULTS_DIR': 'fusion',
        'CHEM_REPRO_RTE_RESULTS_DIR': 'rte',
        'CHEM_REPRO_OPEN_TOX_RESULTS_DIR': 'open_tox',
        'CHEM_REPRO_TOX_PROFILE_RESULTS_DIR': 'toxicity_profiles',
        'CHEM_REPRO_ARTIFACTS_DIR': 'artifacts',
    }.items():
        configuration[variable] = str(root / 'results' / folder)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'environment': configuration, 'verified_software': verified,
                                      'training_started': False, 'nested_v2_configured': False}, indent=2) + '\n')
    print(f'Worker versions verified; absolute-path configuration written to {args.output}')


if __name__ == '__main__':
    main()
