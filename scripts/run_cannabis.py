"""Start Cannabis MCP with a reviewed absolute-path worker configuration."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configuration', type=Path, required=True)
    parser.add_argument('--public-base-url', default='http://127.0.0.1:7341')
    args, server_args = parser.parse_known_args()
    configuration = json.loads(args.configuration.read_text())['environment']
    for key, value in configuration.items():
        if not key.startswith('CHEM_REPRO_') or not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError('Worker configuration must contain CHEM_REPRO_* absolute paths only')
    environment = os.environ.copy()
    environment.update(configuration)
    environment['CHEM_REPRO_PUBLIC_BASE_URL'] = args.public_base_url.rstrip('/')
    os.execve(sys.executable, [sys.executable, '-m', 'chem_repro.cannabis.server', *server_args], environment)


if __name__ == '__main__':
    main()
