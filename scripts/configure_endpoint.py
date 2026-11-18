"""Build CoScientist registration and server environment for an operator-supplied address."""
import argparse
import ipaddress
import json
from pathlib import Path
from urllib.parse import urlsplit


def deployment(base_url, spec, allow_local=False, allow_http=False):
    parsed = urlsplit(base_url.strip())
    if parsed.scheme not in {'http','https'} or not parsed.hostname:
        raise ValueError('Supply an absolute HTTP(S) base URL allocated to this deployment')
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('The base URL must not contain credentials, a query, or a fragment')
    if any(c.isspace() for c in base_url) or any(c in base_url for c in '<>{}'):
        raise ValueError('Replace placeholder text with the actual deployment address')
    _ = parsed.port  # Validate a supplied numeric port before emitting configuration.
    host = parsed.hostname.lower()
    if host.endswith(('.invalid','.example')) or host in {'example.com','example.org','example.net'} or host.endswith(('.example.com','.example.org','.example.net')):
        raise ValueError('Documentation example domains are not a deployment address')
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host == 'localhost' or host.endswith('.localhost')
    if loopback and not allow_local:
        raise ValueError('Loopback is local development only; pass --allow-local explicitly')
    if parsed.scheme != 'https' and not (allow_http or (loopback and allow_local)):
        raise ValueError('Use HTTPS for a shared deployment, or explicitly select --allow-http for an internal network')
    base = base_url.strip().rstrip('/')
    if parsed.path.rstrip('/').endswith('/mcp'):
        raise ValueError('Supply the base URL without the /mcp suffix')
    registration = [{'name':spec['registry_name'],'url':base+'/mcp','description':spec['description']}]
    environment = f'CHEM_REPRO_PUBLIC_BASE_URL={base}\nCHEM_REPRO_HOST=0.0.0.0\nCHEM_REPRO_PORT={spec["default_port"]}\n'
    return registration, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',required=True)
    parser.add_argument('--output-dir',type=Path,default=Path('deployment'))
    parser.add_argument('--allow-local',action='store_true')
    parser.add_argument('--allow-http',action='store_true')
    args = parser.parse_args()
    spec = json.loads((Path(__file__).resolve().parents[1]/'server.json').read_text())
    try:
        registration, environment = deployment(args.base_url,spec,args.allow_local,args.allow_http)
    except ValueError as error:
        parser.error(str(error))
    paths = [args.output_dir/'registration.json',args.output_dir/'server.env']
    if any(path.exists() for path in paths):
        parser.error('Output files already exist; select a new --output-dir to preserve the existing configuration')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    paths[0].write_text(json.dumps(registration,ensure_ascii=False,indent=2)+'\n')
    paths[1].write_text(environment)
    print(json.dumps({'registration':str(paths[0]),'environment':str(paths[1]),
                      'endpoint':registration[0]['url'],'deployed':False},ensure_ascii=False))


if __name__ == '__main__':
    main()
