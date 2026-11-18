"""Verify an already running MCP endpoint and its downloadable scientific artifacts."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit
import httpx
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport


async def check(url, timeout, headers):
    root = Path(__file__).resolve().parents[1]
    spec = json.loads((root/'server.json').read_text())
    async with Client(StreamableHttpTransport(url,headers=headers,sse_read_timeout=timeout), timeout=timeout) as client:
        declarations = await client.list_tools()
        actual = {tool.name for tool in declarations}
        expected = set(spec['tool_names'])
        if actual != expected:
            raise RuntimeError(f'Tool inventory mismatch: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}')
        response = await client.call_tool(spec['smoke_tool'],{})
        if response.is_error:
            raise RuntimeError('MCP returned a tool error for the dataset audit')
        result = response.structured_content
    if result['metadata']['study_id'] != spec['dataset_scope']:
        raise RuntimeError('Unexpected study_id in the scientific result')
    if result['status']=='unsupported' or not result['artifacts']:
        raise RuntimeError('Dataset audit is unavailable or returned no downloadable evidence')
    metrics={metric['name']:metric['value'] for metric in result['answer']['metrics']}
    if any(metrics.get(name)!=value for name,value in spec['smoke_metrics'].items()):
        raise RuntimeError('Dataset counts differ from the declared frozen source')
    origin = urlsplit(url)
    checked=[]
    async with httpx.AsyncClient(timeout=timeout,headers=headers) as http:
        for artifact in result['artifacts']:
            target = urlsplit(artifact['download_url'])
            if (target.scheme,target.netloc)!=(origin.scheme,origin.netloc):
                raise RuntimeError('Artifact address differs from the MCP origin; check CHEM_REPRO_PUBLIC_BASE_URL and the gateway')
            response = await http.get(artifact['download_url'])
            response.raise_for_status()
            if len(response.content)!=artifact['size_bytes'] or hashlib.sha256(response.content).hexdigest()!=artifact['sha256']:
                raise RuntimeError('Artifact size or SHA256 mismatch: '+artifact['name'])
            checked.append(artifact['name'])
    return {'endpoint':url,'tool_count':len(actual),'tool':spec['smoke_tool'],
            'scientific_status':result['status'],'verified_downloads':checked,
            'scope':'MCP discovery, dataset audit and artifact delivery; not full paper or CoScientist acceptance'}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',required=True)
    parser.add_argument('--timeout',type=float,default=180)
    parser.add_argument('--headers-file',type=Path,help='Optional local JSON object of gateway request headers; keep it outside Git')
    args=parser.parse_args()
    headers=json.loads(args.headers_file.read_text()) if args.headers_file else {}
    if not isinstance(headers,dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in headers.items()):
        parser.error('Headers must be a JSON object of strings')
    print(json.dumps(asyncio.run(check(args.url,args.timeout,headers)),ensure_ascii=False,indent=2))
