"""Start one real HTTP MCP server and verify discovery, a data call and all returned files."""
import argparse,asyncio,hashlib,json,os,socket,subprocess,sys,tempfile,time
from pathlib import Path
import httpx
from fastmcp import Client

async def check(study):
    specifications={'heracleum':(8,'heracleum2026_dataset_audit'), 'mannich':(10,'mannich_dataset_overview'), 'cannabis':(9,'cannabis2026_dataset_audit')}
    count,tool=specifications[study]
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
    base=f'http://127.0.0.1:{port}'
    with tempfile.TemporaryDirectory(prefix='chem-mcp-smoke-') as directory:
        environment=os.environ.copy();environment.update(CHEM_REPRO_PUBLIC_BASE_URL=base,CHEM_REPRO_ARTIFACTS_DIR=str(Path(directory)/'files'),PYTHON_DOTENV_DISABLED='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MPLBACKEND='Agg')
        with open(Path(directory)/'server.log','w') as log:
            process=subprocess.Popen([sys.executable,'-m',f'chem_repro.{study}.server','--port',str(port)],cwd=directory,env=environment,stdout=log,stderr=subprocess.STDOUT)
            try:
                async with httpx.AsyncClient(timeout=2) as http:
                    for _ in range(360):
                        if process.poll() is not None:raise RuntimeError(Path(directory,'server.log').read_text())
                        try:
                            await http.get(base+'/mcp');break
                        except httpx.TransportError:await asyncio.sleep(.5)
                    else:raise RuntimeError('Server startup timed out: '+Path(directory,'server.log').read_text())
                async with Client(base+'/mcp',timeout=120) as client:
                    declarations=await client.list_tools();assert len(declarations)==count
                    assert all(t.outputSchema and t.inputSchema.get('x-data-contract') for t in declarations)
                    result=await client.call_tool(tool,{});assert not result.is_error
                    data=result.structured_content;assert data['metadata']['study_id']==study+'_2026'
                    metrics={m['name']:m['value'] for m in data['answer']['metrics']}
                    if study=='heracleum':assert metrics['ds1']==2008 and metrics['ds2']==102
                    if study=='mannich':assert metrics['reaction_records']==4015
                async with httpx.AsyncClient(timeout=60) as http:
                    for artifact in data['artifacts']:
                        response=await http.get(artifact['download_url']);response.raise_for_status()
                        assert len(response.content)==artifact['size_bytes']
                        assert hashlib.sha256(response.content).hexdigest()==artifact['sha256']
                return {'study':study,'tool_count':count,'tool':tool,'status':data['status'],'verified_downloads':len(data['artifacts']),'scope':'Actual HTTP MCP smoke check; not a complete paper or CoScientist E2E acceptance'}
            finally:
                process.terminate()
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:process.kill();process.wait()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--study',choices=['heracleum','mannich','cannabis'],required=True);args=parser.parse_args()
    print(json.dumps(asyncio.run(check(args.study)),ensure_ascii=False))
