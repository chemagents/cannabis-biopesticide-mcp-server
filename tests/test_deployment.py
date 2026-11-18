"""Check deployment configuration without assuming any example host is actually deployed."""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('configure_endpoint',ROOT/'scripts/configure_endpoint.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
SERVER=json.loads((ROOT/'server.json').read_text())

class DeploymentConfiguration(unittest.TestCase):
    def test_tool_inventory_matches_exported_mcp_contract(self):
        contract=json.loads((ROOT/'api/tools.json').read_text())
        self.assertEqual(contract['core_commit'],SERVER['core_commit'])
        self.assertEqual(sorted(tool['name'] for tool in contract['tools']),SERVER['tool_names'])
        self.assertEqual(len(contract['tools']),SERVER['tool_count'])
        self.assertTrue(all(tool['inputSchema']['x-data-contract']['input_mode']=='fixed_dataset' for tool in contract['tools']))

    def test_endpoint_and_download_base_share_the_operator_supplied_prefix(self):
        rows,environment=module.deployment('https://chemistry.test/study/',SERVER)
        self.assertEqual(rows[0]['url'],'https://chemistry.test/study/mcp')
        self.assertIn('CHEM_REPRO_PUBLIC_BASE_URL=https://chemistry.test/study\n',environment)
        self.assertNotIn('timeout',rows[0])  # The checked CoScientist cmd_load ignores that field.

    def test_local_address_is_explicitly_development_only(self):
        for host in ['127.0.0.1','localhost','[::1]']:
            with self.assertRaises(ValueError):module.deployment('http://'+host+':7343',SERVER)
            rows,_=module.deployment('http://'+host+':7343',SERVER,allow_local=True)
            self.assertTrue(rows[0]['url'].endswith('/mcp'))

    def test_ambiguous_or_example_configuration_is_rejected(self):
        for address in ['https://science.example.org','https://chemistry.test:invalid','https://<host>','https://chemistry.test/mcp',
                        'https://user:secret@chemistry.test','https://chemistry.test?token=x']:
            with self.assertRaises(ValueError):module.deployment(address,SERVER)

    def test_internal_http_is_a_deliberate_operator_choice(self):
        with self.assertRaises(ValueError):module.deployment('http://mannich-service:7343',SERVER)
        rows,_=module.deployment('http://mannich-service:7343',SERVER,allow_http=True)
        self.assertEqual(rows[0]['url'],'http://mannich-service:7343/mcp')

if __name__=='__main__':unittest.main()
