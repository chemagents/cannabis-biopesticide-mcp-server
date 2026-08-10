"""Regression guard for the cross-server tool-name collision fix.

`dataset_overview`, `reproduce_all` and `reproduce_claims` were defined under those exact
names in tox-antitargets, heracleum-tox AND cannabis-biopesticide. All three servers are exposed
to the same orchestrator, so the agent saw three tools per name and hallucinated a non-existent
`tox_reproducer`. The fix renames only the AGENT-VISIBLE name via `@mcp.tool(name=...)`; the
Python function names are deliberately unchanged. Nothing else guarded that, so a revert of the
decorator argument would have been silent.
"""
from __future__ import annotations

import asyncio

from server import canpest_server as srv

RENAMED = {
    "canpest_dataset_overview": "dataset_overview",
    "canpest_reproduce_all": "reproduce_all",
    "canpest_reproduce_claims": "reproduce_claims",
}


def _tools():
    return {t.name: t for t in asyncio.run(srv.mcp.list_tools())}


def test_colliding_tools_are_exposed_under_the_prefixed_names():
    names = set(_tools())
    missing = set(RENAMED) - names
    assert not missing, f"renamed tools missing from the registry: {sorted(missing)}"


def test_bare_colliding_names_are_not_exposed():
    names = set(_tools())
    leaked = names & set(RENAMED.values())
    assert not leaked, f"colliding bare tool names still on the wire: {sorted(leaked)}"


def test_python_function_names_are_unchanged():
    tools = _tools()
    for wire, fn_name in RENAMED.items():
        fn = getattr(tools[wire], "fn", None)
        assert fn is not None, f"cannot reach the python function behind {wire}"
        assert fn.__name__ == fn_name, (
            f"{wire} must keep the python function name {fn_name}, got {fn.__name__}")


def test_tool_count_is_stable():
    assert len(_tools()) == 11


def test_question_routes_and_artifact_contract_are_complete():
    covered = {tool for tools in srv.QUESTION_TOOLS.values() for tool in tools}
    assert covered == set(_tools()) - {"canpest_reproduce_all", "canpest_reproduce_claims"}
    for question, tools in srv.QUESTION_TOOLS.items():
        for tool in tools:
            chain = srv._chain(question, tool)
            assert chain["question"] == srv.QUESTIONS[question]
            assert tool not in chain["next_tools"]
            assert "URL or path" in chain["artifact_output_policy"]
            assert "SHA-256" in chain["artifact_output_policy"]
            assert "task log" in chain["artifact_output_policy"]
