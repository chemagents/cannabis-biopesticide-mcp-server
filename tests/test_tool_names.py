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
from types import SimpleNamespace

import numpy as np

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
    assert tuple(srv.QUESTION_TOOLS) == srv.QUESTION_ORDER == (1, 2, 3, 4)
    for question, tools in srv.QUESTION_TOOLS.items():
        for position, tool in enumerate(tools):
            chain = srv._chain(question, tool)
            assert chain["question"] == srv.QUESTIONS[question]
            assert chain["question_number"] == question
            expected_sibling = tools[position + 1:position + 2]
            assert chain["next_tools"] == expected_sibling
            assert "URL or path" in chain["artifact_output_policy"]
            assert "kind" in chain["artifact_output_policy"]
            assert "SHA-256" in chain["artifact_output_policy"]
            assert "directly" in chain["artifact_output_policy"]
            assert "task log" in chain["artifact_output_policy"]
            assert "bare confirmation" in chain["artifact_output_policy"]
            assert "every warning" in chain["artifact_output_policy"]

            if expected_sibling:
                assert chain["workflow_status"] == "within_question"
                assert "next_question" not in chain
            elif question < 4:
                assert chain["workflow_status"] == "question_complete"
                assert chain["next_question"] == {
                    "question_number": question + 1,
                    "question": srv.QUESTIONS[question + 1],
                    "entry_tool": srv.QUESTION_TOOLS[question + 1][0],
                }
            else:
                assert chain["workflow_status"] == "reproduction_complete"
                assert chain["next_question"] is None


def test_question_routes_have_no_reciprocal_edges_or_skips():
    edges = []
    for question, tools in srv.QUESTION_TOOLS.items():
        for tool in tools:
            chain = srv._chain(question, tool)
            edges.extend((tool, successor) for successor in chain["next_tools"])
            if chain.get("next_question"):
                edges.append((tool, chain["next_question"]["entry_tool"]))

    canonical = [tool for question in srv.QUESTION_ORDER for tool in srv.QUESTION_TOOLS[question]]
    assert edges == list(zip(canonical, canonical[1:]))
    assert not any((destination, source) in edges for source, destination in edges)


def test_chain_rejects_tool_from_another_question():
    try:
        srv._chain(1, "docking_analysis")
    except ValueError as exc:
        assert "not registered for question 1" in str(exc)
    else:
        raise AssertionError("cross-question routing must be rejected")


def test_registered_question_tools_emit_their_declared_routes(monkeypatch):
    fake_ds = SimpleNamespace(
        n=4,
        active_mask=np.array([True, False, False, False]),
        inactive_mask=np.array([False, True, False, False]),
        metabolite_mask=np.array([False, False, True, True]),
        labelled_mask=np.array([True, True, False, False]),
    )
    monkeypatch.setattr(srv, "load_dataset", lambda: fake_ds)
    monkeypatch.setattr(srv.plotting, "plot_docking", lambda _rows: None)
    monkeypatch.setattr(srv.plotting, "plot_chemical_space", lambda _ds: None)
    monkeypatch.setattr(
        srv.docking,
        "active_vs_inactive",
        lambda _ds: {"per_protein": [], "metabolite_median_range": [-7.0, -5.0]},
    )
    monkeypatch.setattr(
        srv.docking,
        "metabolite_pesticide_overlap",
        lambda _ds: {
            "frac_nn_active": 0.5,
            "n_metabolites": 2,
            "median_nn_similarity": 0.4,
            "frac_nn_similarity_above_0_4": 0.5,
        },
    )
    monkeypatch.setattr(
        srv.models,
        "rmt_selection",
        lambda scaffold=False: {"n_signal": 16, "m_opt": 159, "lambda_plus": 1.9381},
    )
    monkeypatch.setattr(
        srv.models,
        "qsar_ablation",
        lambda scaffold=False: {"results": {"structure": {"roc_auc": 0.93}}, "m_opt": 159},
    )
    monkeypatch.setattr(srv.models, "cb_sd_rte", lambda scaffold=False: {"all6_rte": {"roc_auc": 0.79}})
    monkeypatch.setattr(
        srv.models,
        "model_stack_quality",
        lambda: {
            "blend": {"roc_auc": 0.914},
            "components": {"dmpnn": {"roc_auc": 0.909}, "hgb": {"roc_auc": 0.9}},
            "blend_w_dmpnn": 0.62,
        },
    )
    monkeypatch.setattr(
        srv.models,
        "docking_veto",
        lambda: {
            "fpr_before": 0.157,
            "fpr_after_veto": 0.05,
            "fpr_reduction_pct": 68.0,
            "recall_before": 0.9,
            "recall_after_veto": 0.8,
        },
    )
    monkeypatch.setattr(
        srv.models,
        "predict_biopesticides",
        lambda: {
            "n_metabolites": 100,
            "headline_count": 42,
            "headline_fraction": 0.42,
            "backend": "test",
            "headline_basis": "test basis",
        },
    )

    registered = _tools()
    for question, names in srv.QUESTION_TOOLS.items():
        for name in names:
            result = registered[name].fn()
            expected = srv._chain(question, name)
            assert {key: result["metadata"][key] for key in expected} == expected


def test_missing_chemical_space_evidence_does_not_advance_question(monkeypatch):
    monkeypatch.setattr(srv, "load_dataset", lambda: object())
    monkeypatch.setattr(srv.plotting, "plot_chemical_space", lambda _ds: None)

    def fail_overlap(_ds):
        raise RuntimeError("descriptor calculation failed")

    monkeypatch.setattr(srv.docking, "metabolite_pesticide_overlap", fail_overlap)
    result = srv.chemical_space()
    metadata = result["metadata"]
    assert "could not be computed" in result["answer"]["finding"]
    assert metadata["next_tools"] == []
    assert "next_question" not in metadata
    assert metadata["workflow_status"] == "awaiting_retry"
    assert metadata["question_status"] == "evidence_unavailable"
    assert metadata["retry_tool"] == "chemical_space"
    assert metadata["retry_parameters"] == {}


def test_audit_fallback_tools_still_carry_the_answer_policy():
    tools = _tools()
    for name in ("canpest_reproduce_all", "canpest_reproduce_claims"):
        code = tools[name].fn.__code__
        assert "audit_fallback" in code.co_consts
        assert "ARTIFACT_OUTPUT_POLICY" in code.co_names
