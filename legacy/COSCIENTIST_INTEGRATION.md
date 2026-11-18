# Integrating cannabis-biopesticide into CoScientist

Standard CoScientist MCP server (FastMCP, HTTP `/mcp`, tools return `{"answer", "metadata"}`).
Integration is the same three steps as the other `chemagents` servers (e.g.
`heracleum-tox-mcp-server`, whose identical wiring was verified end-to-end through
Orchestrator → RAG → FEDOT.MAS → tools).

## 1. Place the server

```bash
cd CoScientist/mcp-servers
git clone https://github.com/chemagents/cannabis-biopesticide-mcp-server
```

## 2. Add the docker-compose service

Append to `mcp-servers/docker-compose.yml` (build context = repo root → use
**`Dockerfile.coscientist`**; the plain `Dockerfile` is for standalone `context: .`):

```yaml
  cannabis-biopesticide-mcp-server:
    build:
      context: ..
      dockerfile: mcp-servers/cannabis-biopesticide-mcp-server/Dockerfile.coscientist
    container_name: cannabis-biopesticide-mcp-server
    env_file:
      - ./cannabis-biopesticide-mcp-server/.env
    environment:
      PYTHONUNBUFFERED: "1"
    ports:
      - "7337:7331"
    restart: unless-stopped
```

```bash
cp cannabis-biopesticide-mcp-server/.env.example cannabis-biopesticide-mcp-server/.env
docker compose up -d --build cannabis-biopesticide-mcp-server
```

## 3. Register it in the RAG

```bash
python scripts/rag_tools/cli.py load mcp-servers/cannabis-biopesticide-mcp-server/rag_registration.json
```

The `ToolRetrieverAgent` then surfaces the 11 tools for biopesticide / insecticide / docking /
plant-metabolite queries, and `ExperimentAgent` (FEDOT.MAS) calls them by URL. Shared Docker
network → register `http://cannabis-biopesticide-mcp-server:7331/mcp`.

Use the four-question scenario in
[`REPRODUCTION_QUESTIONS.md`](./REPRODUCTION_QUESTIONS.md), rather than one broad reproduction
prompt. Each question explicitly asks CoScientist to return the figure artifacts (URL/path, kind
and SHA-256) alongside a direct scientific answer.

---

## Verified reproduction (local)

Run `python reproduce_paper.py` or `uv run pytest tests -v`. Observed:

```text
canpest_reproduce_all: 8/10 checks match their stated tolerances
  [PASS] docking direction, docking median, lambda+, m_opt
  [FAIL] RMT signal eigenvalues: 16 vs paper 19
  [PASS] recomputed QSAR ROC-AUC, CB-SD, veto effect, candidate fraction
  [FAIL] pre-veto baseline FPR: 15.7% vs paper 12.2%
canpest_reproduce_claims: 5/6 claims reproduced
  [NOT REPRODUCED] C5 safety comparison: published Syntelly lookup, no live model here
HTTP: 11 tools exposed (the three cross-server collisions use `canpest_` prefixes)
```

The upstream molecular docking (AutoDock Vina-GPU) is bundled as scores + 390 residue-term
energies, so the server has **no GPU/torch dependency** — it recomputes the docking statistics,
the RMT (Marchenko–Pastur) filter, the QSAR ablation, the applicability domain and the candidate
list deterministically. See [`README.md`](./README.md) for the open-analogue mapping and the
documented divergences. The server reports failures explicitly instead of inflating the
reproduction count; in particular, the §3.5 Syntelly safety claim remains a published lookup.
