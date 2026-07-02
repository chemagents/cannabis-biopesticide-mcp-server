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

The `ToolRetrieverAgent` then surfaces the 9 tools for biopesticide / insecticide / docking /
plant-metabolite queries, and `ExperimentAgent` (FEDOT.MAS) calls them by URL. Shared Docker
network → register `http://cannabis-biopesticide-mcp-server:7331/mcp`.

Example CoScientist prompt:

> "Use the cannabis-biopesticide tools: analyse the docking of Cannabis sativa metabolites to
> the pest targets and state how many are candidate biopesticides."

---

## Verified reproduction (local)

Run `python reproduce_paper.py` or `uv run pytest tests -v`. Observed:

```text
reproduce_all: 7/7 headline results reproduced within tolerance
  [PASS] docking_5_negative_OR28_positive : 5 neg / anomaly=OR28
  [PASS] metabolite_docking_median_range  : [-7.3, -5.3]   (paper -7.2..-5.2)
  [PASS] rmt_lambda_plus                  : 1.9381         (paper 1.938, exact)
  [PASS] rmt_m_opt                        : ~134-153       (paper 161)
  [PASS] qsar_roc_auc_high                : ~0.93          (paper DMPNN-SD 0.928)
  [PASS] cb_sd_docking_in_range           : 0.756          (paper 0.68-0.80)
  [PASS] biopesticide_candidates_fraction : ~0.35          (paper 0.41)
reproduce_claims: 5/5 ;  pytest: 6 passed ;  HTTP: 9 tools exposed
```

The upstream molecular docking (AutoDock Vina-GPU) is bundled as scores + 390 residue-term
energies, so the server has **no GPU/torch dependency** — it recomputes the docking statistics,
the RMT (Marchenko–Pastur) filter, the QSAR ablation, the applicability domain and the candidate
list deterministically. See [`README.md`](./README.md) for the open-analogue mapping and the
documented divergences (HGB vs DMPNN-SD ≈0.92 vs 0.93; Tanimoto-kNN AD stricter than the paper's).
```
