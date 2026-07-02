# cannabis-biopesticide-mcp-server

An MCP server that **reproduces the results** of:

> *Biopesticidal Potential of the Cannabis sativa L. Metabolome: A Denoised, Docking-Informed QSAR Model.*

The upstream molecular docking (AutoDock **Vina-GPU** of 5920 ligands to 6 pest-relevant
proteins) was the one-time data-generation step; its output — SMILES, pesticide activity
labels, six docking scores, and 390 residue-term energies (RTE) — is **bundled** in
`server/data/`, so this server recomputes every downstream analysis **deterministically and
fast, with no GPU**. It follows the same pattern as `tox-antitargets-mcp-server` (bundle the
expensive docking output; recompute the analyses).

## Open-source analogue stack

| Paper component | Open-source analogue here |
|---|---|
| SynMap / chemical space | differential (Bemis–Murcko scaffold) fingerprint + t-SNE |
| Molecular docking (Vina-GPU) | **bundled** docking scores + 390 RTE features |
| RMT denoising (Lee/Brenner/Colwell) | numpy/scipy **Marchenko–Pastur** filter (port of `rmt_filter.py`) |
| **DMPNN-SD** QSAR (ROC-AUC ~0.93) | **HistGradientBoosting** on RDKit-2D descriptors (torch-free, ~0.92–0.93; optional torch DMPNN) |
| CB-SD / SVM-SD (docking features) | gradient boosting on docking scores / RTE |
| Applicability domain (Probability) | Tanimoto kNN(k=5) + Gaussian |
| Toxicity / ecotoxicity (Syntelly) | `heracleum-tox` server (open Syntelly analogue) + published Table S1/3/4 values |

## Tools

| Tool | Reproduces | Returns |
|------|-----------|---------|
| `dataset_overview` | §3.1 | pesticides / inactives / metabolites, 6 target proteins |
| `docking_analysis` | §3.2 / Fig 2 | per-protein Mann–Whitney Δ; OR28 anomaly + figure |
| `rmt_feature_selection` | §3.3 | Marchenko–Pastur λ+, m_opt, signal features |
| `qsar_model_quality` | §3.3 | RDKit2D + docking/RMT ablation; docking-only baseline |
| `predict_biopesticides` | §3.4 | metabolite biopesticide probabilities + AD → candidate count |
| `chemical_space` | Fig S2 | metabolites-vs-pesticides t-SNE map |
| `tox_ecotox_reference` | §3.5 / Table S1 | tox/ecotox model metrics + safety comparison |
| `reproduce_all` | — | headline numbers vs the paper |
| `reproduce_claims` | all | the paper's conclusions restated with reproduced numbers |

Each tool returns `{"answer": ..., "metadata": ...}`; figures are PNG artifacts (local or S3).

## Reproduction fidelity

| Metric | Paper | This server |
|---|---|---|
| Docking trend | 5 targets Δ<0, OR28 Δ>0 | **5 negative, OR28 +1.4 (exact)** |
| Metabolite docking median | −5.2…−7.2 kcal/mol | **−5.3…−7.3** |
| RMT Marchenko–Pastur λ+ | 1.938 | **1.9381 (exact)** |
| RMT m_opt (random split) | 161 | **~134–153** |
| QSAR ROC-AUC | ~0.928 (DMPNN-SD) | **~0.92–0.93 (HGB)** |
| Docking-only ROC-AUC | 0.68–0.80 (CB-SD) | **0.756** |
| Biopesticide candidates | 1010 (40.97%) | **~960 (35%)** |

Documented open-analogue divergences: the QSAR uses RDKit-2D descriptors + HistGradientBoosting
instead of the proprietary 217-descriptor DMPNN-SD, so ROC-AUC lands ~0.92 vs 0.93; the AD is a
Tanimoto kNN Gaussian (stricter than the paper's Probability<0.5), giving ~35% candidates vs
41%. The docking statistics and RMT λ+ reproduce exactly.

## Run locally

```bash
git clone https://github.com/chemagents/cannabis-biopesticide-mcp-server
cd cannabis-biopesticide-mcp-server
cp .env.example .env
uv sync
uv run python -m server.canpest_server      # serves http://0.0.0.0:7331/mcp
uv run pytest tests -v                       # reproduction tests
```

## Run with Docker

```bash
docker compose up -d --build      # host 7337 -> container 7331
```

To run it inside the CoScientist stack, add a service using `Dockerfile.coscientist` — see
[`COSCIENTIST_INTEGRATION.md`](./COSCIENTIST_INTEGRATION.md).

## Attach to CoScientist

```bash
python scripts/rag_tools/cli.py load mcp-servers/cannabis-biopesticide-mcp-server/rag_registration.json
```

The `ToolRetrieverAgent` then surfaces these tools for biopesticide / insecticide / docking /
plant-metabolite queries, and `ExperimentAgent` (FEDOT.MAS) calls them by URL.

## Data / credit

Docking scores, RTE features, splits and reference metrics are the authors' `rmt_canpest`
package (RMT-RTE feature selection + DMPNN+GBM ablation). Docking with AutoDock Vina-GPU 2.0;
RMT after Lee, Brenner & Colwell (PNAS 2016); DMPNN after Yang et al. 2019.
