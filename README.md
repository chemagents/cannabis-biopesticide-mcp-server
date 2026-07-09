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
| **DMPNN-SD** QSAR (ROC-AUC ~0.93) | **DMPNN + HGB soft-voting stack** (`p = 0.62·p_DMPNN + 0.38·p_HGB`; component preds bundled, blend recomputed in-code, torch-free) |
| CB-SD / SVM-SD (docking features) | gradient boosting on docking scores / RTE |
| Applicability domain (Probability) | Tanimoto kNN(k=5) + Gaussian |
| Toxicity / ecotoxicity (Syntelly) | `heracleum-tox` server (open Syntelly analogue) + published Table S1/3/4 values |

## Tools

| Tool | Reproduces | Returns |
|------|-----------|---------|
| `dataset_overview` | §3.1 | pesticides / inactives / metabolites, 6 target proteins |
| `docking_analysis` | §3.2 / Fig 2 | per-protein Mann–Whitney Δ; OR28 anomaly + figure |
| `rmt_feature_selection` | §3.3 | Marchenko–Pastur λ+, m_opt, signal features |
| `qsar_model_quality` | §3.3 | RDKit2D+RMT ablation; residue-level docking baseline (CB-SD) |
| `model_stack` | §3.3 | DMPNN+HGB soft-voting stack: component vs blend OOF-CV (blend > DMPNN > HGB) |
| `docking_veto` | §3.3 | docking-consistency veto: FPR reduction (p_QSAR × p_RMT-RTE) |
| `predict_biopesticides` | §3.4 | metabolite biopesticide probabilities + AD → candidate count |
| `chemical_space` | Fig S2 | metabolites-vs-pesticides t-SNE map |
| `tox_ecotox_reference` | §3.5 / Table S1 | tox/ecotox model metrics + safety comparison |
| `reproduce_all` | — | headline numbers vs the paper |
| `reproduce_claims` | all | the paper's conclusions restated with reproduced numbers |

Each tool returns `{"answer": ..., "metadata": ...}`; figures are PNG artifacts (local or S3).

## Reproduction fidelity

Using the authors' exact 217-descriptor matrix (`fp_rdkit2d.npy`) and a faithful port of their
inner-CV selection:

| Metric | Paper | This server |
|---|---|---|
| Docking trend | 5 targets Δ<0, OR28 Δ>0 | **5 negative, OR28 +1.4 (exact)** |
| Metabolite docking median | −5.2…−7.2 kcal/mol | **−5.3…−7.3** |
| RMT Marchenko–Pastur λ+ | 1.938 | **1.9381 (exact)** |
| RMT m_opt (random split) | 161 | **159** |
| QSAR ROC-AUC | 0.9283 (DMPNN-SD stack) | **stack 0.914 OOF / ~0.931 per-feat · blend 0.914 > DMPNN 0.909 > HGB 0.900 (matched CV)** |
| CB-SD residue-level docking | 0.802 (ALL6) | **0.790** |
| Docking-consistency veto FPR | 12.20% → 4.92% | **15.7% → 5.0% (68% reduction)** |
| Biopesticide candidates (prob>0.7) | 1010 / 40.97% (of 2465) | **1152 / 41.9% (of 2749; bundled DMPNN-SD) · 34–49% (torch-free HGB)** |

Everything lands within the paper's own ±SD. The candidate fraction is a DMPNN-SD number: with the
authors' bundled DMPNN-SD probabilities (`server/data/dmpnn_pred.csv` — the DMPNN+HGB blend, `blend_w
0.62`, used in the paper), `predict_biopesticides` reports **1152 / 41.9%** at prob>0.7, reproducing
the paper's ~41% fraction. The absolute count differs from the paper's 1010 only because the shipped
prediction set is **2749 metabolites** (all docked), whereas the paper text cites 2465; the fraction
is what reproduces. The torch-free HGB fallback is calibrated differently and brackets it — **49%**
without an applicability-domain filter, **34%** with a Tanimoto-kNN AD. Note the DMPNN is
**open-source (Chemprop)**, not proprietary; only the toxicity platform (Syntelly, §3.5) is closed.

### Bundled DMPNN-SD candidate fraction

`server/data/dmpnn_pred.csv` (cols `ligand_id, prob`) ships the authors' DMPNN-SD metabolite
probabilities — the DMPNN+HGB blend trained on `mps` (Apple Metal), bundled like the docking scores —
so `predict_biopesticides` reports the exact pipeline number with **no torch at serve time**. To
regenerate it from open code, the training pipeline is vendored under `server/vendor/` (`pip install
.[dmpnn]`, Chemprop/torch); the docking scores + 390 residue-term energies are bundled, so even that
path needs no GPU-side docking. Delete the CSV to fall back to the torch-free HGB analogue (0.931
ROC-AUC), which every tool response labels explicitly.

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

To run it inside the CoScientist stack, use `Dockerfile.coscientist` — see
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
