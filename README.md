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
| `canpest_dataset_overview` | §3.1 | pesticides / inactives / metabolites, 6 target proteins |
| `docking_analysis` | §3.2 / Fig 2 | per-protein Mann–Whitney Δ; OR28 anomaly + figure |
| `rmt_feature_selection` | §3.3 | Marchenko–Pastur λ+, m_opt, signal features |
| `qsar_model_quality` | §3.3 | RDKit2D+RMT ablation; residue-level docking baseline (CB-SD) |
| `model_stack` | §3.3 | DMPNN+HGB soft-voting stack: component vs blend OOF-CV (blend > DMPNN > HGB) |
| `docking_veto` | §3.3 | docking-consistency veto: FPR reduction (p_QSAR × p_RMT-RTE) |
| `predict_biopesticides` | §3.4 | metabolite biopesticide probabilities + AD → candidate count |
| `chemical_space` | Fig S2 | metabolites-vs-pesticides t-SNE map |
| `tox_ecotox_reference` | §3.5 / Table S1 | published Syntelly metrics + reported safety comparison (lookup only) |
| `canpest_reproduce_all` | — | headline numbers vs the paper |
| `canpest_reproduce_claims` | all | the paper's conclusions restated with reproduced numbers |

Each tool returns `{"answer": ..., "metadata": ...}`; figures are structured PNG artifact
references (local or S3) with SHA-256. Fully configured S3 is checked at startup and fails closed;
an upload failure cannot silently turn into a container-local path unless
`CANPEST_S3_ALLOW_LOCAL_FALLBACK=true` is explicitly enabled. Use
`CANPEST_S3_PUBLIC_ENDPOINT_URL` when the caller-visible presign host differs from the internal
upload endpoint.

> **Agent-visible tool names.** `dataset_overview`, `reproduce_all` and `reproduce_claims` collide
> with identically-named tools in `tox-antitargets-mcp-server` and `heracleum-tox-mcp-server`, which
> are exposed to the agent at the same time. They are therefore registered under the `canpest_`-
> prefixed names above via `@mcp.tool(name=...)`. The Python functions in `server/canpest_server.py`
> keep their original names — only the MCP-facing name changed.

## Reproduction fidelity

Using the authors' exact 217-descriptor matrix (`fp_rdkit2d.npy`) and a faithful port of their
inner-CV selection:

| Metric | Paper | This server |
|---|---|---|
| Docking trend | 5 targets Δ<0 (−1.01…−0.70), OR28 Δ=+1.20 | **5 negative (−1.20…−0.70), OR28 +1.40** — direction reproduces, magnitudes differ |
| Metabolite docking median | −5.2…−7.2 kcal/mol | **−5.3…−7.3** |
| RMT Marchenko–Pastur λ+ | 1.938 | **1.9381 (exact)** |
| RMT m_opt (random split) | 161 | **159** |
| RMT signal eigenvalues | 19 | **16 — does not reproduce** |
| QSAR ROC-AUC | 0.9283 (DMPNN-SD stack) | **0.9306 recomputed here** (best ablation feature set). The stack figures 0.914 / 0.909 / 0.900 are the authors' *bundled* OOF-CV values echoed by `model_stack`, not measured by this server. |
| CB-SD residue-level docking | 0.802 (ALL6) | **0.790** |
| Docking-consistency veto FPR | 12.20% → 4.92% | **15.7% → 5.0% (68% reduction)** — the *effect* reproduces, the baseline FPR does not (15.7% vs 12.20%) |
| Biopesticide candidates (prob>0.7) | 1010 / 40.97% (of 2465) | **1152 / 41.9% (of 2749; bundled DMPNN-SD) · 34–49% (torch-free HGB)** |

Two quantities do **not** reproduce and are reported as such by `canpest_reproduce_all` (`rmt_n_signal`: 16 vs 19; `docking_veto_baseline_fpr`: 15.7% vs 12.20%), and `C5` (metabolites safer than synthetic pesticides) is **not verifiable on this server at all** — it is a published Syntelly result and this server runs no toxicity model. Everything else lands within the tolerances each check states. The candidate fraction is a DMPNN-SD number: with the
authors' bundled DMPNN-SD probabilities (`server/data/dmpnn_pred.csv` — the DMPNN+HGB blend, `blend_w
0.62`, used in the paper), `predict_biopesticides` reports **1152 / 41.9%** at prob>0.7, reproducing
the paper's ~41% fraction. The absolute count differs from the paper's 1010 only because the shipped
prediction set is **2749 metabolites** (all docked), whereas the paper text cites 2465; the fraction
is what reproduces. The torch-free HGB fallback is calibrated differently and brackets it — **49%**
without an applicability-domain filter, **34%** with a Tanimoto-kNN AD. Note the DMPNN is
**open-source (Chemprop)**, not proprietary; only the toxicity platform (Syntelly, §3.5) is closed.

### Bundled DMPNN-SD candidate fraction

`server/data/dmpnn_pred.csv` (cols `ligand_id, prob, proba_dmpnn, proba_hgb`; the code reads the latter two and blends them) ships the authors' DMPNN-SD metabolite
probabilities — the DMPNN+HGB blend trained on `mps` (Apple Metal), bundled like the docking scores —
so `predict_biopesticides` reports the exact pipeline number with **no torch at serve time**. To
regenerate it from open code, the training pipeline is vendored under `server/vendor/` (`pip install
.[dmpnn]`, Chemprop/torch); the docking scores + 390 residue-term energies are bundled, so even that
path needs no GPU-side docking. Delete the CSV to fall back to the torch-free HGB analogue (0.931
ROC-AUC). `predict_biopesticides` labels the active backend in its `backend` field; the other
QSAR tools do not.

## Run locally

```bash
git clone https://github.com/chemagents/cannabis-biopesticide-mcp-server.git
cd cannabis-biopesticide-mcp-server
cp .env.example .env
uv sync
uv run python -m server.canpest_server      # serves http://0.0.0.0:7331/mcp
uv run pytest tests -v                       # reproduction tests
```

## Run with Docker

```bash
cp .env.example .env
docker compose up -d --build      # host 7337 -> container 7331
```

The standalone `Dockerfile` uses `context: .`; `Dockerfile.coscientist` is the hardened
monorepo-context variant. Both use the locked dependency set, a digest-pinned Python base image,
and an unprivileged runtime user. See [COSCIENTIST_INTEGRATION.md](COSCIENTIST_INTEGRATION.md) for
the exact CoScientist compose wiring.

## Attach to CoScientist

```bash
python scripts/rag_tools/cli.py load mcp-servers/cannabis-biopesticide-mcp-server/rag_registration.json
```

The `ToolRetrieverAgent` then surfaces these tools for biopesticide / insecticide / docking /
plant-metabolite queries, and `ExperimentAgent` (FEDOT.MAS) calls them by URL.

## Retained research audits

The standalone repository also keeps the earlier confidence/calibration audit
([CONFIDENCE.md](CONFIDENCE.md)) and the exploratory open-data toxicity experiments
([TOX_REPRODUCTION.md](TOX_REPRODUCTION.md), `server/tox/`, `server/data/tox/`) for provenance and
independent reruns. They are not exposed as reproduced headline claims by the MCP server: the
production tool surface follows the article-reproduction contract above, and §3.5 remains clearly
labelled as a published Syntelly lookup unless the original training snapshot is available.

## Data / credit

Docking scores, RTE features, splits and reference metrics are the authors' `rmt_canpest`
package (RMT-RTE feature selection + DMPNN+GBM ablation). Docking with AutoDock Vina-GPU 2.0;
RMT after Lee, Brenner & Colwell (PNAS 2016); DMPNN after Yang et al. 2019.
