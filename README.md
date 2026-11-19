# cannabis-biopesticide-mcp-server

An MCP server for reproducing and auditing the analyses in:

> Karamanova, D.; Erin, N.; Bodrov, A.; Tkachenko, V.; Safronov, V.; Fedorov, M.
> *Biopesticidal Potential of the Cannabis sativa L. Metabolites: A Denoised, Docking-Informed QSAR Model.*
> **International Journal of Molecular Sciences 2026, 27, 8000.**
> https://doi.org/10.3390/ijms27188000

The server reconstructs docking-informed QSAR analyses and candidate prioritization from the available source data: **5,920 records — 1,680 pesticides, 1,491 AChE-inactive controls and 2,749 query compounds**. It combines residue interaction descriptors with a Chemprop DMPNN and provides separate open toxicology analyses. Stored docking scores are reused; advanced model calculations require additional workers.

Scientific code and source data are shared through [chem-repro-suite](https://github.com/chemagents/chem-repro-suite/tree/4be1e94c912db7a2790c831fb01d098ad290a88a), pinned to version 0.4.4, commit `4be1e94c912db7a2790c831fb01d098ad290a88a`. This repository supplies the MCP entry point, deployment configuration and checks. Tools operate on the fixed article dataset; they do not accept arbitrary uploaded datasets.

## Tools

| Tool | What it returns |
|---|---|
| `cannabis2026_dataset_audit` | Source cohorts, identity overlap and residue-matrix coverage |
| `cannabis2026_docking_comparisons` | Six target comparisons, effect directions and BH FDR correction |
| `cannabis2026_select_residues` | Training-only Appendix A RMT and energy-based residue selection |
| `cannabis2026_validate_rte` | CatBoost and LinearSVM for six targets and combined descriptors |
| `cannabis2026_validate_dmpnn` | Actual Chemprop graph model under paired random/scaffold validation |
| `cannabis2026_evaluate_fusion` | Component scores and their product on the same held-out compounds |
| `cannabis2026_prioritize_candidates` | Candidate scores, ranks and mapping of Figure 4 structures |
| `cannabis2026_open_toxicity_profile` | Endpoint-specific open-model predictions and applicability checks |
| `cannabis2026_open_toxicity_validation` | Supplementary open benchmarks, source cohorts and validation limits |

Exact parameters and output schemas are in [api/tools.json](api/tools.json). Each result includes metrics, tables, limitations and provenance. Downloadable artifacts have a URL, byte count and SHA-256; full CSV files accompany table previews. `success` means the computation completed, not that every paper claim matched.

For example, ask CoScientist:

> Does multiplying the structural and RTE scores reduce false positives? Compare both components and their product on identical held-out compounds under random and scaffold validation. Return ROC-AUC, PR-AUC, FPR, recall, confusion matrices and coverage, and distinguish the available source cohort from the final paper cohort.

See [REPRODUCTION_QUESTIONS.md](REPRODUCTION_QUESTIONS.md) for further questions and the tools expected to answer them.

## Reproduction fidelity

| Analysis | Recomputed result / limitation |
|---|---|
| Source cohort | 5,920 records; mapping to the paper’s final 2,465 queries is unavailable |
| RTE validation | 420 fits; ALL6 CatBoost ROC-AUC 0.821755 ± 0.014796 versus reported 0.802 ± 0.017 |
| Paired DMPNN ROC-AUC | 0.915494 random / 0.892974 scaffold on the available source cohort |
| Paired product-score ROC-AUC | 0.913629 random / 0.886159 scaffold |
| Product-score FPR reduction | Approximately 48% random / 57% scaffold, with lower recall |

RTE statistics are mean ± SD over 30 repeated holdouts, not independent OOF observations. The available cohort is not silently truncated to match the paper. Some query structures overlap the labeled reference sets, so their predictions are not external validation. Missing original component predictions and occurrence annotations prevent an exact replay of the 44-candidate selection.

Open toxicology is a reconstruction informed by the 2024 method, **not the original Syntelly service or weights**. `fixed_public_v1` and `nested_paper2024_v2` have different training requirements; the nested recipe requires separately registered completed runs. A product score is not a calibrated probability, and candidate ranking does not establish efficacy or environmental safety. The superseded implementation remains in `legacy/`; its HGB outputs are not current DMPNN results.

## Run locally

Requires Git, Python 3.12 and uv:

```bash
git clone https://github.com/chemagents/cannabis-biopesticide-mcp-server.git
cd cannabis-biopesticide-mcp-server
uv sync --frozen --python 3.12
uv run cannabis-biopesticide-mcp-server
```

The default port is 7341. This starts a local development instance; no LLM API key is required by the MCP server.

## Attach to CoScientist

Deploy once on a host reachable by CoScientist and register its Streamable HTTP `/mcp` endpoint. The same service exposes result files at `/files/…`. End users then ask questions through CoScientist.

**No shared hosted endpoint is currently published.** Set `MCP_BASE_URL` to the actual deployment address, without `/mcp`, and generate the configuration:

```bash
uv run python scripts/configure_endpoint.py --base-url "$MCP_BASE_URL"
```

[Deployment, Docker and registration instructions](COSCIENTIST_INTEGRATION.md) cover startup, artifact URLs, optional workers and verification from the client network. [server.json](server.json) records the integration contract and deployment status.

## Tests

```bash
python -m unittest discover -s tests -v
uv run python scripts/smoke_server.py --study cannabis
```

[CI](https://github.com/chemagents/cannabis-biopesticide-mcp-server/actions/workflows/smoke.yml) checks configuration, startup, tool discovery and artifact downloads in Python and Docker. These checks do not establish full scientific reproduction or acceptance of every CoScientist answer. The English questions define the system evaluation. Current end-to-end runs use local CoScientist fixes for MCP result capture, source access and reporting; they do not certify the unmodified upstream version. Scientific findings and source limitations are reviewed separately from successful execution.

## License / data

Code: [MIT](LICENSE). Source provenance and dataset licenses are listed in the [data manifest](https://github.com/chemagents/chem-repro-suite/blob/4be1e94c912db7a2790c831fb01d098ad290a88a/chem_repro/data/cannabis/manifest.json) and [NOTICE](https://github.com/chemagents/chem-repro-suite/blob/4be1e94c912db7a2790c831fb01d098ad290a88a/NOTICE.md). Please cite the article when using its data or results.
