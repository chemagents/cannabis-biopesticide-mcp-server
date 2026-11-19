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
| `cannabis2026_reproduce_author_code` | Original-code RMT95 refits on all 30 deposited splits; complete metrics, checkpoints checks and pure DMPNN results |
| `cannabis2026_recovered_author_cohort` | Local 2465-record archive, 1010 archived scores above 0.70, exact structure alignment and unresolved score provenance |
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

The deposited [canpest code](https://github.com/chemagents/canpest) is now replayed separately from the article-based reconstruction. All 30 original train/test partitions were used;95 selected residues match the source in membership and order. The original DMPNN, descriptor preprocessing, validation-selected HGB blend, CatBoost and SVM functions were executed without algorithm changes. All 30 exported neural checkpoints were reloaded and their test scores checked. The MCP independently recomputes metrics from 19,050 held-out prediction rows.

| Actual computation | ROC-AUC mean | FPR mean |
|---|---:|---:|
| Source DMPNN/HGB blend | 0.929048 | 11.7391% |
| Source blend × CatBoost RMT95 | 0.918767 | 4.5596% |
| Pure DMPNN, separately evaluated | 0.923708 | 12.1182% |
| Pure DMPNN × CatBoost RMT95 | 0.917570 | 5.0613% |

Full per-split values, sample SDs, recall and true positives lost are returned by the tool. The source score called DMPNN contains a nonzero HGB contribution in 25/30 new fits. Pure neural results are kept separate. Original-code RMT selection uses global labels and an in-sample inner diagnostic; this is source-code reproduction, not leakage-free independent validation.

The source RMT95 reference is ROC-AUC 0.9167 / FPR 5.14%. The article's 0.9176 / 4.92% is attributed by the deposited documentation to a later RMT95+RTE390 geometric mixture, excluded from its rerun. These protocols are not silently substituted. The article also differs from deposited code in dropout, readout aggregation, selection and SVM kernel. [Exact protocol comparison and refitting instructions](https://github.com/chemagents/chem-repro-suite/blob/9e7a91e4aa643053ac7632daa5e556304b3c35b0/CANNABIS_AUTHOR_REPLAY.md).

A local author table recovers 2,465 exact source structures and 1,010 archived scores >=0.70. Its score lineage remains unverified; chemical class annotations do not establish CCD secondary-metabolite status. All ten Figure 4 structures are aligned, but the complete 44-compound selection is not recovered. The tool exposes the two candidates below 0.70 in the archived score rather than treating it as a proven initial DMPNN probability.

Open toxicology remains a separate reconstruction, not the original Syntelly weights. Stored docking scores are reanalyzed; a new docking campaign is not implied. Full-article reproduction and an end-to-end CoScientist report are not certified by these direct MCP results.

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
