# CoScientist integration

The server uses Streamable HTTP at `/mcp` and serves artifacts at `/files/{sha256}.{extension}`. It exposes 9 tools under registry name `cannabis-2026`. See [server.json](server.json) and [api/tools.json](api/tools.json) for the exact contract. The scientific backend is pinned to `4be1e94c912db7a2790c831fb01d098ad290a88a`.

## Start the service

Install with `uv sync --frozen --python 3.12`. Set `MCP_BASE_URL` to the real base address reachable from the CoScientist backend, without `/mcp`. For a shared HTTPS deployment:

```bash
uv run python scripts/configure_endpoint.py --base-url "$MCP_BASE_URL"
set -a
. deployment/server.env
set +a
export CHEM_REPRO_ARTIFACTS_DIR="$PWD/outputs"
uv run cannabis-biopesticide-mcp-server
```

The generator creates `deployment/registration.json` and `deployment/server.env`, refuses overwrites and rejects placeholder or loopback addresses by default. It does not deploy the service or create DNS/certificates. Explicit internal HTTP is supported with `--allow-http`; local development additionally requires `--allow-local`.

`CHEM_REPRO_PUBLIC_BASE_URL` controls artifact URLs, `CHEM_REPRO_HOST` the bind address and `CHEM_REPRO_PORT` the port (default 7341). A reverse proxy must forward both routes. When using a URL prefix, strip it before forwarding and retain it in the public base URL. Keep the artifact directory on persistent storage.

## Docker

After generating the environment file, build and start a local image behind a host proxy:

```bash
docker build -t cannabis-mcp-local .
docker run --rm --name cannabis-mcp \
  --env-file deployment/server.env \
  -v "$PWD/outputs:/outputs" \
  -p 127.0.0.1:7341:7341 \
  cannabis-mcp-local
```

The loopback binding is the proxy’s internal upstream, not the address to register for a remote client. For a proxy in another container, use a shared container network. No prebuilt public image is advertised.

Dataset audit and stored-score analysis use the base environment. RTE validation can perform 420 fits. DMPNN, fusion and toxicology require separate pinned workers and may train on the first uncached call:

```bash
uv run python scripts/setup_cannabis_workers.py --directory "$PWD/workers" --output worker_configuration.json
uv run python scripts/run_cannabis.py --configuration worker_configuration.json --public-base-url "$MCP_BASE_URL" --host 0.0.0.0
```

The installer prepares environments, not trained models. `nested_paper2024_v2` needs separately registered completed runs. The base Docker image does not contain these optional workers; configure workers built for the deployment OS. Host-native macOS environments cannot be reused inside a Linux container.

## Register and verify

From a configured [CoScientist_dev](https://github.com/aimclub/CoScientist_dev) checkout with its RAG registry running:

```bash
uv run python -m scripts.rag_tools.cli load /absolute/path/to/deployment/registration.json
uv run python -m scripts.rag_tools.cli list
```

Compatibility was checked against CoScientist revision `900aeae29d77416a798bdf706bc1c6df8d8e1911`. Its loader reads `name`, `url`, `description` and optional `token`; it ignores a JSON `timeout` field. Configure `MCP__REQUEST_TIMEOUT` and `MCP__SSE_READ_TIMEOUT` in the CoScientist process (checked defaults: 60 s and 7,200 s), together with gateway and agent execution limits. Long model fits need an appropriate execution window.

The loader may report a per-server error without a failing process exit code. Confirm the registry entry and discovered tools. Then, from the client network using the MCP adapter environment:

```bash
uv run python scripts/check_endpoint.py --url "$MCP_BASE_URL/mcp"
```

This checks the exact tool inventory, dataset-audit counts, and each returned file’s URL, size and SHA-256. Optional `--headers-file /secure/path/request.headers.json` supplies a JSON object of request headers to both MCP and file downloads. Authentication is not built into this adapter; a protected gateway must cover both routes. Keep credentials out of Git.

A browser GET of `/mcp` is not an MCP test. If calls succeed but files fail, check the public base URL, proxy routing and access headers. Publish a shared endpoint in `server.json` only after testing the deployed instance.

## Results and execution

Results contain `answer.metrics`, `answer.tables`, `answer.limitations`, and metadata for the DOI, protocol, parameters, software and source hashes. `evidence_kind` distinguishes `computed`, `source_record` and `paper_reported` evidence. Artifacts include `download_url`, `size_bytes` and `sha256`. For tables marked `is_preview`, use `total_rows` and `full_table_artifact_sha256` to retrieve the complete CSV.

Preserve `partial` and `unsupported` outcomes and all scientific limitations in the final answer. A successful calculation does not imply agreement with the paper. Scientific calls are serialized within each process; discovery and file delivery remain available. HTTP cancellation may not stop ongoing CPU work immediately.

CoScientist handles planning and the final narrative; the MCP supplies scientific calculations and provenance. Validate routing, actual calls, numerical evidence, interpretation and artifact accessibility separately. English prompt specifications have not undergone a new complete end-to-end acceptance run.

For same-host development, [examples/registration.local.json](examples/registration.local.json) contains the explicitly local configuration. `--transport stdio` is also supported, but artifact downloads still need an HTTP route; use HTTP for the shared deployment.
