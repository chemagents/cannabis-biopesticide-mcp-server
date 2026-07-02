# Standalone image for the cannabis-biopesticide MCP server.
# (For building inside the CoScientist monorepo, use Dockerfile.coscientist instead.)
FROM python:3.12-slim

# RDKit / scientific stack runtime libraries (slim image omits these).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 \
    libxext6 \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Build context is this repository's root.
COPY pyproject.toml README.md ./
COPY server ./server
RUN pip install --no-cache-dir .

ENV CANPEST_ARTIFACTS_DIR=/app/artifacts
RUN mkdir -p /app/artifacts

EXPOSE 7331
CMD ["python", "-m", "server.canpest_server"]
