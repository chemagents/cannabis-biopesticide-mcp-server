FROM python:3.12-slim@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36

# RDKit / scientific stack runtime libraries omitted by the slim base image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 \
    libxext6 \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir "uv==0.7.11"

# Standalone build context is this repository root.
COPY pyproject.toml README.md uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY server ./server

# Run the service without root privileges. The local artifact fallback and the
# scientific libraries' caches remain writable; application code stays root-owned.
RUN groupadd --gid 10001 mcp \
  && useradd --uid 10001 --gid 10001 --home-dir /app --no-create-home \
       --shell /usr/sbin/nologin mcp \
  && mkdir -p /app/artifacts /app/.cache/matplotlib /app/.config /app/.local \
  && chown -R 10001:10001 /app/artifacts /app/.cache /app/.config /app/.local

ENV UV_SYSTEM_PYTHON=1 \
    HOME=/app \
    PYTHONPATH=/app \
    XDG_CACHE_HOME=/app/.cache \
    MPLCONFIGDIR=/app/.cache/matplotlib \
    MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1 \
    CANPEST_ARTIFACTS_DIR=/app/artifacts

USER 10001:10001
EXPOSE 7331

CMD ["/app/.venv/bin/python", "-m", "server.canpest_server"]
