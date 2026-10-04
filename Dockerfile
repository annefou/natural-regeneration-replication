FROM ghcr.io/prefix-dev/pixi:0.68.1

LABEL org.opencontainers.image.source="https://github.com/annefou/natural-regeneration-replication"
LABEL org.opencontainers.image.description="Replication study container for natural-regeneration-replication"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app

# Install the pinned environment first (separate from source copy so the lock
# layer is cached across source-only edits).
# git is needed by pixi to fetch the GitHub-pinned PyPI dependency
# (healpix-connector @ v0.1.0); the pixi base image (Debian) does not ship it.
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY pixi.toml pixi.lock /app/
RUN pixi install --locked

COPY . /app

# The full run needs a NASA Earthdata login (SRTM slope; MODIS burned-area
# fallback), tens of GB of disk and several hours. Mount the credentials and a
# data directory at runtime, e.g.:
#   docker run -v ~/.netrc:/root/.netrc:ro -v $PWD/data:/app/data \
#     ghcr.io/annefou/natural-regeneration-replication:latest \
#     pixi run snakemake --cores 12
# Small end-to-end test: ... pixi run snakemake --cores 4 --config smoke=1

CMD ["pixi", "run", "snakemake", "--cores", "1"]
