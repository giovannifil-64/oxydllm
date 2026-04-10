# ─────────────────────────────────────────────────────────────────────────────
# rLLM — CUDA image (Linux x86_64, NVIDIA Ada Lovelace+)
#
# Build:
#   docker build -t rllm:cuda --build-arg CUDA_ARCH=89 .
#
# Run:
#   docker run --gpus all -p 11313:11313 \
#     -v ~/.rllm/models:/root/.rllm/models \
#     rllm:cuda start
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: build ───────────────────────────────────────────────────────────
FROM nvidia/cuda:12.9.0-devel-ubuntu22.04 AS builder

ARG CUDA_ARCH=89

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_COMPUTE_CAP=${CUDA_ARCH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential pkg-config libssl-dev git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Copy manifests first for dependency-layer caching
COPY Cargo.toml Cargo.lock ./

# Stub build: compile deps only, then remove stub so real build re-links.
# The `|| true` swallows the expected link failure (no real main.rs yet).
RUN mkdir -p src && echo 'fn main(){}' > src/main.rs \
    && cargo build --release --no-default-features --features cuda 2>/dev/null || true \
    && rm -rf src

# Real build
COPY src ./src
RUN touch src/main.rs \
    && cargo build --release --no-default-features --features cuda

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
# Same CUDA + Ubuntu version as builder to minimize runtime compatibility risk.
FROM nvidia/cuda:12.9.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/rllm /usr/local/bin/rllm
RUN chmod +x /usr/local/bin/rllm

VOLUME ["/root/.rllm/models"]
EXPOSE 11313

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:11313/health || exit 1

ENTRYPOINT ["rllm"]
CMD ["start", "--models-dir", "/root/.rllm/models"]
