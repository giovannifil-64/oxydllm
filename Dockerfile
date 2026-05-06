# ─────────────────────────────────────────────────────────────────────────────
# oxydLLM — CUDA image (Linux x86_64 + arm64, Ada / Hopper / Blackwell / Thor)
#
# Build (x86_64):
#   docker build -t oxydllm:cuda-ada --build-arg CUDA_ARCH=89 .
#   docker build -t oxydllm:cuda-hopper --build-arg CUDA_ARCH=90 .
#   docker build -t oxydllm:cuda-blackwell --build-arg CUDA_ARCH=100 .
#   docker build -t oxydllm:cuda-blackwell-ultra --build-arg CUDA_ARCH=103 .
#   docker build -t oxydllm:cuda-blackwell-consumer --build-arg CUDA_ARCH=120 .
#
# Build (arm64 — DGX Spark / GH200 / GB300 / Jetson Thor):
#   docker buildx build --platform linux/arm64 -t oxydllm:cuda-blackwell-arm64 --build-arg CUDA_ARCH=100 .
#   docker buildx build --platform linux/arm64 -t oxydllm:cuda-thor-arm64 --build-arg CUDA_ARCH=110 .
#
# Run:
#   docker run --gpus all -p 11313:11313 \
#     -v ~/.oxydllm/models:/root/.oxydllm/models \
#     oxydllm:cuda start
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: build ───────────────────────────────────────────────────────────
FROM nvidia/cuda:13.2.1-devel-ubuntu22.04 AS builder

ARG CUDA_ARCH=89

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_COMPUTE_CAP=${CUDA_ARCH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential pkg-config libssl-dev git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pin the Rust toolchain to the same version used by CI (single source of truth:
# rust-toolchain.toml). Copy it BEFORE Cargo manifests so the toolchain-install
# layer is cached independently of dependency/source changes.
COPY rust-toolchain.toml ./
RUN TOOLCHAIN=$(grep '^channel' rust-toolchain.toml | cut -d'"' -f2) && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain "$TOOLCHAIN" --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

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
FROM nvidia/cuda:13.2.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/oxydllm /usr/local/bin/oxydllm
RUN chmod +x /usr/local/bin/oxydllm

VOLUME ["/root/.oxydllm/models"]
EXPOSE 11313

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:11313/health || exit 1

ENTRYPOINT ["oxydllm"]
CMD ["start", "--models-dir", "/root/.oxydllm/models"]
