# Default image for `fal dev` boxes. Use your own with `fal dev deploy --dockerfile`,
# starting from a copy of this one: the box needs the tools installed here.
# Match the CUDA / torch stack your production apps use so benchmarks agree.
# CUDA 12.8 is the first release with Blackwell support (B200, RTX PRO 6000).
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# Set by `fal dev deploy`: the Python that deploys (major.minor, which cloudpickle
# needs to match) and the fal release to install.
ARG PYTHON_VERSION=3.11
ARG FAL_SPEC=fal

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates git git-lfs tmux htop vim less openssh-client openssh-server build-essential \
        zstd rsync \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.19 /uv /uvx /usr/local/bin/

ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_PYTHON_BIN_DIR=/usr/local/bin
RUN uv python install ${PYTHON_VERSION} \
    && ln -sf /usr/local/bin/python${PYTHON_VERSION} /usr/local/bin/python \
    && ln -sf /usr/local/bin/python${PYTHON_VERSION} /usr/local/bin/python3
ARG PIP="uv pip install --system --break-system-packages --python /usr/local/bin/python3"

# Heavy, slow-changing dependencies belong here (fast local disk, cached build).
RUN $PIP torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
RUN $PIP transformers==4.56.1 diffusers==0.35.1 accelerate==1.10.1 \
        safetensors==0.6.2 huggingface_hub==0.34.4

# Claude Code, for `fal dev deploy --claude`. Auto-update is off so the binary
# the server started from stays in place; redeploy to update.
ENV DISABLE_AUTOUPDATER=1
RUN curl -fsSL https://claude.ai/install.sh | bash \
    && ln -sf /root/.local/bin/claude /usr/local/bin/claude

# fal goes last so its pinned pydantic/protobuf versions win. The box's own fal
# code ships with the app; this install provides its dependencies and the CLI.
RUN $PIP "${FAL_SPEC}" \
    && ln -sf "$(dirname "$(readlink -f /usr/local/bin/python3)")/fal" /usr/local/bin/fal
