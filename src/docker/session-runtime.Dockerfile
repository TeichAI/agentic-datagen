FROM ubuntu:24.04

COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /uvx /usr/local/bin/

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/usr/local/bin:$PATH
ENV UV_LINK_MODE=copy
ENV UV_CACHE_DIR=/workspace/.agent-home/.cache/uv
WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        gnupg \
        build-essential \
        make \
        g++ \
        pkg-config \
        python3 \
        python-is-python3 \
        python3-pip \
        python3-venv \
        ripgrep \
        sqlite3 \
        libsqlite3-dev \
        xz-utils \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_22.x nodistro main" > /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && node --version \
    && npm --version \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/.agent-home /workspace/.agent-home/.cache/uv

ENV HOME=/workspace/.agent-home
CMD ["tail", "-f", "/dev/null"]
