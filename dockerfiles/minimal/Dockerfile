# =============================================================================
# quant-insight Minimal Environment Dockerfile
# =============================================================================
# mixseekプラグインとして利用するための最小構成イメージ。
#
# Build Args:
#   MIXSEEK_USERNAME: コンテナユーザー名 (default: quant_insight)
#   MIXSEEK_UID: ユーザーID (default: 1000)
#   MIXSEEK_GID: グループID (default: 1000)

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# =============================================================================
# BUILD ARGUMENTS
# =============================================================================

ARG MIXSEEK_USERNAME=quant_insight
ARG MIXSEEK_UID=1000
ARG MIXSEEK_GID=1000

# =============================================================================
# SYSTEM PACKAGES
# =============================================================================

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
        cmake \
        pkg-config \
        libprotobuf-dev \
        protobuf-compiler \
        locales \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# LOCALE
# =============================================================================

RUN sed -i -e 's/# ja_JP.UTF-8 UTF-8/ja_JP.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen

ENV LANG=ja_JP.UTF-8
ENV LC_ALL=ja_JP.UTF-8
ENV LANGUAGE=ja_JP:ja

# =============================================================================
# USER SETUP
# =============================================================================

RUN groupadd -g ${MIXSEEK_GID} ${MIXSEEK_USERNAME} && \
    useradd -m -u ${MIXSEEK_UID} -g ${MIXSEEK_GID} -s /bin/bash ${MIXSEEK_USERNAME}

RUN mkdir -p /app /venv && \
    chown -R ${MIXSEEK_USERNAME}:${MIXSEEK_USERNAME} /app /venv

# =============================================================================
# PYTHON ENVIRONMENT
# =============================================================================

USER ${MIXSEEK_USERNAME}
WORKDIR /app

ENV PYTHONPATH="/app"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV VIRTUAL_ENV=/venv
ENV PATH="/venv/bin:${PATH}"

ENV UV_HTTP_TIMEOUT=120
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/venv

# =============================================================================
# DEPENDENCIES
# =============================================================================

COPY --chown=${MIXSEEK_USERNAME}:${MIXSEEK_USERNAME} pyproject.toml uv.lock README.md /app/

RUN --mount=type=cache,target=/home/${MIXSEEK_USERNAME}/.cache/uv,uid=${MIXSEEK_UID},gid=${MIXSEEK_GID} \
    uv sync --frozen --no-install-project

COPY --chown=${MIXSEEK_USERNAME}:${MIXSEEK_USERNAME} src/ /app/src/

RUN --mount=type=cache,target=/home/${MIXSEEK_USERNAME}/.cache/uv,uid=${MIXSEEK_UID},gid=${MIXSEEK_GID} \
    uv sync --frozen

# =============================================================================
# SECURITY
# =============================================================================

USER root
RUN find /usr/bin /usr/local/bin -perm /4000 -exec chmod -s {} + 2>/dev/null || true
USER ${MIXSEEK_USERNAME}

# =============================================================================
# FINAL
# =============================================================================

WORKDIR /app

CMD ["bash"]
