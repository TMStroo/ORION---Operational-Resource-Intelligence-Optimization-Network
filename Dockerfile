# ORION - Operational Resource Intelligence & Optimization Network
#
# One image serves three roles so that a reviewer needs one build:
#   backend   the FastAPI application (default)
#   tests     the full test suite
#   demo      the end-to-end demonstration
#
# The build context deliberately excludes .git (see .dockerignore), so the
# commit is supplied as a build argument. Without it the image would report an
# unknown commit and every experiment manifest it wrote would be unattributable.

# ---------------------------------------------------------------- build stage
FROM python:3.12-slim AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# or-tools, scipy and pandas need a compiler only if no wheel matches; the
# pinned build stage below resolves wheels for this interpreter. build-essential
# is kept because scikit-style transitive deps occasionally fall back to source.
RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY pyproject.toml README.md ./
COPY backend/src ./backend/src

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel \
    && pip install ".[dev]"

# ----------------------------------------------------------------- test stage
# Tests run from the build stage so a failure fails the build, not a later
# `docker compose run`.
FROM build AS test
# The suite reads the stored experiment artifacts and the demo results: the
# evidence layer refuses to invent a number, so a container without them would
# either skip those checks or, worse, pass vacuously. Omitting them was a real
# failure found by running the suite in the image.
COPY tests ./tests
COPY scripts ./scripts
COPY configs ./configs
COPY experiments ./experiments
COPY results ./results
COPY docs ./docs
ENV PYTHONPATH=/src/backend/src
CMD ["python", "-m", "pytest", "-q"]

# --------------------------------------------------------------- runtime stage
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="ORION" \
      org.opencontainers.image.description="Operational Resource Intelligence & Optimization Network" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/backend/src \
    ORION_DATABASE_URL="sqlite:////data/orion.db" \
    ORION_GIT_COMMIT="unknown"

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 orion

COPY --from=build /opt/venv /opt/venv

WORKDIR /app
COPY backend/src ./backend/src
COPY configs ./configs
COPY experiments ./experiments
COPY results ./results
COPY scripts ./scripts
COPY pyproject.toml README.md ./

RUN mkdir -p /data && chown -R orion:orion /app /data
USER orion
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "orion.api.asgi:app", "--host", "0.0.0.0", "--port", "8000"]
