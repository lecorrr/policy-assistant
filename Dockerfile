FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POLICY_ASSISTANT_HOME=/app

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY config ./config
COPY corpus ./corpus
COPY evals ./evals

# uid 1000 is the usual first user on the host, who owns the mounted ./data folder.
RUN useradd --create-home --uid 1000 app && mkdir -p /app/data && chown app:app /app/data
USER app

EXPOSE 8010
CMD ["policy-assistant", "serve", "--host", "127.0.0.1", "--port", "8010"]
