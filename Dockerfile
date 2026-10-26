FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 PORT=8000 \
    ORACULO_DB=/data/oraculo.db ORACULO_CACHE=/data/cache

WORKDIR /app
COPY pyproject.toml README.md ./
COPY oraculo ./oraculo
RUN pip install --no-cache-dir . && useradd --create-home oraculo && mkdir -p /data && chown oraculo /data
USER oraculo

EXPOSE 8000
HEALTHCHECK CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/healthz')"
CMD ["oraculo-b3"]
