# Completen este Dockerfile siguiendo la sección 3.1 del notebook.
#
# Instrucciones que van a necesitar: FROM, COPY, ENV, WORKDIR, RUN, USER,
# EXPOSE, HEALTHCHECK y CMD.

FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /usr/local/bin/uv

# Sin caché de uv en la capa; el entorno virtual queda en el PATH.
ENV UV_NO_CACHE=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Primero las dependencias (capa que se reutiliza si no cambian)...
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --only-group servicio --no-install-project

# ...y recién después el código y el modelo.
COPY src ./src
COPY models ./models

# Usuario sin privilegios.
RUN useradd --system --no-create-home aguaapr
USER aguaapr

EXPOSE 8000 8501

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["bash", "-c", "uvicorn src.aguaapr.main:app --host 0.0.0.0 --port 8000 & streamlit run src/aguaapr/ui.py --server.port 8501 --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false & wait -n"]
