# One image serves the main FastAPI control plane.
# ``slim`` removes compilers and OS utilities the runtime does not need, reducing
# download size and attack surface while retaining normal CPython compatibility.
FROM python:3.12-slim
# Do not create .pyc files in the container; flush logs immediately; avoid a pip
# cache layer that would only make the final image larger.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# Every subsequent relative COPY/CMD path is resolved under /app.
WORKDIR /app
# Copy dependency metadata before source code so Docker can reuse this expensive
# installation layer when application code changes but dependencies do not.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Copy only files required by the API image. .dockerignore excludes local secrets,
# notebooks, tests, slides, caches, and runtime databases from build context.
COPY app app
COPY data data
COPY scripts scripts
# Running as an unprivileged user limits damage if application code is exploited.
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser
# EXPOSE documents the listening port; Docker Compose performs actual publishing.
EXPOSE 8000
# Exec-form CMD forwards termination signals directly to Uvicorn for clean shutdown.
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
