FROM python:3.11-slim
RUN useradd -m -u 1000 runner
WORKDIR /app
COPY requirements.server.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements.server.txt -r requirements.txt
COPY . .
ENV HF_HOME=/opt/cache
RUN mkdir -p /opt/cache && python prefetch.py && chmod -R a+rX /opt/cache
# Offline env vars go AFTER prefetch so downloads work at build time but fail loudly at runtime.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
# The official runtime root filesystem is read-only; only /tmp is writable.
ENV HOME=/tmp TMPDIR=/tmp XDG_CACHE_HOME=/tmp/.cache PYTHONDONTWRITEBYTECODE=1
USER runner
EXPOSE 8080
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--no-access-log", "--log-level", "warning"]
