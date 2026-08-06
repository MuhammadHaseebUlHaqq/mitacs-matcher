FROM python:3.12-slim

# Hugging Face Spaces runs the container as uid 1000. The app writes parsed
# documents to data/kb/ at runtime, so the tree has to be owned by that user —
# running as root instead would work but leaves root-owned files behind.
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Requirements first: this layer is cached unless the deps themselves change,
# so a code-only push rebuilds in seconds rather than reinstalling everything.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

# Shared instance, so the per-process fan-out ceiling is lower than the local
# default of 8 — several concurrent users each open their own batch of calls.
ENV MAX_CONCURRENCY=4

# Render injects $PORT and expects the process to honour it; 7860 is the
# fallback, which is also what a Hugging Face Space would expect. Shell form,
# not exec form, so ${PORT} is actually expanded rather than passed literally.
# Binding 0.0.0.0 rather than app.py's local-dev 127.0.0.1 is what makes the
# container reachable from outside.
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
