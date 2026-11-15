FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/mpl

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy frozen pipeline code and the pre-built Phase 2b manifest (~210 KB, embeds APPLAUSE scan URLs).
COPY src/ ./src/
COPY data/processed/phase2b_manifest.parquet ./data/processed/phase2b_manifest.parquet

# Run Phase 2b extraction across Stages A -> B -> C (packaging Core A+B early and full A+B+C at end).
CMD ["python", "-u", "-m", "src.phase2b_extract", "--stages", "A", "B", "C"]
