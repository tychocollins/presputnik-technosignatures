FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the pipeline code and the pre-verified strictly-sunlit sample (74 KB).
# Shadow plates and exposures.parquet are intentionally excluded from this image.
COPY src/ ./src/
COPY data/processed/sunlit_m8_sample.parquet ./data/processed/sunlit_m8_sample.parquet

# Run ONLY the strictly-sunlit m_flash >= 8.0 processor.
CMD ["python", "-u", "-m", "src.process_sunlit_m8"]
