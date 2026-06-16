FROM python:3.11-slim

WORKDIR /app

# Install system deps for pdfplumber
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create data directories
RUN mkdir -p /app/data/chroma

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s \
    CMD python3 -c "import httpx; httpx.get('http://localhost:8501', timeout=5)" || exit 1

# Default: launch UI
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
