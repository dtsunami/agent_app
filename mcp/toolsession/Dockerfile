FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy base MCP foundation files first
COPY basemcp/ ./basemcp/

# Copy pyproject.toml for package installation
COPY toolsession/pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Copy application code
COPY toolsession/ .


# Create directories for logs and sessions
RUN mkdir -p /app/logs

# Create non-root user for security
RUN useradd -m -u 1000 toolsession && \
    chown -R toolsession:toolsession /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

USER toolsession

# Health check (check HTTP health endpoint)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Default command - launch ToolSession MCP server
CMD ["python", "server.py"]