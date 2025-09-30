# FileIO MCP Server Dockerfile
FROM python:3.11-slim

LABEL maintainer="n8n Stack"
LABEL description="FileIO MCP Server for n8n workflow file operations"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r mcpuser && useradd -r -g mcpuser mcpuser

# Create necessary directories
RUN mkdir -p /app/logs /app/config && \
    chown -R mcpuser:mcpuser /app

# Copy base MCP foundation files first
COPY basemcp/ ./basemcp/

# Copy source code
COPY fileio/*.py .
COPY fileio/*.json .

# Copy package definition for installation
COPY fileio/pyproject.toml .

# Install the package in development mode (editable)
RUN pip install -e .

# Create logs directory and set permissions
RUN mkdir -p /app/logs && chown -R mcpuser:mcpuser /app

# Switch to non-root user
USER mcpuser

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from models import FileIOConfig; FileIOConfig.load()" || exit 1

# Default command
CMD ["python", "server.py"]