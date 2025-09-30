FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r search && useradd -r -g search search

# Copy base MCP foundation files first
COPY basemcp/ ./basemcp/


# Copy pyproject.toml for package installation
COPY search/pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir -e .


# Copy application code
COPY search/ .
# Create logs directory
RUN mkdir -p /app/logs && chown -R search:search /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Switch to non-root user
USER search

# Expose port
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

# Run the server
CMD ["python", "server.py"]