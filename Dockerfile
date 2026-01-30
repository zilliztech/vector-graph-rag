# ============================================
# Stage 1: Build Frontend
# ============================================
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy package files
COPY frontend/package*.json ./

# Install dependencies
RUN npm ci

# Copy frontend source
COPY frontend/ ./

# Build production bundle
RUN npm run build

# ============================================
# Stage 2: Python Runtime
# ============================================
FROM python:3.11-slim AS runtime

# Build argument for PyTorch backend: cpu, cu124, cu128
ARG TORCH_BACKEND=cpu

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy Python project files
COPY pyproject.toml uv.lock README.md ./

# Modify pyproject.toml to use specified PyTorch backend
# Replace the conditional torch source with explicit backend selection
RUN sed -i '/^\[tool\.uv\.sources\]/,/^$/c\[tool.uv.sources]\ntorch = [{ index = "pytorch-'"${TORCH_BACKEND}"'" }]\n' pyproject.toml

# Install Python dependencies with specified PyTorch backend
# Note: Not using --frozen because uv.lock may have different torch version locked
RUN uv sync --no-dev --extra api

# Copy backend source code
COPY src/ ./src/
COPY api/ ./api/

# Copy frontend build from stage 1
COPY --from=frontend-builder /app/frontend/dist ./static

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Run the application
CMD ["uv", "run", "uvicorn", "vector_graph_rag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
