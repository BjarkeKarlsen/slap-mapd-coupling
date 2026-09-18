# Multi-stage build
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 as base

WORKDIR /workspace

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy environment and install
COPY environment.yml requirements.txt ./
RUN pip install -r requirements.txt

# Copy source code
COPY . .
RUN pip install -e .

# Expose TensorBoard port
EXPOSE 6006

# Default command: run training
CMD ["python", "-m", "slap_mapd_coupling.main", "train", "--help"]
