# Base image with CUDA 12.4
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# Install dependencies
RUN apt-get update -y && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6

# Define environment variables for UID and GID
ENV UID=${UID:-1000}
ENV GID=${GID:-1000}
ENV OPENAI_API_KEY=${OPENAI_API_KEY}

# Create a group with the specified GID
RUN groupadd -g "${GID}" appuser
# Create a user with the specified UID and GID
RUN useradd -m -s /bin/sh -u "${UID}" -g "${GID}" appuser

WORKDIR /app

# Fix NumPy version for compatibility with OpenCV

# Install OpenCV first

# Copy the current directory (sd-scripts) to the container
COPY . /app/

# Install dependencies from the existing requirements.txt
RUN pip install --no-cache-dir -r ./requirements.txt
RUN pip install --no-cache-dir numpy==1.26.4
RUN pip install --no-cache-dir opencv-python==4.9.0.80

# Install Torch, Torchvision, and Torchaudio for CUDA 12.4
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Fix triton version
RUN pip install --upgrade --force-reinstall triton==2.1.0

# Create directory structure with proper permissions
RUN mkdir -p /app/models/unet \
    /app/models/clip \
    /app/models/vae \
    /app/datasets \
    /app/outputs \
    && chmod -R 777 /app/models \
    && chmod -R 777 /app/datasets \
    && chmod -R 777 /app/outputs

# Copy our training script and library
COPY train_model.py /app/
COPY library/ /app/library/

# Define volumes to persist data across container lifecycles
VOLUME ["/app/models", "/app/outputs", "/app/datasets"]

# Set environment variables
ENV PYTHONPATH=/app

# Make directories accessible to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Make train_model.py executable
RUN chmod +x /app/train_model.py

# Set the entrypoint to run the script
ENTRYPOINT ["python3", "/app/train_model.py"]

# Default CMD arguments that can be overridden
CMD ["--help"]