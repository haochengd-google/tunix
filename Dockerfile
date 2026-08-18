# Base image with Python 3.12
FROM python:3.12-slim

# Set environment variables to non-interactive to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies, including Python 3 and pip
RUN apt-get update && \
    apt-get install -y \
        vim lsof procps \
        apt-transport-https ca-certificates gnupg \
        build-essential curl git python3 python3-pip

# Install gcloud, kubectl, k9s
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | gpg --batch --yes --no-tty --dearmor -o /usr/share/keyrings/cloud.google.gpg
RUN apt-get update && \
    apt-get install -y \
        google-cloud-cli \
        google-cloud-cli-gke-gcloud-auth-plugin \
        kubectl
RUN curl -sS https://webinstall.dev/k9s | bash

RUN rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Create a virtual environment
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

RUN pip install git+https://github.com/ayaka14732/jax-smi.git
# If you encounter a checkpoint issue, try using following old version of pathways-utils.
# RUN pip install git+https://github.com/AI-Hypercomputer/pathways-utils.git@b72729bb152b7b3426299405950b3af300d765a9#egg=pathwaysutils
RUN pip install gcsfs
RUN pip install wandb

# Set the working directory
WORKDIR /app

# Copy the project files to the image
COPY . .

# Install the project in editable mode
RUN pip install -e .

RUN bash /app/scripts/install_tunix_vllm_requirement.sh

# Build argument to conditionally install DeepSWE evaluation dependencies
ARG INSTALL_DEEPSWE_DEPS=false

# Install DeepSWE specific dependencies and apply runtime patches conditionally
RUN if [ "$INSTALL_DEEPSWE_DEPS" = "true" ]; then \
      pip install kubernetes gym swebench==3.0.2 && \
      pip install --no-deps git+https://github.com/r2e-gym/r2e-gym.git@0d94c4eb9431cd195c55a7ea3abd54006c9a1735 && \
      sed -i 's/create_repo, upload_folder, HfFolder/create_repo, upload_folder/' /opt/venv/lib/python3.12/site-packages/r2egym/agenthub/utils/utils.py && \
      sed -i 's/self.commit = ParsedCommit(\*\*json.loads(self.commit_json))/self.commit = ParsedCommit(\*\*(json.loads(self.commit_json) if isinstance(self.commit_json, str) else self.commit_json))/' /opt/venv/lib/python3.12/site-packages/r2egym/agenthub/runtime/docker.py; \
    fi

# --- Raiden weight-sync additions ---
# --- Raiden weight-sync additions ---
ARG JAX_PIN=0.11.0
ARG RAIDEN_WHEEL
RUN pip install "jax==${JAX_PIN}" "jaxlib==${JAX_PIN}" "libtpu==0.0.44" "flax==0.12.8" pathwaysutils
COPY ${RAIDEN_WHEEL} /tmp/
RUN pip install --no-deps --force-reinstall /tmp/*.whl && rm -f /tmp/*.whl
RUN python -c "import jax; assert jax.__version__ == '${JAX_PIN}', jax.__version__" \
 && python -c "import jaxlib; assert jaxlib.__version__ == '${JAX_PIN}', jaxlib.__version__" \
 && python -c "import tunix" \
 && python -c "from tpu_sync.api.jax import weight_synchronizer" \
 && python -c "from tpu_sync.rpc import raiden_controller" \
 && echo "ACCEPT: full stack OK"

# Set the default command to bash
CMD ["bash"]
