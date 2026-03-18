ARG ROCM_PYTORCH_VERSION

FROM rocm/pytorch:${ROCM_PYTORCH_VERSION}

RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir --no-deps moshi==0.2.13

CMD ["tail", "-f", "/dev/null"]
