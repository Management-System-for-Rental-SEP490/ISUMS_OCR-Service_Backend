FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    FLAGS_use_mkldnn=False

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3-pip \
    libglib2.0-0t64 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3

WORKDIR /app

COPY requirements.txt .

# Model initialization happens at container startup; doing it during image
# build makes CI depend on Paddle's remote model registry availability.
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY main.py ocr_worker.py ./

EXPOSE 9000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]
