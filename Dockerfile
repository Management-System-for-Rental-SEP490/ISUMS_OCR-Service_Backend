FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12=3.12.3-1ubuntu0.12 \
    python3-pip=24.0+dfsg-1ubuntu1.3 \
    libglib2.0-0t64=2.80.0-6ubuntu3.8 \
    libsm6=2:1.2.3-1build3 \
    libxrender1=1:0.9.10-1.1build1 \
    libxext6=2:1.3.4-1build2 \
    libgl1=1.7.0-1build1 \
    libgomp1=14.2.0-4ubuntu2~24.04.1 \
    curl=8.5.0-2ubuntu10.8 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --break-system-packages -r requirements.txt \
    && python -c "from paddleocr import PaddleOCR; PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, device='cpu')"

COPY main.py .

EXPOSE 9000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]
