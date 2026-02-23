FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y python3.10 python3-pip && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY app.py .
COPY whisper-aviation-model-final ./whisper-aviation-model-final

EXPOSE 5000

CMD ["python3", "app.py"]
