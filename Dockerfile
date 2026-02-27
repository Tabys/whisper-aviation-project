FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
WORKDIR /app
RUN apt-get update && apt-get install -y python3.10 python3-pip ffmpeg && \
    rm -rf /var/lib/apt/lists/* && apt-get clean
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py normalize_text.py ./
EXPOSE 5000
CMD ["python3", "app.py"]
