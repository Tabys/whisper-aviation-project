from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os

app = Flask(__name__)
CORS(app)

# Автоматически выбираем GPU или CPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Loading model on {device}...")

# Загружаем обученную модель при запуске сервера
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "./whisper-aviation-model-final",
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
).to(device)

processor = AutoProcessor.from_pretrained("./whisper-aviation-model-final")

asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=0 if device.startswith("cuda") else -1,
)

print("Model loaded! API ready.")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """
    Принимает аудиофайл через POST-запрос, возвращает транскрипцию.
    Пример: curl -X POST -F "audio=@myfile.wav" http://SERVER_IP:5000/transcribe
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided. Send file as 'audio' field."}), 400

    audio_file = request.files["audio"]
    audio_path = f"tmp_{audio_file.filename}"
    audio_file.save(audio_path)

    try:
        result = asr_pipe(audio_path)
        # Удаляем временный файл
        os.remove(audio_path)
        return jsonify({"transcription": result["text"], "status": "success"})
    except Exception as e:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/health", methods=["GET"])
def health():
    """Проверка, что сервер жив."""
    return jsonify({"status": "healthy", "device": device}), 200


if __name__ == "__main__":
    # host=0.0.0.0 — сервер слушает на всех интерфейсах (доступен извне)
    # port=5000 — порт
    app.run(host="0.0.0.0", port=5000, debug=False)
