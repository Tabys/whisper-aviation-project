from flask import Flask, request, jsonify
from flask_cors import CORS

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os

from normalize_text import normalize_aviation_text

app = Flask(__name__)
CORS(app)

# Автоматически выбираем GPU или CPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Loading model on {device}...")

# Загружаем обученную модель
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "./whisper-aviation-model-medium-final",
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
).to(device)

processor = AutoProcessor.from_pretrained("./whisper-aviation-model-medium-final")

asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=0 if device.startswith("cuda") else -1,
    generate_kwargs={
        "max_new_tokens": 64,
        "min_new_tokens": 2,
        "no_repeat_ngram_size": 6,
        "repetition_penalty": 3.0,
        "temperature": 0.05,
        "top_p": 0.7,
        "num_beams": 5,
        "early_stopping": True,
        "length_penalty": 0.6,
    }
)

print("Model loaded with optimal aviation params! API ready.")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided. Send file as 'audio' field."}), 400

    audio_file = request.files["audio"]
    audio_path = f"tmp_{audio_file.filename}"
    audio_file.save(audio_path)

    try:
        result = asr_pipe(audio_path)
        
        # Нормализуем результат
        raw_text = result["text"].strip()
        normalized_text = normalize_aviation_text(raw_text)
        
        os.remove(audio_path)
        
        return jsonify({
            "transcription_raw": raw_text,
            "transcription_norm": normalized_text,
            "status": "success"
        })
        
    except Exception as e:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "device": device}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
