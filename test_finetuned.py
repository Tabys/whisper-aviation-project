import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def main():
    # Путь к обученной модели (папка, созданная после finetune_whisper.py)
    model_id = "./whisper-aviation-model-final"

    # Автоматически выбираем GPU или CPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Using device: {device}")
    print(f"Loading model from: {model_id}")

    # Загружаем модель
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)

    # Загружаем процессор (токенизатор + feature extractor)
    processor = AutoProcessor.from_pretrained(model_id)

    # Создаём pipeline — удобная обёртка для распознавания
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=0 if device.startswith("cuda") else -1,
    )

    # Распознаём тестовый файл
    # Положи файл testradiocall.wav в папку проекта на сервере
    test_audio_path = "testradiocall.wav"
    print(f"Transcribing: {test_audio_path}")

    result = pipe(test_audio_path)
    print(f"Result: {result['text']}")


if __name__ == "__main__":
    main()
