import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset, Audio
from jiwer import wer, cer


def main():
    # 1. Загружаем тестовую часть датасета
    print("Loading test dataset...")
    raw = load_dataset("jacktol/ATC-ASR-Dataset")

    # Берём test split (или создаём из train, если нет)
    if "test" in raw:
        test_ds = raw["test"]
    else:
        splits = raw["train"].train_test_split(test_size=0.1, seed=42)
        test_ds = splits["test"]

    test_ds = test_ds.cast_column("audio", Audio(sampling_rate=16000))

    # Берём первые 50 примеров (можно увеличить)
    num_samples = min(50, len(test_ds))
    test_subset = test_ds.select(range(num_samples))
    print(f"Evaluating on {num_samples} samples...")

    # 2. Загружаем обученную модель
    model_id = "./whisper-aviation-model-final"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Loading model from {model_id}...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=0 if device.startswith("cuda") else -1,
    )

    # 3. Прогоняем каждый пример
    predictions = []
    references = []

    # Определяем имя текстовой колонки
    text_column = "text"  # <-- ПОМЕНЯЙ, если поле называется иначе

    for i, sample in enumerate(test_subset):
        audio_array = sample["audio"]["array"]
        sampling_rate = sample["audio"]["sampling_rate"]
        ref_text = sample[text_column]

        result = asr_pipe(
            {"array": audio_array, "sampling_rate": sampling_rate}
        )
        pred_text = result["text"]

        predictions.append(pred_text)
        references.append(ref_text)

        if i < 5:  # показываем первые 5
            print(f"\n--- Example {i+1} ---")
            print(f"  REF:  {ref_text}")
            print(f"  PRED: {pred_text}")

    # 4. Считаем метрики
    word_error_rate = wer(references, predictions)
    char_error_rate = cer(references, predictions)

    print(f"\n{'='*50}")
    print(f"RESULTS on {num_samples} samples:")
    print(f"  Word Error Rate  (WER): {word_error_rate:.4f} ({word_error_rate*100:.1f}%)")
    print(f"  Char Error Rate  (CER): {char_error_rate:.4f} ({char_error_rate*100:.1f}%)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
