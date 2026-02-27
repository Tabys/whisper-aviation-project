import os
import tempfile

import torch
import torchaudio
import torchaudio.transforms as T

from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer, cer

from normalize_text import normalize_aviation_text


def load_audio_torchaudio(path: str, target_sr: int = 16000):
    waveform, sr = torchaudio.load(path)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sr != target_sr:
        waveform = T.Resample(orig_freq=sr, new_freq=target_sr)(waveform)

    return waveform.squeeze(0).numpy(), target_sr


def main():
    print("Loading test dataset...")
    raw = load_dataset("jacktol/ATC-ASR-Dataset")

    if "test" in raw:
        test_ds = raw["test"]
    else:
        splits = raw["train"].train_test_split(test_size=0.1, seed=42)
        test_ds = splits["test"]

    num_samples = min(50, len(test_ds))
    test_subset = test_ds.select(range(num_samples))
    print(f"Evaluating on {num_samples} samples...")

    model_id = "./whisper-aviation-model-medium-final"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Loading model from {model_id}...")
    processor = WhisperProcessor.from_pretrained(model_id)

    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device)

    model.eval()

    # Берём низкоуровневую Arrow-таблицу
    table = test_subset.data  # pyarrow.Table
    audio_col = table.column("audio")

    text_col_name = "text"
    text_list = table.column(text_col_name).to_pylist()

    predictions = []
    references = []
    tmp_paths = []
    
    # ДИАГНОСТИКА: статистика длин
    ref_raw_lens = []
    ref_norm_lens = []
    pred_raw_lens = []
    pred_norm_lens = []

    for i in range(num_samples):
        audio_struct = audio_col[i].as_py()  # обычно dict: {"path": ..., "bytes": ...}
        audio_bytes = None
        audio_path = None

        if isinstance(audio_struct, dict):
            audio_bytes = audio_struct.get("bytes")
            audio_path = audio_struct.get("path")
        elif isinstance(audio_struct, str):
            audio_path = audio_struct

        if audio_bytes:
            suffix = ".wav"
            if isinstance(audio_path, str) and os.path.splitext(audio_path)[1]:
                suffix = os.path.splitext(audio_path)[1]

            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp.write(audio_bytes)
            tmp.flush()
            tmp.close()
            tmp_paths.append(tmp.name)
            path_to_load = tmp.name
        else:
            if not audio_path or not os.path.exists(audio_path):
                raise RuntimeError(
                    f"Audio file not found and no bytes available. audio_path={audio_path!r}"
                )
            path_to_load = audio_path

        audio_array, sr = load_audio_torchaudio(path_to_load, target_sr=16000)

        # Сырой → нормализованный референс
        ref_text_raw = (text_list[i] or "").strip()
        ref_text = normalize_aviation_text(ref_text_raw)
        
        # Сохраняем длины для анализа
        ref_raw_lens.append(len(ref_text_raw.split()))
        ref_norm_lens.append(len(ref_text.split()))

        inputs = processor(audio_array, sampling_rate=sr, return_tensors="pt")
        input_features = inputs.input_features.to(device, dtype=torch_dtype)

        with torch.no_grad():
            generated_ids = model.generate(
                input_features=input_features,
                language="en",
                task="transcribe",
                max_new_tokens=128,           # Датасет короткий!
                min_new_tokens=2,
                no_repeat_ngram_size=6,      # Анти-повторы
                repetition_penalty=4.0,      # Жёсткий анти-спам
                temperature=0.05,            # Детерминировано
                top_p=0.7,
                do_sample=True,
                num_beams=5,                 # Beam search
                early_stopping=True,
                length_penalty=0.6,          # Штраф за длинные ответы
            )

        # Сырой → нормализованный предсказание
        pred_text_raw = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
        pred_text = normalize_aviation_text(pred_text_raw)
        
        # Сохраняем длины
        pred_raw_lens.append(len(pred_text_raw.split()))
        pred_norm_lens.append(len(pred_text.split()))

        predictions.append(pred_text)
        references.append(ref_text)

        if i < 5:
            print(f"\n--- Example {i+1} ---")
            print(f" REF RAW:  {ref_text_raw}")
            print(f" REF NORM: {ref_text}")
            print(f" PRED RAW:  {pred_text_raw}")
            print(f" PRED NORM: {pred_text}")
            print(f" REF words: {len(ref_text.split())} | PRED words: {len(pred_text.split())}")

    # Диагностика
    print(f"\n{'='*60}")
    print("LENGTH ANALYSIS")
    print(f"{'='*60}")
    print(f"REF RAW  avg: {sum(ref_raw_lens)/len(ref_raw_lens):5.1f} words")
    print(f"REF NORM avg: {sum(ref_norm_lens)/len(ref_norm_lens):5.1f} words")
    print(f"PRED RAW avg: {sum(pred_raw_lens)/len(pred_raw_lens):5.1f} words")
    print(f"PRED NORM avg: {sum(pred_norm_lens)/len(pred_norm_lens):5.1f} words")
    print(f"PRED/REF ratio: {sum(pred_norm_lens)/sum(ref_norm_lens):5.2f}")
    print(f"{'='*60}")

    word_error_rate = wer(references, predictions)
    char_error_rate = cer(references, predictions)

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS on {num_samples} samples:")
    print(f" Word Error Rate (WER): {word_error_rate:.4f} ({word_error_rate*100:.1f}%)")
    print(f" Char Error Rate (CER): {char_error_rate:.4f} ({char_error_rate*100:.1f}%)")
    print(f"{'='*60}")

    for p in tmp_paths:
        try:
            os.remove(p)
        except OSError:
            pass


if __name__ == "__main__":
    main()
