import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from normalize_text import normalize_aviation_text


def main():
    model_id = "./whisper-aviation-model-medium-final"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Using device: {device}")
    print(f"Loading model from: {model_id}")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
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

    test_audio_path = "testradiocall.wav"
    print(f"Transcribing: {test_audio_path}")

    result = pipe(test_audio_path)
    
    raw_text = result["text"].strip()
    normalized_text = normalize_aviation_text(raw_text)
    
    print(f"RAW RESULT:     {raw_text}")
    print(f"NORMALIZED:     {normalized_text}")
    print(f"FLIGHT LEVELS:  {'FL' in normalized_text}")
    print(f"DIGITS PRESENT: any(c in normalized_text for c in '0123456789')")
    print(f"Words count:    {len(normalized_text.split())}")

if __name__ == "__main__":
    main()
