import os
import functools
import numpy
import torch

# ✅ Меньше фрагментации VRAM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ✅ ФИКС PyTorch 2.6: патчим torch.load для resume checkpoint
_original_torch_load = torch.load
@functools.wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torchaudio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    set_seed,
)
import evaluate
from transformers.trainer_utils import get_last_checkpoint

from prepare_dataset_hf import load_and_normalize
from normalize_text import normalize_aviation_text


@dataclass
class SimpleSpeechSeq2SeqCollator:
    processor: WhisperProcessor
    decoder_start_token_id: Optional[int] = None
    text_column: str = "text"

    def load_audio_from_path(self, path: str):
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform.squeeze(0).numpy(), sr

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_arrays = []
        sampling_rate = None
        labels_list = []

        for f in features:
            arr, sr = self.load_audio_from_path(f["audio_path"])
            audio_arrays.append(arr)
            sampling_rate = sr

            label_tokens = self.processor.tokenizer(f[self.text_column]).input_ids
            labels_list.append(label_tokens)

        inputs = self.processor(
            audio_arrays,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        input_features = inputs.input_features

        label_tensors = [torch.tensor(lbl, dtype=torch.long) for lbl in labels_list]
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            label_tensors,
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id,
        )

        padding_mask = labels_padded == self.processor.tokenizer.pad_token_id
        labels_padded[padding_mask] = -100

        return {
            "input_features": input_features,
            "labels": labels_padded,
        }


class WhisperSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Кастомный Trainer: language='en' жёстко задан + анти-зацикливание.
    """

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss

            if prediction_loss_only:
                return (loss, None, None)

            generated_tokens = model.generate(
                input_features=inputs["input_features"],
                language="en",
                task="transcribe",
                max_new_tokens=128,
                min_new_tokens=2,
                no_repeat_ngram_size=6,
                repetition_penalty=3.0,
                temperature=0.0,
                num_beams=5,
                early_stopping=True,
                length_penalty=0.6,
                do_sample=False,
            )

        labels = inputs.get("labels")
        return (loss, generated_tokens, labels)


def main():
    set_seed(42)

    print("=" * 50)
    print("STEP 1: Loading dataset (already extracted)...")
    print("=" * 50)
    dataset_dict, text_column = load_and_normalize()

    print("=" * 50)
    print("STEP 2: Loading Whisper model and processor...")
    print("=" * 50)

    model_id = "openai/whisper-medium"  # ✅ medium многоязычная

    processor = WhisperProcessor.from_pretrained(
        model_id,
        language="english",             # ✅ EN датасет сейчас
        task="transcribe"               # ✅ В будущем можно сменить на "ru"
    )

    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False      # ✅ ОБЯЗАТЕЛЬНО с gradient_checkpointing!

    print("=" * 50)
    print("STEP 3: Audio will be loaded on-the-fly during training.")
    print("=" * 50)

    train_dataset = dataset_dict["train"]
    val_dataset = dataset_dict["validation"]

    print("=" * 50)
    print("STEP 4: Setting up training...")
    print("=" * 50)

    data_collator = SimpleSpeechSeq2SeqCollator(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        text_column=text_column,
    )

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str_raw = processor.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True
        )
        label_str_raw = processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )

        # ✅ Нормализация для метрики
        pred_str = [normalize_aviation_text(s) for s in pred_str_raw]
        label_str = [normalize_aviation_text(s) for s in label_str_raw]

        # Первые 3 примера
        for i in range(min(3, len(pred_str))):
            print(f" REF RAW:  {label_str_raw[i]}")
            print(f" REF NORM: {label_str[i]}")
            print(f" PRED RAW:  {pred_str_raw[i]}")
            print(f" PRED NORM: {pred_str[i]}")
            print()

        wer_value = wer_metric.compute(predictions=pred_str, references=label_str)
        print(f"\n[Eval] Word Error Rate (WER): {wer_value * 100:.2f}%\n")

        return {"wer": wer_value}

    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-aviation-model-medium",
        per_device_train_batch_size=2,       # ✅ P100 16GB
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,      # ✅ Итоговый batch = 32
        learning_rate=5e-6,                  # ✅ Меньше LR для medium
        warmup_steps=500,
        max_steps=8000,                      # ✅ Больше шагов для medium
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        fp16=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        predict_with_generate=True,
        generation_max_length=64,            # ✅ Короткие ATC фразы
        gradient_checkpointing=True,         # ✅ КЛЮЧЕВОЙ ФИКС OOM!
    )

    print("=" * 50)
    print("STEP 5: Starting training...")
    print("=" * 50)

    trainer = WhisperSeq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.tokenizer,
    )

    last_checkpoint = get_last_checkpoint("./whisper-aviation-model-medium")
    if last_checkpoint is not None:
        print(f"Resuming training from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("No checkpoint found. Starting from scratch.")
        trainer.train()

    print("=" * 50)
    print("STEP 6: Saving final model...")
    print("=" * 50)

    model.save_pretrained("./whisper-aviation-model-medium-final")
    processor.save_pretrained("./whisper-aviation-model-medium-final")

    print("DONE! Model saved to ./whisper-aviation-model-medium-final")


if __name__ == "__main__":
    main()
