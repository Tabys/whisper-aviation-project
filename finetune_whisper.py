import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorSpeechSeq2SeqWithPadding,
    set_seed,
)
import evaluate

from prepare_dataset_hf import load_and_normalize


def main():
    set_seed(42)

    # ===== 1. ЗАГРУЗКА ДАТАСЕТА =====
    print("=" * 50)
    print("STEP 1: Loading and preparing dataset...")
    print("=" * 50)
    dataset_dict, text_column = load_and_normalize()

    # ===== 2. ЗАГРУЗКА МОДЕЛИ И ПРОЦЕССОРА =====
    print("=" * 50)
    print("STEP 2: Loading Whisper model and processor...")
    print("=" * 50)
    model_id = "openai/whisper-small"
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)

    # Отключаем принудительные токены (нужно для fine-tuning)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # ===== 3. ПОДГОТОВКА ДАТАСЕТА ДЛЯ МОДЕЛИ =====
    print("=" * 50)
    print("STEP 3: Preparing dataset for model...")
    print("=" * 50)

    def prepare_batch(batch):
        # Извлекаем аудио-массивы и частоту дискретизации
        audio_arrays = [a["array"] for a in batch["audio"]]
        sampling_rate = batch["audio"][0]["sampling_rate"]

        # Преобразуем аудио в формат, понятный Whisper
        inputs = processor(
            audio_arrays,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        # Преобразуем текст в токены (числа)
        labels = processor.tokenizer(batch[text_column]).input_ids

        return {
            "input_features": inputs.input_features,
            "labels": labels,
        }

    print("Preparing train dataset...")
    train_dataset = dataset_dict["train"].map(
        prepare_batch,
        remove_columns=dataset_dict["train"].column_names,
        batched=True,
        num_proc=4,
    )

    print("Preparing validation dataset...")
    val_dataset = dataset_dict["validation"].map(
        prepare_batch,
        remove_columns=dataset_dict["validation"].column_names,
        batched=True,
        num_proc=4,
    )

    # ===== 4. НАСТРОЙКА ОБУЧЕНИЯ =====
    print("=" * 50)
    print("STEP 4: Setting up training...")
    print("=" * 50)

    # Data Collator — собирает батчи одинаковой длины
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Метрика WER (Word Error Rate) — чем ниже, тем лучше
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Заменяем -100 на pad_token_id (технический нюанс)
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # Декодируем числа обратно в текст
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        # Считаем WER
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # Параметры обучения
    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-aviation-model",       # куда сохранять чекпоинты
        per_device_train_batch_size=16,               # размер батча (уменьши до 8, если не хватит памяти GPU)
        per_device_eval_batch_size=8,                 # размер батча при оценке
        gradient_accumulation_steps=2,                # накопление градиентов
        learning_rate=1e-5,                           # скорость обучения
        warmup_steps=500,                             # разогрев
        max_steps=4000,                               # сколько шагов обучения
        evaluation_strategy="steps",                  # оценка каждые N шагов
        eval_steps=500,                               # оценка каждые 500 шагов
        save_strategy="steps",                        # сохранение каждые N шагов
        save_steps=500,                               # сохранение каждые 500 шагов
        logging_steps=25,                             # логирование каждые 25 шагов
        report_to=["tensorboard"],                    # логи для TensorBoard
        load_best_model_at_end=True,                  # в конце загрузить лучшую модель
        metric_for_best_model="wer",                  # лучшая = наименьший WER
        greater_is_better=False,                      # меньше WER = лучше
        push_to_hub=False,                            # не загружать на HF Hub
        fp16=True,                                    # обучение в половинной точности (экономит память GPU)
        num_train_epochs=3,                           # максимум эпох
    )

    # ===== 5. ЗАПУСК ОБУЧЕНИЯ =====
    print("=" * 50)
    print("STEP 5: Starting training...")
    print("=" * 50)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.tokenizer,
    )

    trainer.train()

    # ===== 6. СОХРАНЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ =====
    print("=" * 50)
    print("STEP 6: Saving final model...")
    print("=" * 50)

    model.save_pretrained("./whisper-aviation-model-final")
    processor.save_pretrained("./whisper-aviation-model-final")

    print("DONE! Model saved to ./whisper-aviation-model-final")


if __name__ == "__main__":
    main()
