from datasets import load_dataset, DatasetDict, Audio
from normalize_text import normalize_aviation_text


def load_and_normalize():
    """
    Загружает датасет jacktol/ATC-ASR-Dataset с Hugging Face,
    нормализует текст и приводит аудио к 16 kHz.
    """
    # 1. Скачиваем датасет (может занять несколько минут)
    print("Downloading ATC ASR Dataset from Hugging Face...")
    raw = load_dataset("jacktol/ATC-ASR-Dataset")
    print("Download complete!")
    print(raw)

    # 2. Собираем в словарь train / validation / test
    #    Если в датасете нет какого-то сплита, нужно будет
    #    сделать split вручную (см. комментарий ниже)
    if "validation" in raw and "test" in raw:
        dataset = DatasetDict(
            train=raw["train"],
            validation=raw["validation"],
            test=raw["test"],
        )
    else:
        # Если сплитов нет — делаем сами из train
        print("No validation/test splits found. Creating them...")
        train_test = raw["train"].train_test_split(test_size=0.2, seed=42)
        temp = train_test["test"].train_test_split(test_size=0.5, seed=42)
        dataset = DatasetDict(
            train=train_test["train"],
            validation=temp["train"],
            test=temp["test"],
        )

    # 3. Нормализуем авиационный текст
    #    Поле с текстом может называться "text", "sentence" или "transcript"
    #    Проверь вывод print(raw) выше и поменяй, если нужно
    text_column = "text"  # <-- ПОМЕНЯЙ, если поле называется иначе

    def normalize_batch(batch):
        return {text_column: [normalize_aviation_text(t) for t in batch[text_column]]}

    print("Normalizing text...")
    for split_name in dataset:
        dataset[split_name] = dataset[split_name].map(
            normalize_batch, batched=True, batch_size=100
        )

    # 4. Указываем, что колонка "audio" — это аудио, и приводим к 16 kHz
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    return dataset, text_column


if __name__ == "__main__":
    ds, col = load_and_normalize()
    print(ds)
    print(f"Train:      {len(ds['train'])} examples")
    print(f"Validation: {len(ds['validation'])} examples")
    print(f"Test:       {len(ds['test'])} examples")
    print(f"Text column: '{col}'")
    # Покажем первый пример
    print("First example text:", ds["train"][0][col])
