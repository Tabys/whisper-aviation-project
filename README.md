# Whisper Aviation ASR ✈️🎙️

Заготовка для дообучения модели **Whisper-medium** на авиационных радиопереговорах (Air Traffic Control).  
Репозиторий содержит полный пайплайн: подготовка датасета → дообучение → оценка качества (WER/CER) → Flask API для инференса.

> **Важно:** Обученная модель в репозитории **не хранится** (размер ~9 GB). После дообучения она сохраняется локально в папку `./whisper-aviation-model-medium-final/`. Базовая модель `openai/whisper-medium` скачивается автоматически с Hugging Face при запуске обучения.

---

## 🗂 Структура проекта

- `app.py` — Flask API сервер для транскрибации аудио.
- `finetune_whisper.py` — скрипт дообучения Whisper на датасете Hugging Face.
- `prepare_dataset_hf.py` — загрузка и подготовка датасета (`jacktol/ATC-ASR-Dataset`).
- `normalize_text.py` — кастомный нормализатор авиационного текста (FL, позывные, цифры).
- `evaluate_on_testset.py` — оценка качества дообученной модели (WER/CER).
- `test_finetuned.py` — быстрая проверка инференса модели на одном файле.
- `docker-compose.yml` — конфигурация для развертывания API с поддержкой GPU.
- `requirements.txt` — зависимости проекта.

---

## 💻 Рекомендации по железу

### Для дообучения модели (Fine-Tuning)

- **GPU:** минимум 16 GB VRAM (RTX 4080, RTX 3090, A10G, L4, A100).
- **RAM:** 32 GB и выше.
- **Диск:** 50+ GB SSD (чекпоинты + кэш датасета).
- Скрипт использует `gradient_checkpointing=True` и FP16 — это снижает потребление VRAM, но модель `medium` всё равно требовательна.

### Для инференса (API)

- **Минимум:** GPU 8 GB VRAM (RTX 3060 Ti, RTX 4060, T4) + 16 GB RAM.
- CPU-only вариант возможен, но задержки значительно выше.

### Рекомендуемые облачные провайдеры

| Провайдер | Железо | Примечание |
|---|---|---|
| [RunPod](https://runpod.io) / [Vast.ai](https://vast.ai) | RTX 4090, A100, L4 | Дёшево, оплата по часам |
| AWS (g5 / p3) | A10G / V100 | Стабильно, дорого |
| GCP (G2 / A2) | L4 / A100 | Хорошая интеграция с HF |
| Yandex Cloud / Selectel | T4 / A100 | Для аренды в РФ/СНГ |

---

## 🛠 Установка и подготовка окружения

### 1. Клонирование репозитория

```bash
git clone https://github.com/Tabys/whisper-aviation-project.git
cd whisper-aviation-project
```

### 2. Создание виртуального окружения

```bash
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
```

### 3. Установка зависимостей

Сначала установите PyTorch с поддержкой CUDA (подберите версию под ваш сервер):

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Затем остальные зависимости:

```bash
pip install -r requirements.txt
```

---

## 🧠 Дообучение модели (Fine-Tuning)

### Шаг 1 — Подготовка датасета

Скрипт автоматически скачает `jacktol/ATC-ASR-Dataset` с Hugging Face и применит нормализацию текста:

```bash
python prepare_dataset_hf.py
```

> Датасет загрузится в локальный кэш HF (`~/.cache/huggingface/`). Не добавляйте аудиофайлы в Git.

### Шаг 2 — Запуск обучения

```bash
python finetune_whisper.py
```

Что происходит:
- Автоматически скачивается базовая модель `openai/whisper-medium` с Hugging Face.
- Чекпоинты сохраняются в `./whisper-aviation-model/` каждые 500 шагов.
- При перезапуске обучение **автоматически продолжается** с последнего чекпоинта.
- По завершении финальная модель сохраняется в `./whisper-aviation-model-medium-final/`.

Мониторинг обучения через TensorBoard:

```bash
tensorboard --logdir ./whisper-aviation-model/runs
```

### Шаг 3 — Оценка качества (WER/CER)

```bash
python evaluate_on_testset.py
```

Скрипт выведет:
- WER (Word Error Rate) и CER (Char Error Rate) на 50 тестовых примерах.
- Диагностику длин (REF vs PRED).
- 5 примеров сравнения эталона и предсказания.

### Шаг 4 — Быстрый тест на одном файле

```bash
python test_finetuned.py
```

---

## 📡 Запуск и использование API

> **Перед запуском** убедитесь, что папка `./whisper-aviation-model-medium-final/` существует (модель обучена на шаге выше).

### Вариант A: Локально

```bash
python app.py
```

### Вариант B: Через Docker Compose (рекомендуется для продакшена)

Требуется установленный [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
docker-compose up -d --build
```

API будет доступно по адресу `http://localhost:5000`.

---

### Эндпоинты API

#### `GET /health` — Проверка статуса

```bash
curl http://localhost:5000/health
```

Ответ:
```json
{
  "device": "cuda:0",
  "status": "healthy"
}
```

#### `POST /transcribe` — Транскрибация аудио

Отправьте аудиофайл (wav, mp3, ogg и др.) в поле `audio`:

```bash
curl -X POST -F "audio=@your_audio.wav" http://localhost:5000/transcribe
```

Ответ:
```json
{
  "status": "success",
  "transcription_raw": "Happar cleared 653. Rheader identified. Climb FL310",
  "transcription_norm": "HAPPAR CLEARED 6 5 3 RHEADER IDENTIFIED CLIMB FLIGHT LEVEL 3 1 0"
}
```

- `transcription_raw` — сырой вывод модели.
- `transcription_norm` — нормализованный текст в авиационном формате.

#### Пример запроса на Python

```python
import requests

with open("your_audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:5000/transcribe",
        files={"audio": f}
    )

data = response.json()
print("RAW: ", data["transcription_raw"])
print("NORM:", data["transcription_norm"])
```

---

## 🔧 Мониторинг сервера

Скрипт `monitor.sh` позволяет следить за нагрузкой GPU/CPU во время обучения:

```bash
bash monitor.sh
```

---

## 📦 Зависимости (`requirements.txt`)

```
torch
torchaudio
transformers
datasets
evaluate
accelerate
tensorboard
jiwer
flask
flask-cors
huggingface_hub
pandas
soundfile
pyarrow
numpy
```

---

## 📝 .gitignore (рекомендуется)

```
venv/
__pycache__/
*.pyc
hf_audio/
audio_files/
*.wav
!test_radiocall.wav
.logs/
whisper-aviation-model/
whisper-aviation-model-medium-final/
```

> Обученная модель не хранится в репозитории. После дообучения она сохраняется локально. Если вы хотите поделиться моделью — загрузите её на [Hugging Face Hub](https://huggingface.co/models).
