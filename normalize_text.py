import re


def normalize_aviation_text(text: str) -> str:
    # "flight level 150" -> "FL150"
    text = re.sub(r"flight level\s+", r"FL", text, flags=re.IGNORECASE)

    # Слова-цифры в настоящие цифры
    number_words = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
    }
    for word, digit in number_words.items():
        text = re.sub(rf"\b{word}\b", digit, text, flags=re.IGNORECASE)

    # Убираем лишние пробелы
    text = re.sub(r"\s+", " ", text).strip()

    # Всё в верхний регистр (стандарт для авиации)
    return text.upper()
