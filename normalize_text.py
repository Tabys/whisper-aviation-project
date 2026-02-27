import re


def normalize_aviation_text(text: str) -> str:
    """
    Нормализация авиационных транскрипций:
    - FL + слова → FL + цифры + пробелы (FLTHREE10 → FL 3 1 0)
    - Flight level → FL
    - Слова → цифры
    - Пробелы перед цифрами после слов
    - RUNWAY31 без пробелов
    - Верхний регистр
    """
    
    # Сохраняем оригинал для дебага
    original = text
    
    # 1. "flight level XXX" → "FL XXX"
    text = re.sub(r"flight level\s+", r"FL ", text, flags=re.IGNORECASE)
    
    # 2. Пробел перед FL если слиплось
    text = re.sub(r"([A-Z]+)FL", r"\1 FL", text)
    
    # 3. СЛОВА → ЦИФРЫ (всегда сначала!)
    number_words = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"
    }
    for word, digit in number_words.items():
        text = re.sub(rf"\b{word}\b", digit, text, flags=re.IGNORECASE)
    
    # 4. ✅ КРИТИЧНО: ВСТАВЛЯЕМ ПРОБЕЛЫ ПЕРЕД ЦИФРАМИ ПОСЛЕ БУКВ
    # WIND120 → WIND 120, FL310 → FL 3 1 0
    text = re.sub(r"([A-Z]{2,})([0-9])", r"\1 \2", text)
    
    # 5. RUNWAY + номер → RUNWAY31 (убираем пробел)
    text = re.sub(r"RUNWAY\s+([0-9]{1,2})\b", r"RUNWAY\1", text, flags=re.IGNORECASE)    
    
    # 6. ВСЕ ОСТАЛЬНЫЕ ПРОБЕЛЫ ПЕРЕД ЦИФРАМИ (после одиночных букв тоже)
    text = re.sub(r"([A-Z])([0-9])", r"\1 \2", text)
    
    # 7. Runway номера подряд → RUNWAY31 (если пробелы уже вставлены)
    text = re.sub(r"RUNWAY\s+([0-9])\s+([0-9])", r"RUNWAY\1\2", text, flags=re.IGNORECASE)
    
    # 8. УБИРАЕМ ЛИШНИЕ ПРОБЕЛЫ
    text = re.sub(r"\s+", " ", text).strip()
    
    # 9. ВСЁ В ВЕРХНИЙ РЕГИСТР (ATC стандарт)
    text = text.upper()
    
    return text


# 🧪 ТЕСТЫ
if __name__ == "__main__":
    tests = [
        "flight level three one zero",                    # → FL 3 1 0
        "FLTHREE10",                                     # → FL 3 1 0
        "WIND one two zero degrees",                     # → WIND 1 2 0 DEGREES
        "WIND120",                                       # → WIND 120
        "RUNWAY THREE ONE",                              # → RUNWAY31
        "RUNWAY 3 1",                                    # → RUNWAY31
        "cleared to FL310",                              # → CLEARED TO FL 3 1 0
        "HOTEL ECHO FOUR FIVE SIX",                      # → HOTEL ECHO 4 5 6
        "standby flight level one five zero",            # → STANDBY FL 1 5 0
    ]
    
    print("=== NORMALIZATION TESTS ===")
    for test in tests:
        result = normalize_aviation_text(test)
        print(f"'{test}' → '{result}'")
    print("="*50)
