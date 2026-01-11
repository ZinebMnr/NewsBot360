import re
from typing import Optional

def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()

def stars_to_binary(label: str) -> str:
    stars = int(label[0])
    if stars <= 2:
        return "NEGATIVE"
    if stars == 3:
        return "NEUTRAL"
    return "POSITIVE"