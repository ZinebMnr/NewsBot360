import os
import time
import json
import re
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase = create_client(supabase_url, supabase_key)

client = OpenAI()

LABELS = [
  "POLITIQUE",
  "ECONOMIE",
  "ENTREPRISES",
  "TECHNOLOGIE",
  "SCIENCE_SANTE",
  "SOCIETE",
  "FAITS_DIVERS_JUSTICE",
  "SPORT",
  "CULTURE_LOISIRS",
  "AUTRE_INDETERMINE"
]

def _extract_text(resp) -> str:
    txt = getattr(resp, "output_text", None)
    if txt and txt.strip():
        return txt.strip()

    try:
        chunks = []
        for item in resp.output:
            for c in item.content:
                t = getattr(c, "text", None)
                if t:
                    chunks.append(t)
        txt2 = "\n".join(chunks).strip()
        if txt2:
            return txt2
    except Exception:
        pass

    return ""

def _extract_first_json_object(text: str) -> str:
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return m.group(0) if m else ""

def recup_title(limit=300):
    resp = (
        supabase
        .table("articles")
        .select("id,title,label")
        .range(0, limit - 1)
        .execute()
    )
    return pd.DataFrame(resp.data)

def label_title(title: str) -> dict:
    prompt = f"""
Tu es une API de classification de titres de presse.

Choisis EXACTEMENT UN label parmi cette liste :
{LABELS}

Règles :
- Réponds UNIQUEMENT avec un objet JSON (pas de markdown, pas de texte)
- Format EXACT :
{{"label":"<label>"}}

- Si ambigu ou vague : AUTRE_INDETERMINE

Titre : {title}
""".strip()

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    text = _extract_text(resp)
    if not text:
        raise RuntimeError("Réponse OpenAI vide: impossible d'extraire du texte.")

    json_str = _extract_first_json_object(text)
    if not json_str:
        raise RuntimeError(f"Réponse non-JSON:\n{text}")

    data = json.loads(json_str)

    # garde-fou: si le modèle hallucine un label hors liste
    lab = data.get("label")
    if lab not in LABELS:
        data["label"] = "AUTRE_INDETERMINE"

    return data

def update_label_in_supabase(article_id: int, new_label: str):
    return (
        supabase
        .table("articles")
        .update({"label": new_label})
        .eq("id", article_id)
        .execute()
    )

def main(limit=300, only_if_empty=True, sleep_s=0.2):
    df = recup_title(limit=limit)

    updated = 0
    failed = 0

    for _, row in df.iterrows():
        article_id = row["id"]
        title = row["title"]
        current_label = row.get("label")

        if only_if_empty and current_label not in (None, "", "AUTRE_INDETERMINE"):
            # si tu veux relabel TOUT, mets only_if_empty=False
            continue

        try:
            pred = label_title(title)
            new_label = pred["label"]

            update_label_in_supabase(article_id, new_label)
            updated += 1
            print(f"[OK] id={article_id} -> {new_label} | {title}")

            if sleep_s:
                time.sleep(sleep_s)

        except Exception as e:
            failed += 1
            print(f"[ERR] id={article_id} | {title}\n   {e}")

    print(f"\nTerminé. Updated={updated}, Failed={failed}")
    return updated, failed

if __name__ == "__main__":
    main(limit=300, only_if_empty=True, sleep_s=0.2)
